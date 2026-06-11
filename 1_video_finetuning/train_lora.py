"""
LoRA Training Script for I2V on LIBERO Dataset

Fine-tunes the Hummingbird I2V model (LatentVisualDiffusion) on robot
manipulation videos using LoRA. The conditioning image is always the
first frame — no random frame placement.
"""

import os
import sys
import argparse
import yaml
import random
from pathlib import Path
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hum_infer"))

from omegaconf import OmegaConf

from dataset import LiberoI2VDataset, create_dataloader
from dataset_all_clips import (
    LiberoI2VDatasetAllClips,
    create_dataloader as create_dataloader_all_clips,
)
from lora_utils import (
    get_lora_config,
    apply_lora_to_model,
    prepare_i2v_model_for_lora,
    get_trainable_parameters,
    save_lora_weights,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_i2v_model(config: Dict, device: str = "cuda"):
    """
    Load the pretrained I2V LatentVisualDiffusion model.

    Follows the same approach as hum_infer/scripts/evaluation/inference.py
    """
    from utils.utils import instantiate_from_config

    # Load model config
    base_config_path = config['model']['base_config']
    base_config_path = str(Path(__file__).parent / base_config_path)
    full_config = OmegaConf.load(base_config_path)
    model_config = full_config.pop("model", OmegaConf.create())

    # Disable use_checkpoint to avoid DeepSpeed issues
    model_config['params']['unet_config']['params']['use_checkpoint'] = False

    # Disable rand_cond_frame — we always condition on frame 0
    if 'rand_cond_frame' not in model_config['params']:
        model_config['params']['rand_cond_frame'] = False

    # Instantiate model
    model = instantiate_from_config(model_config)

    # Load base checkpoint
    checkpoint_path = config['model']['checkpoint']
    checkpoint_path = str(Path(__file__).parent / checkpoint_path)
    if os.path.exists(checkpoint_path):
        print(f"Loading base checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print(">>> Base checkpoint loaded.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load distilled UNet weights
    unet_path = config['model'].get('unet_path')
    if unet_path:
        unet_path = str(Path(__file__).parent / unet_path)
        if os.path.exists(unet_path):
            print(f"Loading UNet weights from {unet_path}")
            unet_sd = torch.load(unet_path, map_location="cpu")
            model.model.diffusion_model.load_state_dict(unet_sd, strict=False)
            print(">>> UNet loaded.")

    # Load image projection model weights
    img_proj_path = config['model'].get('img_proj_path')
    if img_proj_path:
        img_proj_path = str(Path(__file__).parent / img_proj_path)
        if os.path.exists(img_proj_path):
            print(f"Loading image projection from {img_proj_path}")
            img_proj_sd = torch.load(img_proj_path, map_location="cpu")
            model.image_proj_model.load_state_dict(img_proj_sd, strict=False)
            print(">>> Image projection model loaded.")

    model = model.to(device)
    return model, model_config


def get_latent_z(model, videos):
    """Encode video frames to latent space."""
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def compute_i2v_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute I2V diffusion training loss with first-frame conditioning.

    The first frame of each video is used as the conditioning image.
    This mirrors the inference pipeline in image_guided_synthesis().
    """
    video = batch['video'].to(device)  # (B, C, T, H, W)
    captions = batch['caption']

    b, c, t, h, w = video.shape

    # === Encode full video to latent space ===
    with torch.no_grad():
        z = get_latent_z(model, video)  # (B, C_z, T, H_z, W_z)

    # === Build conditioning (same as LatentVisualDiffusion.get_batch_input) ===

    # 1. Text conditioning
    with torch.no_grad():
        cond_emb = model.get_learned_conditioning(captions)  # (B, L_text, D)

    # 2. Image conditioning — always use frame 0
    img = video[:, :, 0]  # (B, C, H, W) — first frame
    with torch.no_grad():
        img_emb = model.embedder(img)  # (B, L_img, D)
        img_emb = model.image_proj_model(img_emb)  # (B, L_proj, D)

    # 3. Cross-attention conditioning: concat text + image embeddings
    cond = {}
    cond["c_crossattn"] = [torch.cat([cond_emb, img_emb], dim=1)]

    # 4. Hybrid conditioning: concat first-frame latent across all timesteps
    if model.model.conditioning_key == 'hybrid':
        img_cat_cond = z[:, :, :1, :, :]  # First frame latent
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]

    # === Diffusion loss ===
    # Sample timesteps
    timesteps = torch.randint(0, model.num_timesteps, (b,), device=device).long()

    # Add noise to latent video
    noise = torch.randn_like(z)
    noisy_z = model.q_sample(x_start=z, t=timesteps, noise=noise)

    # Get model prediction (apply_model handles the conditioning dict)
    model_output = model.apply_model(noisy_z, timesteps, cond)

    # Target depends on parameterization
    if model.parameterization == "eps":
        target = noise
    elif model.parameterization == "x0":
        target = z
    elif model.parameterization == "v":
        target = model.get_v(z, noise, timesteps)
    else:
        raise NotImplementedError(f"Parameterization '{model.parameterization}' not supported")

    loss = F.mse_loss(model_output, target, reduction='mean')

    return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    config: Dict,
    epoch: int,
    device: str = "cuda",
    global_step: int = 0,
) -> int:
    """Train for one epoch."""
    model.train()
    # Keep frozen components in eval mode
    if hasattr(model, 'first_stage_model'):
        model.first_stage_model.eval()
    if hasattr(model, 'cond_stage_model'):
        model.cond_stage_model.eval()
    if hasattr(model, 'embedder'):
        model.embedder.eval()
    if hasattr(model, 'image_proj_model'):
        model.image_proj_model.eval()

    train_config = config['training']
    accumulation_steps = train_config['gradient_accumulation_steps']
    max_grad_norm = train_config['max_grad_norm']
    use_fp16 = train_config['fp16']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    total_loss = 0.0
    step_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        with autocast(enabled=use_fp16):
            loss = compute_i2v_loss(model, batch, device)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        step_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Wandb logging
            avg_step_loss = step_loss / accumulation_steps
            lr = scheduler.get_last_lr()[0]
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'train/loss': avg_step_loss,
                    'train/lr': lr,
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                }, step=global_step)
            step_loss = 0.0

            # Save checkpoint periodically
            if global_step > 0 and global_step % train_config['save_steps'] == 0:
                save_path = os.path.join(
                    config['model']['output_dir'],
                    f'checkpoint-{global_step}'
                )
                os.makedirs(save_path, exist_ok=True)
                save_lora_weights(model, save_path)
                print(f"\nSaved checkpoint at step {global_step}")

        total_loss += loss.item() * accumulation_steps

        # Console logging
        if (step + 1) % train_config['logging_steps'] == 0:
            avg_loss = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}',
                'step': global_step,
            })

    return global_step


def create_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler."""
    train_config = config['training']
    warmup_steps = train_config['warmup_steps']

    if train_config['lr_scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=train_config['learning_rate'] * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_steps // 3,
            gamma=0.5,
        )

    return scheduler


def main(args):
    """Main training function."""
    use_full_dataset = getattr(args, 'full_dataset', False)
    clip_stride = getattr(args, 'clip_stride', 4)
    task_ids = getattr(args, 'task_ids', None)
    if task_ids is not None:
        task_ids = [int(x.strip()) for x in task_ids.split(',')]
    print("=" * 60)
    print("I2V LoRA Fine-tuning on LIBERO Dataset")
    print("=" * 60)

    # Load config
    config = load_config(args.config)

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    if use_full_dataset:
        output_dir = os.path.join(os.path.dirname(config['model']['output_dir']), 'checkpoints_full_data')
        print(">>> FULL DATASET MODE: using all clips from every episode")
        print(f">>> Weights will be saved to: {output_dir}")
    else:
        output_dir = config['model']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.get('logging', {}).get('project_name', 'i2v-lora-libero'),
            name=config.get('logging', {}).get('run_name', None),
            config=config,
        )
        print("Wandb initialized")
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Install with: pip install wandb")

    # Load I2V model
    print("\nLoading I2V model...")
    model, model_config = load_i2v_model(config, device)

    # Prepare model for LoRA (freeze everything except UNet)
    print("\nPreparing model for LoRA training...")
    model = prepare_i2v_model_for_lora(
        model,
        freeze_vae=config['freeze']['vae'],
        freeze_text_encoder=config['freeze']['text_encoder'],
        freeze_image_embedder=config['freeze']['image_embedder'],
        freeze_image_proj=config['freeze']['image_proj'],
    )

    # Apply LoRA to UNet
    if config['lora']['enabled']:
        print("\nApplying LoRA to UNet...")
        lora_config = get_lora_config(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            lora_dropout=config['lora']['lora_dropout'],
            target_modules=config['lora']['target_modules'],
            bias=config['lora']['bias'],
        )

        if hasattr(model, 'model'):
            model.model = apply_lora_to_model(model.model, lora_config)
        else:
            model = apply_lora_to_model(model, lora_config)

    # Print trainable parameters
    param_info = get_trainable_parameters(model)
    print(f"\nTrainable parameters: {param_info['trainable']:,} / {param_info['total']:,}")
    print(f"Percentage trainable: {param_info['percentage']:.2f}%")
    print(f"Trainable size: {param_info['trainable_mb']:.2f} MB")

    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset_config = config['dataset']
    data_dir = os.path.expanduser(
        os.path.join(dataset_config['cache_dir'], dataset_config['data_path'])
    )
    meta_dir = os.path.expanduser(
        os.path.join(dataset_config['cache_dir'], dataset_config['meta_path'])
    )

    if task_ids:
        print(f"Filtering to task IDs: {task_ids}")

    if use_full_dataset:
        print(f"Using ALL CLIPS dataset (clip_stride={clip_stride})")
        dataloader = create_dataloader_all_clips(
            data_dir=data_dir,
            meta_dir=meta_dir,
            batch_size=config['training']['batch_size'],
            video_length=dataset_config['video_length'],
            frame_stride=dataset_config['frame_stride'],
            clip_stride=clip_stride,
            resolution=tuple(dataset_config['resolution']),
            num_workers=4,
            shuffle=True,
            overlap=True,
            task_ids=task_ids,
        )
        print(f"Dataset size: {len(dataloader.dataset)} clips (all clips from all episodes)")
    else:
        dataloader = create_dataloader(
            data_dir=data_dir,
            meta_dir=meta_dir,
            batch_size=config['training']['batch_size'],
            video_length=dataset_config['video_length'],
            frame_stride=dataset_config['frame_stride'],
            resolution=tuple(dataset_config['resolution']),
            num_workers=4,
            shuffle=True,
            task_ids=task_ids,
        )
        print(f"Dataset size: {len(dataloader.dataset)} episodes (1 clip per episode)")
    print(f"Batches per epoch: {len(dataloader)}")

    # Calculate total steps
    train_config = config['training']
    if train_config['max_steps'] > 0:
        total_steps = train_config['max_steps']
    else:
        steps_per_epoch = len(dataloader) // train_config['gradient_accumulation_steps']
        total_steps = max(steps_per_epoch * train_config['max_epochs'], 1)
    print(f"Total training steps: {total_steps}")

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config['learning_rate'],
        betas=(train_config['adam_beta1'], train_config['adam_beta2']),
        eps=train_config['adam_epsilon'],
        weight_decay=train_config['weight_decay'],
    )

    # Create scheduler and scaler
    scheduler = create_scheduler(optimizer, config, total_steps)
    scaler = GradScaler(enabled=train_config['fp16'])

    # Training loop
    print("\nStarting training...")
    print(f"  Conditioning: first frame ALWAYS (no random placement)")
    print(f"  Parameterization: {model.parameterization}")
    print(f"  Resolution: {dataset_config['resolution']}")
    print(f"  Video length: {dataset_config['video_length']} frames")
    print()

    global_step = 0

    for epoch in range(train_config['max_epochs']):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{train_config['max_epochs']}")
        print(f"{'='*40}")

        global_step = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch + 1,
            device=device,
            global_step=global_step,
        )

        # Save epoch checkpoint
        save_path = os.path.join(output_dir, f'epoch-{epoch + 1}')
        os.makedirs(save_path, exist_ok=True)
        save_lora_weights(model, save_path)

        if train_config['max_steps'] > 0 and global_step >= train_config['max_steps']:
            print(f"Reached max steps ({train_config['max_steps']}), stopping.")
            break

    # Save final model
    final_path = os.path.join(output_dir, 'final')
    os.makedirs(final_path, exist_ok=True)
    save_lora_weights(model, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Finish wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train I2V with LoRA on LIBERO")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--full_dataset",
        action="store_true",
        help="Use all clips from every episode instead of 1 random clip per episode",
    )
    parser.add_argument(
        "--clip_stride",
        type=int,
        default=4,
        help="Stride between clip start positions (lower = more overlap, more clips). Default: 4",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to filter by, e.g. '10,11'",
    )

    args = parser.parse_args()
    main(args)
