"""
LoRA Inference Script for I2V Robot Future Frame Prediction

Loads the base I2V model + LoRA weights, takes a single robot observation
image + task description, and generates future video frames.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hum_infer"))

from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config
from lora_utils import load_lora_weights


def load_i2v_model(config_path, ckpt_path, unet_path=None, img_proj_path=None, device="cuda"):
    """Load I2V model following inference.py pattern."""
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False

    model = instantiate_from_config(model_config)

    # Load base checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    print(">>> Base checkpoint loaded.")

    # Load UNet
    if unet_path and os.path.exists(unet_path):
        unet_sd = torch.load(unet_path, map_location="cpu")
        model.model.diffusion_model.load_state_dict(unet_sd, strict=False)
        print(">>> UNet loaded.")

    # Load image projection
    if img_proj_path and os.path.exists(img_proj_path):
        img_proj_sd = torch.load(img_proj_path, map_location="cpu")
        model.image_proj_model.load_state_dict(img_proj_sd, strict=False)
        print(">>> Image projection loaded.")

    model = model.to(device)
    model.eval()
    return model


def get_latent_z(model, videos):
    """Encode video frames to latent space."""
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def generate_future_frames(
    model, image, prompt, 
    height=256, width=256, video_length=16,
    ddim_steps=16, ddim_eta=1.0, 
    unconditional_guidance_scale=7.5,
    fs=10, device="cuda",
    reneg_path=None,
):
    """
    Generate future frames given a conditioning image and text prompt.
    
    Args:
        model: I2V model (with or without LoRA)
        image: PIL Image or tensor (C, H, W)
        prompt: Text description of the task
        height, width: Output resolution
        video_length: Number of frames to generate
        ddim_steps: Number of denoising steps
        ddim_eta: DDIM eta parameter
        unconditional_guidance_scale: CFG scale
        fs: Frame stride / fps condition
        device: CUDA device
        reneg_path: Path to negative conditioning checkpoint
    
    Returns:
        video tensor (T, C, H, W) in [0, 1]
    """
    ddim_sampler = DDIMSampler(model)

    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Process input image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if isinstance(image, Image.Image):
        image = transform(image)
    
    # Create video tensor with image repeated (B, C, T, H, W)
    image = image.unsqueeze(0).to(device)  # (1, C, H, W)
    videos = image.unsqueeze(2).repeat(1, 1, video_length, 1, 1)  # (1, C, T, H, W)

    # Noise shape
    h, w = height // 8, width // 8
    noise_shape = [1, 4, video_length, h, w]

    # Frame stride
    fs_tensor = torch.tensor([fs], dtype=torch.long, device=device)

    # === Build conditioning ===
    img = videos[:, :, 0]  # First frame (1, C, H, W)
    img_emb = model.embedder(img)
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning([prompt])
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos)
        img_cat_cond = z[:, :, :1, :, :]
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]

    # === Build unconditional conditioning ===
    if unconditional_guidance_scale != 1.0:
        if reneg_path and os.path.exists(reneg_path):
            uc_emb = torch.load(reneg_path).to(device)
        else:
            uc_emb = model.get_learned_conditioning([""])
        
        uc_img_emb = model.embedder(torch.zeros_like(img))
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    # === Sample ===
    with torch.no_grad(), torch.amp.autocast('cuda'):
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=1,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            fs=fs_tensor,
            timestep_spacing='uniform_trailing',
            guidance_rescale=0.7,
        )

    # Decode
    batch_images = model.decode_first_stage(samples)

    # Convert to [0, 1] range
    video_out = batch_images[0]  # (C, T, H, W)
    video_out = torch.clamp((video_out + 1.0) / 2.0, 0, 1)
    video_out = video_out.permute(1, 0, 2, 3)  # (T, C, H, W)

    return video_out


def main():
    parser = argparse.ArgumentParser(description="I2V LoRA Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Task description")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--config", type=str, 
                        default="../configs/inference_i2v_512_v2.0_distil.yaml")
    parser.add_argument("--ckpt", type=str, 
                        default="../hum_infer/checkpoints/stage_1.ckpt")
    parser.add_argument("--unet_path", type=str, 
                        default="../hum_infer/checkpoints/unet.pt")
    parser.add_argument("--img_proj_path", type=str, 
                        default="../hum_infer/checkpoints/img_proj.pt")
    parser.add_argument("--reneg_path", type=str,
                        default="../hum_infer/checkpoints/reneg_checkpoint.bin")
    parser.add_argument("--output", type=str, default="output_prediction.mp4")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    config_path = str(script_dir / args.config)
    ckpt_path = str(script_dir / args.ckpt)
    unet_path = str(script_dir / args.unet_path)
    img_proj_path = str(script_dir / args.img_proj_path)
    reneg_path = str(script_dir / args.reneg_path) if args.reneg_path else None

    # Load model
    print("Loading I2V model...")
    model = load_i2v_model(config_path, ckpt_path, unet_path, img_proj_path, device)

    # Load LoRA weights if provided
    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}...")
        if hasattr(model, 'model'):
            model.model = load_lora_weights(model.model, args.lora_path)
        else:
            model = load_lora_weights(model, args.lora_path)

    # Generate
    print(f"Generating future frames for: '{args.prompt}'")
    video = generate_future_frames(
        model=model,
        image=args.image,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        video_length=args.video_length,
        ddim_steps=args.ddim_steps,
        unconditional_guidance_scale=args.guidance_scale,
        device=device,
        reneg_path=reneg_path,
    )

    # Save video
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    video_uint8 = (video * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()  # (T, H, W, C)
    torchvision.io.write_video(args.output, video_uint8, fps=args.fps, video_codec='h264')
    print(f"Saved prediction video to {args.output}")


if __name__ == "__main__":
    main()
