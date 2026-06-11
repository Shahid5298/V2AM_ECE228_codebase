"""Finetune the flow matching action head for task 10 using I2V LoRA-generated
video frames as the visual conditioning stream.

SYNCHRONIZATION LOGIC
─────────────────────
  I2V LoRA  : generates 8 frames at frame_stride=2
               → covers raw timesteps [t, t+2, t+4, t+6, t+8, t+10, t+12, t+14]
  Flow policy: predicts chunk_size=16 actions at action_stride=1
               → covers raw timesteps [t, t+1, …, t+15]

  Because the I2V model is conditioned on frame t and produces frames for
  exactly the same temporal span as the action chunk, no resampling is needed.
  We just pass the 8 I2V frames directly to VideoMAE (which already supports
  num_frames=8 via the stored positional-embedding interpolation).

WORKFLOW
────────
  1.  Load the task-10 MimicGen parquet files and build a window dataset that
      yields (first_frame_PIL, wrist_frames_np, proprio_np, actions_np).
  2.  For every unique episode×window, run I2V LoRA inference on the first
      frame to obtain 8 predicted future frames and cache them on disk.
  3.  Load `best.pt` (flow matching head + action stats).
  4.  Fine-tune: replace the real agentview stream with the cached I2V frames;
      keep the real wrist stream unchanged.

Usage
─────
  cd <repo-root>
  python finetune_on_i2v.py \\
      --flow-ckpt   outputs/flow_matching_task_10/best.pt \\
      --lora-path   $HUMMINGBIRD_I2V/lora/checkpoints_mimicgen_t10/epoch-4 \\
      --output-dir  outputs/flow_matching_task_10_i2v \\
      --cache-dir   /tmp/i2v_frames_task10 \\
      --task-id     10 \\
      --epochs      20 \\
      --lr          5e-5 \\
      --batch-size  8
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor
from tqdm import tqdm

# ── locate the repo root so that `src` resolves ──────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── stage-1 i2v LoRA scripts (in 1_video_finetuning/). Override with the env
#    var HUMMINGBIRD_I2V if you keep them elsewhere. ───────────────────────────
LORA_DIR    = Path(os.environ.get("HUMMINGBIRD_I2V", str(ROOT / "1_video_finetuning")))
HUMMINGBIRD = LORA_DIR
sys.path.insert(0, str(LORA_DIR))

from src.flow_matching.config  import FlowMatchingConfig
from src.flow_matching.model   import FlowMatchingActionHead
from src.videomae_encoder      import VideoMAEFeatureExtractor
from src.utils                 import (
    set_seed, save_checkpoint, load_checkpoint,
    get_cosine_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  I2V inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_i2v_model(lora_path: str, device: str, hummingbird_root: Path):
    """Load Hummingbird I2V base model and inject LoRA weights."""
    from inference_lora import load_i2v_model  # noqa: F401
    from lora_utils    import load_lora_weights  # noqa: F401

    cfg_path      = str(hummingbird_root / "i2v/configs/inference_i2v_512_v2.0_distil.yaml")
    ckpt_path     = str(hummingbird_root / "i2v/hum_infer/checkpoints/stage_1.ckpt")
    unet_path     = str(hummingbird_root / "i2v/hum_infer/checkpoints/unet.pt")
    img_proj_path = str(hummingbird_root / "i2v/hum_infer/checkpoints/img_proj.pt")
    reneg_path_s  = str(hummingbird_root / "i2v/hum_infer/checkpoints/reneg_checkpoint.bin")

    model = load_i2v_model(cfg_path, ckpt_path, unet_path, img_proj_path, device)
    if hasattr(model, "model"):
        model.model = load_lora_weights(model.model, lora_path)
    else:
        model = load_lora_weights(model, lora_path)

    reneg_path = reneg_path_s if Path(reneg_path_s).exists() else None
    return model, reneg_path


@torch.no_grad()
def _run_i2v(i2v_model, reneg_path: str | None, pil_image: Image.Image,
             prompt: str, device: str,
             num_frames: int = 8, resolution: tuple[int, int] = (256, 256)
             ) -> np.ndarray:
    """Generate `num_frames` future frames with I2V LoRA.

    Returns:
        frames_np: (num_frames, H, W, 3) uint8 numpy array
    """
    from inference_lora import generate_future_frames  # noqa: F401

    video = generate_future_frames(
        model=i2v_model,
        image=pil_image,
        prompt=prompt,
        height=resolution[0],
        width=resolution[1],
        video_length=num_frames,
        ddim_steps=16,
        unconditional_guidance_scale=7.5,
        device=device,
        reneg_path=reneg_path,
    )
    # video: (T, C, H, W) float32 in [0, 1]
    frames_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return frames_np   # (8, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pre-generate (or load cached) I2V frames for every window
# ─────────────────────────────────────────────────────────────────────────────

def _window_cache_path(cache_dir: Path, pq_path: str, start: int) -> Path:
    key = hashlib.md5(f"{pq_path}:{start}".encode()).hexdigest()
    return cache_dir / f"{key}.npy"


def pregenerate_i2v_frames(
    parquet_files: list[str],
    task_prompts: dict[int, str],
    cache_dir: Path,
    lora_path: str,
    hummingbird_root: Path,
    device: str,
    config: FlowMatchingConfig,
    force_regen: bool = False,
) -> None:
    """Run I2V inference for every window start-frame and cache the results."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame_stride    = config.frame_stride   # 2
    num_frames      = config.num_frames     # 8
    future_offset   = config.future_frame_offset  # 1
    chunk_size      = config.chunk_size     # 16
    action_stride   = config.action_stride  # 1
    window_stride   = config.window_stride  # 8
    raw_window_len  = future_offset + frame_stride * (num_frames - 1) + 1

    # Load I2V model once
    print("[I2V] Loading I2V LoRA model …")
    i2v_model, reneg_path = _load_i2v_model(lora_path, device, hummingbird_root)

    total_windows = 0
    already_cached = 0

    for pq_path in tqdm(parquet_files, desc="Episodes"):
        table = pq.read_table(pq_path)
        T = len(table)
        task_id = table["task_index"][0].as_py()
        prompt  = task_prompts.get(task_id, "robot arm manipulating objects")

        # All image bytes for the episode (agentview only)
        main_dicts = table["observation.image"].to_pylist()

        for start in range(0, T - raw_window_len + 1, window_stride):
            total_windows += 1
            cache_path = _window_cache_path(cache_dir, pq_path, start)

            if cache_path.exists() and not force_regen:
                already_cached += 1
                continue

            # Conditioning image = the window's first frame
            first_frame_bytes = main_dicts[start]["bytes"]
            pil_img = Image.open(io.BytesIO(first_frame_bytes)).convert("RGB")

            frames_np = _run_i2v(i2v_model, reneg_path, pil_img, prompt, device,
                                  num_frames=num_frames)
            np.save(str(cache_path), frames_np)

    print(f"[I2V] Done. {total_windows} windows total, "
          f"{already_cached} were already cached, "
          f"{total_windows - already_cached} newly generated.")
    del i2v_model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset that replaces the real agentview stream with I2V frames
# ─────────────────────────────────────────────────────────────────────────────

class I2VConditionedDataset(Dataset):
    """Sliding-window dataset for task-10 that uses cached I2V frames as main
    visual conditioning while keeping real wrist frames and ground-truth actions.
    """

    def __init__(self, config: FlowMatchingConfig, parquet_files: list[str],
                 cache_dir: Path):
        self.config           = config
        self.cache_dir        = cache_dir
        self.num_frames       = config.num_frames       # 8
        self.frame_stride     = config.frame_stride     # 2
        self.chunk_size       = config.chunk_size       # 16
        self.action_stride    = config.action_stride    # 1
        self.future_offset    = config.future_frame_offset  # 1
        self.proprio_hist     = config.proprio_history_size  # 4
        self.window_stride    = config.window_stride    # 8

        self.raw_window_len   = (
            self.future_offset + self.frame_stride * (self.num_frames - 1) + 1
        )

        # Flat index: (pq_path, start, task_id)  — only include windows with cache
        self.index: list[tuple[str, int, int]] = []
        skipped = 0

        for pq_path in tqdm(parquet_files, desc="Indexing dataset", unit="ep"):
            md       = pq.read_metadata(pq_path)
            T        = md.num_rows
            table_ti = pq.read_table(pq_path, columns=["task_index"])
            task_id  = table_ti["task_index"][0].as_py()

            for start in range(0, T - self.raw_window_len + 1, self.window_stride):
                cp = _window_cache_path(cache_dir, pq_path, start)
                if not cp.exists():
                    skipped += 1
                    continue
                self.index.append((pq_path, start, task_id))

        if skipped:
            print(f"[Dataset] WARNING: {skipped} windows missing from I2V cache "
                  "— run pregenerate_i2v_frames first.")
        print(f"[Dataset] {len(self.index)} windows indexed.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        pq_path, start, task_id = self.index[idx]

        # ── Load I2V-generated main frames from cache ─────────────────────
        cache_path = _window_cache_path(self.cache_dir, pq_path, start)
        i2v_frames_np = np.load(str(cache_path))  # (8, H, W, 3) uint8

        # ── Load real data from parquet ────────────────────────────────────
        table = pq.read_table(pq_path)
        T     = len(table)

        # Wrist frames (real) at the same temporal indices
        wrist_dicts = table["observation.image_wrist"].to_pylist()
        frame_indices = [
            start + self.future_offset + j * self.frame_stride
            for j in range(self.num_frames)
        ]
        wrist_frames = []
        for fi in frame_indices:
            fi = min(fi, T - 1)
            wrist_d = wrist_dicts[fi]
            img_w = Image.open(io.BytesIO(wrist_d["bytes"])).convert("RGB")
            wrist_frames.append(np.array(img_w))
        frames_wrist_np = np.stack(wrist_frames)   # (8, H, W, 3)

        # Actions for the chunk starting at `start`
        raw_actions = np.stack(
            table["action"].to_numpy(zero_copy_only=False)
        ).astype(np.float32)
        action_indices = [
            min(start + j * self.action_stride, T - 1)
            for j in range(self.chunk_size)
        ]
        actions = raw_actions[action_indices]       # (16, 7)

        # Proprio history: `proprio_hist` steps ending at `start`
        pos   = np.stack(table["observation.robot0_eef_pos"].to_numpy(zero_copy_only=False))
        quat  = np.stack(table["observation.robot0_eef_quat"].to_numpy(zero_copy_only=False))
        grip  = np.stack(table["observation.robot0_gripper_qpos"].to_numpy(zero_copy_only=False))
        euler = R.from_quat(quat).as_euler("xyz", degrees=False)
        proprio_full = np.concatenate([pos, euler, grip], axis=-1).astype(np.float32)

        p_start = max(0, start - self.proprio_hist + 1)
        proprio  = proprio_full[p_start : start + 1]
        if len(proprio) < self.proprio_hist:
            pad = np.repeat(proprio[:1], self.proprio_hist - len(proprio), axis=0)
            proprio = np.concatenate([pad, proprio], axis=0)

        return {
            "frames":        i2v_frames_np,    # (8, H, W, 3)  — I2V generated
            "frames_wrist":  frames_wrist_np,  # (8, H, W, 3)  — real
            "proprio":       proprio,           # (4, 8)
            "actions":       actions,           # (16, 7)
            "task_id":       task_id,
        }


def collate_fn_i2v(batch: list[dict], processor: VideoMAEImageProcessor) -> dict:
    """Collate batch: run VideoMAE pre-processing on I2V frames + real wrist."""
    videos_main  = [list(s["frames"])       for s in batch]
    videos_wrist = [list(s["frames_wrist"]) for s in batch]

    pixel_values       = processor(videos_main,  return_tensors="pt")["pixel_values"]
    pixel_values_wrist = processor(videos_wrist, return_tensors="pt")["pixel_values"]

    proprio  = torch.stack([torch.from_numpy(s["proprio"])  for s in batch])
    actions  = torch.stack([torch.from_numpy(s["actions"])  for s in batch])
    task_ids = torch.tensor([s["task_id"] for s in batch], dtype=torch.long)

    return {
        "pixel_values":       pixel_values,        # (B, 16, 3, 224, 224)
        "pixel_values_wrist": pixel_values_wrist,  # (B, 16, 3, 224, 224)
        "proprio":            proprio,             # (B, 4, 8)
        "actions":            actions,             # (B, 16, 7)
        "task_ids":           task_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune_one_epoch(
    model: FlowMatchingActionHead,
    videomae: VideoMAEFeatureExtractor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: FlowMatchingConfig,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, float]:
    model.train()

    total_loss         = 0.0
    total_flow_loss    = 0.0
    total_gripper_loss = 0.0
    n_batches          = 0

    a_mean = action_mean.to(config.device)
    a_std  = action_std.to(config.device)

    pbar = tqdm(dataloader, desc="Finetune", leave=False)
    for batch in pbar:
        proprio  = batch["proprio"].to(config.device)
        actions  = batch["actions"].to(config.device)

        continuous_actions = (actions[:, :, :config.continuous_action_dim] - a_mean) / a_std
        gripper_actions    = (actions[:, :, config.continuous_action_dim:] + 1.0) / 2.0

        # Encode I2V main frames + real wrist frames through frozen VideoMAE
        feats_main  = videomae(batch["pixel_values"].to(config.device))
        feats_wrist = videomae(batch["pixel_values_wrist"].to(config.device))
        video_features = torch.cat([feats_main, feats_wrist], dim=1)

        losses = model.compute_losses(
            video_features, proprio, continuous_actions, gripper_actions,
        )
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss         += loss.item()
        total_flow_loss    += losses["flow_loss"].item()
        total_gripper_loss += losses["gripper_loss"].item()
        n_batches          += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {
        "loss":         total_loss         / max(n_batches, 1),
        "flow_loss":    total_flow_loss    / max(n_batches, 1),
        "gripper_loss": total_gripper_loss / max(n_batches, 1),
        "lr":           optimizer.param_groups[0]["lr"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Finetune flow matching policy on I2V-generated frames (task 10)"
    )
    parser.add_argument("--flow-ckpt",   default="outputs/flow_matching_task_10/best.pt",
                        help="Path to the pre-trained flow matching checkpoint (best.pt)")
    parser.add_argument("--lora-path",   required=True,
                        help="Path to LoRA checkpoint dir (e.g. .../checkpoints_mimicgen_t10/epoch-4)")
    parser.add_argument("--output-dir",  default="outputs/flow_matching_task_10_i2v",
                        help="Where to save finetuned checkpoints")
    parser.add_argument("--cache-dir",   default="/tmp/i2v_frames_task10",
                        help="Directory for cached I2V frame numpy files")
    parser.add_argument("--task-id",     type=int, default=10, help="MimicGen task index")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--force-regen", action="store_true",
                        help="Force re-generation of cached I2V frames")
    parser.add_argument("--skip-pregen", action="store_true",
                        help="Skip pre-generation (assume cache already populated)")
    parser.add_argument("--val-split",   type=float, default=0.2,
                        help="Fraction of demos held out for validation")
    args = parser.parse_args()

    out_dir   = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build config (override relevant fields) ───────────────────────────
    config = FlowMatchingConfig()
    config.task_filter_indices = (args.task_id,)
    config.batch_size   = args.batch_size
    config.lr           = args.lr
    config.epochs       = args.epochs
    config.num_workers  = args.num_workers
    config.device       = args.device
    config.output_dir   = out_dir
    # Keep condition_on_future_video=True; no history stream needed here
    config.include_history_frames   = False
    config.condition_on_future_video = True

    # visual_feature_dim must match what the restored model expects
    # (main + wrist → 2 × videomae_seq_len tokens, same as original training)

    set_seed(config.seed)

    # ── Discover task-10 parquet files ────────────────────────────────────
    mimicgen_data = Path(config.mimicgen_data_root)
    all_parquets  = sorted(mimicgen_data.glob("data/chunk-*/*.parquet"))
    task_files   = []
    for pq_path in tqdm(all_parquets, desc="Filtering task files"):
        table_ti = pq.read_table(str(pq_path), columns=["task_index"])
        if table_ti["task_index"][0].as_py() == args.task_id:
            task_files.append(str(pq_path))

    if not task_files:
        raise RuntimeError(
            f"No parquet files found for task_id={args.task_id} "
            f"in {mimicgen_data}"
        )
    print(f"Found {len(task_files)} episodes for task {args.task_id}.")

    # Train / val split
    split_idx  = max(1, int(len(task_files) * (1 - args.val_split)))
    train_files = task_files[:split_idx]
    val_files   = task_files[split_idx:]
    print(f"Split: {len(train_files)} train, {len(val_files)} val episodes.")

    # ── Load task prompts ─────────────────────────────────────────────────
    tasks_path = mimicgen_data / "meta" / "tasks.jsonl"
    task_prompts: dict[int, str] = {}
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                d = json.loads(line)
                task_prompts[d["task_index"]] = d["task"]

    hummingbird_root = HUMMINGBIRD.parent  # .../Hummingbird

    # ── Step 1: Pre-generate I2V frames (with disk cache) ─────────────────
    if not args.skip_pregen:
        print("\n" + "=" * 60)
        print("STEP 1 — Pre-generating I2V frames (agentview stream)")
        print("=" * 60)
        pregenerate_i2v_frames(
            parquet_files   = train_files + val_files,
            task_prompts    = task_prompts,
            cache_dir       = cache_dir,
            lora_path       = args.lora_path,
            hummingbird_root = hummingbird_root,
            device          = args.device,
            config          = config,
            force_regen     = args.force_regen,
        )

    # ── Step 2: Build datasets ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Building I2V-conditioned datasets")
    print("=" * 60)

    processor  = VideoMAEImageProcessor.from_pretrained(str(config.model_dir))
    collate    = functools.partial(collate_fn_i2v, processor=processor)

    train_ds   = I2VConditionedDataset(config, train_files, cache_dir)
    val_ds     = I2VConditionedDataset(config, val_files,   cache_dir)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=collate, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate, pin_memory=True,
        drop_last=False,
    )

    print(f"  Train: {len(train_ds)} windows | Val: {len(val_ds)} windows")

    # ── Step 3: Load frozen VideoMAE ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Loading frozen VideoMAE encoder")
    print("=" * 60)
    videomae = VideoMAEFeatureExtractor(
        config.model_dir, layer_idx=config.videomae_layer,
        num_frames_expected=config.num_frames, device=config.device,
    ).to(config.device)

    # ── Step 4: Load pre-trained flow matching head ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Loading pre-trained flow matching policy")
    print("=" * 60)
    ckpt_path = Path(args.flow_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flow matching checkpoint not found: {ckpt_path}")

    model = FlowMatchingActionHead(config).to(config.device)
    ckpt  = torch.load(str(ckpt_path), weights_only=False, map_location=config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    action_mean = ckpt["action_mean"].to(config.device)
    action_std  = ckpt["action_std"].to(config.device)
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"  Action mean: {action_mean.cpu().numpy()}")
    print(f"  Action std:  {action_std.cpu().numpy()}")

    # ── Step 5: Optimizer + scheduler ────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                     weight_decay=config.weight_decay)
    total_steps  = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * min(2, config.warmup_epochs)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"\n  Optimizer: AdamW  lr={config.lr}  wd={config.weight_decay}")
    print(f"  Total steps: {total_steps}  warmup: {warmup_steps}")

    # ── Step 6: Finetune loop ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 5 — Finetuning for {config.epochs} epochs")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        t0 = time.time()

        train_metrics = finetune_one_epoch(
            model, videomae, train_loader, optimizer, scheduler,
            config, action_mean, action_std,
        )

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        val_n     = 0
        with torch.no_grad():
            for batch in val_loader:
                proprio  = batch["proprio"].to(config.device)
                actions  = batch["actions"].to(config.device)
                cont_act = (actions[:, :, :config.continuous_action_dim] - action_mean) / action_std
                grip_act = (actions[:, :, config.continuous_action_dim:] + 1.0) / 2.0

                feats_main  = videomae(batch["pixel_values"].to(config.device))
                feats_wrist = videomae(batch["pixel_values_wrist"].to(config.device))
                video_feats = torch.cat([feats_main, feats_wrist], dim=1)

                losses  = model.compute_losses(video_feats, proprio, cont_act, grip_act)
                val_total += losses["loss"].item()
                val_n     += 1

        val_loss = val_total / max(val_n, 1)
        elapsed  = time.time() - t0

        print(
            f"  Epoch {epoch+1:3d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(flow={train_metrics['flow_loss']:.4f} "
            f"grip={train_metrics['gripper_loss']:.4f}) | "
            f"val_loss={val_loss:.6f} | "
            f"lr={train_metrics['lr']:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"loss": val_loss},
                action_mean.cpu(), action_std.cpu(),
                out_dir / "best.pt",
            )
            print(f"    --> New best val loss: {best_val_loss:.6f}  [saved]")

    save_checkpoint(
        model, optimizer, scheduler, config.epochs - 1,
        {"loss": val_loss},
        action_mean.cpu(), action_std.cpu(),
        out_dir / "final.pt",
    )

    print("\n" + "=" * 60)
    print(f"Done.  Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints → {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
