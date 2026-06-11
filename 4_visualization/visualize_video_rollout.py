"""
Hummingbird autoregressive rollout visualization.

Starts from a real training frame and autoregressively generates future frames:
  - Generate 16 frames from current frame
  - Use last generated frame as new input
  - Repeat for N windows → long continuous video

This shows whether Hummingbird can "imagine" the task completing over time.

One full video per (task, demo):
  - 10 windows × 16 frames = 160 generated frames
  - At 10fps → 16-second video per task

Outputs: ablation_results/hb_videos/

Usage:
    cd <repo-root>
    CUDA_VISIBLE_DEVICES=0 python visualize_video_rollout.py
"""

from __future__ import annotations
import io, os, sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
# External dependencies (set via env vars):
#   HUMMINGBIRD_I2V : the Hummingbird i2v tree (with its LoRA checkpoints)
#   MIMICGEN_DATA   : directory of the MimicGen demonstration parquet files
REPO         = Path(__file__).resolve().parents[1]
HB_ROOT      = Path(os.environ.get("HUMMINGBIRD_I2V", str(Path.home() / "Hummingbird" / "i2v")))
LORA_CKPT    = str(HB_ROOT / "lora" / "checkpoints_mimicgen" / "checkpoint-16000")
MIMICGEN_250 = Path(os.environ.get("MIMICGEN_DATA", str(Path.home() / "mimicgen_training_data_250")))

for p in [str(HB_ROOT), str(HB_ROOT / "lora")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pyarrow.parquet as pq


# ── tasks to visualize ────────────────────────────────────────────────────────
TASKS = [
    (0,  "coffee_pod",    "robot arm inserting a coffee pod"),
    (10, "cube_stacking", "robot arm stacking a cube on top of another cube"),
    (12, "stack_three",   "robot arm stacking three blocks"),
]

# 3 demos per task, each starting from step 0
DEMOS_PER_TASK = [0, 1, 2]

# autoregressive windows — more windows = longer video
# 15 windows × 16 frames @ 10fps = 24 seconds per video
NUM_WINDOWS  = 15
FRAMES_PER_WINDOW = 16
FPS = 10


def load_frame(task_id: int, demo_idx: int, step: int) -> np.ndarray:
    ep_idx  = task_id * 250 + demo_idx
    pq_path = MIMICGEN_250 / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    table   = pq.read_table(str(pq_path))
    n       = len(table)
    step    = min(step, n - 1)
    raw     = table["observation.image"][step].as_py()
    img     = Image.open(io.BytesIO(raw["bytes"])).convert("RGB").resize((256, 256))
    return np.array(img)


def load_gt_frames(task_id: int, demo_idx: int) -> list[np.ndarray]:
    """Load all GT frames for a demo (for the GT reference video)."""
    ep_idx  = task_id * 250 + demo_idx
    pq_path = MIMICGEN_250 / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    table   = pq.read_table(str(pq_path))
    frames  = []
    for raw in table["observation.image"].to_pylist():
        img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB").resize((256, 256))
        frames.append(np.array(img))
    return frames


def load_hummingbird(device: str):
    from inference_lora import load_i2v_model
    from lora_utils     import load_lora_weights

    cfg    = str(HB_ROOT / "configs/inference_i2v_512_v2.0_distil.yaml")
    ckpt   = str(HB_ROOT / "hum_infer/checkpoints/stage_1.ckpt")
    unet   = str(HB_ROOT / "hum_infer/checkpoints/unet.pt")
    iprj   = str(HB_ROOT / "hum_infer/checkpoints/img_proj.pt")
    reneg_s = str(HB_ROOT / "hum_infer/checkpoints/reneg_checkpoint.bin")

    model = load_i2v_model(cfg, ckpt, unet, iprj, device)
    if hasattr(model, "model"):
        model.model = load_lora_weights(model.model, LORA_CKPT)
    else:
        model = load_lora_weights(model, LORA_CKPT)

    reneg = reneg_s if Path(reneg_s).exists() else None
    return model, reneg


@torch.no_grad()
def generate_window(hb_model, reneg, frame_np: np.ndarray,
                    prompt: str, device: str) -> np.ndarray:
    """Generate FRAMES_PER_WINDOW frames from a single input frame.
    Returns (FRAMES_PER_WINDOW, H, W, 3) uint8."""
    from inference_lora import generate_future_frames

    pil   = Image.fromarray(frame_np).resize((256, 256))
    video = generate_future_frames(
        model=hb_model,
        image=pil,
        prompt=prompt,
        height=256, width=256,
        video_length=FRAMES_PER_WINDOW,
        ddim_steps=16,
        unconditional_guidance_scale=7.5,
        device=device,
        reneg_path=reneg,
    )
    return (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def make_sidebyside_row(gt_frames: list[np.ndarray],
                        gen_frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Returns frames: LEFT=GT (downsampled to match length), RIGHT=HB generated.
    Both sequences are the same length so you can compare directly.
    """
    n   = len(gen_frames)
    sep = np.ones((256, 6, 3), dtype=np.uint8) * 60

    # downsample GT to n frames
    indices = np.linspace(0, len(gt_frames) - 1, n).round().astype(int)
    gt_ds   = [gt_frames[i] for i in indices]

    combined = []
    for gt_f, gen_f in zip(gt_ds, gen_frames):
        row = np.concatenate([gt_f, sep, gen_f], axis=1)  # (256, 518, 3)
        combined.append(row)
    return combined


def main():
    device  = "cuda:0"
    out_dir = REPO / "ablation_results" / "hb_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Hummingbird LoRA model (~30s)...")
    hb_model, hb_reneg = load_hummingbird(device)
    print("Ready.\n")

    total_vids = 0

    for task_id, task_label, prompt in TASKS:
        print(f"\n{'='*65}")
        print(f"Task {task_id}: {task_label}")
        print(f"  prompt: \"{prompt}\"")
        print(f"  {NUM_WINDOWS} windows × {FRAMES_PER_WINDOW} frames "
              f"= {NUM_WINDOWS*FRAMES_PER_WINDOW} frames "
              f"@ {FPS}fps = {NUM_WINDOWS*FRAMES_PER_WINDOW/FPS:.0f}s per video")
        print(f"{'='*65}")

        for demo_idx in DEMOS_PER_TASK:
            print(f"\n  Demo {demo_idx}:")

            # load starting frame and GT frames for reference
            seed_frame = load_frame(task_id, demo_idx, step=0)
            gt_frames  = load_gt_frames(task_id, demo_idx)
            print(f"    GT episode length: {len(gt_frames)} steps")

            # ── save GT video for reference ───────────────────────────────
            gt_path = out_dir / f"task{task_id}_{task_label}_demo{demo_idx:02d}_GT_reference.mp4"
            if not gt_path.exists():
                w = imageio.get_writer(str(gt_path), fps=FPS)
                for f in gt_frames:
                    w.append_data(f)
                w.close()
                print(f"    saved GT reference: {gt_path.name}")

            # ── autoregressive HB rollout ─────────────────────────────────
            all_gen_frames = []
            current_frame  = seed_frame

            for win in range(NUM_WINDOWS):
                print(f"    window {win+1}/{NUM_WINDOWS} ...", end=" ", flush=True)
                gen = generate_window(hb_model, hb_reneg, current_frame, prompt, device)
                all_gen_frames.extend(gen)
                current_frame = gen[-1]  # last generated frame → next input
                print(f"done (total frames so far: {len(all_gen_frames)})")

            # ── save HB-only video ────────────────────────────────────────
            hb_path = out_dir / f"task{task_id}_{task_label}_demo{demo_idx:02d}_HB_rollout.mp4"
            w = imageio.get_writer(str(hb_path), fps=FPS)
            for f in all_gen_frames:
                w.append_data(f)
            w.close()
            print(f"    saved HB rollout:   {hb_path.name}")

            # ── save side-by-side GT vs HB ────────────────────────────────
            sbs_frames = make_sidebyside_row(gt_frames, all_gen_frames)
            sbs_path = out_dir / f"task{task_id}_{task_label}_demo{demo_idx:02d}_GT_vs_HB.mp4"
            w = imageio.get_writer(str(sbs_path), fps=FPS)
            for f in sbs_frames:
                w.append_data(f)
            w.close()
            print(f"    saved GT vs HB:     {sbs_path.name}")

            total_vids += 3  # GT ref + HB rollout + side-by-side

    print(f"\n{'='*65}")
    print(f"Done. {total_vids} videos saved to:")
    print(f"  {out_dir}")
    print(f"\nFiles:")
    for p in sorted(out_dir.iterdir()):
        mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
