"""
Hummingbird stack-specific visualization.

Uses checkpoints_mimicgen_stack/epoch-9 (stack-finetuned LoRA).
For each demo: takes the starting frame, generates 8 future frames,
saves a side-by-side video: LEFT = starting frame (held), RIGHT = generated.

Tasks: cube stacking (task 10) and stack-three (task 12).
10 demos per task = 20 videos total.

Usage:
    cd <repo-root>
    CUDA_VISIBLE_DEVICES=0 python visualize_stack_rollout.py
"""

from __future__ import annotations
import io, os, sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

# External dependencies (set via env vars):
#   HUMMINGBIRD_I2V : the Hummingbird i2v tree (with its LoRA checkpoints)
#   MIMICGEN_DATA   : directory of the MimicGen demonstration parquet files
REPO         = Path(__file__).resolve().parents[1]
HB_ROOT      = Path(os.environ.get("HUMMINGBIRD_I2V", str(Path.home() / "Hummingbird" / "i2v")))
LORA_CKPT    = str(HB_ROOT / "lora" / "checkpoints_mimicgen_stack" / "epoch-9")
MIMICGEN_250 = Path(os.environ.get("MIMICGEN_DATA", str(Path.home() / "mimicgen_training_data_250")))

for p in [str(HB_ROOT), str(HB_ROOT / "lora")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pyarrow.parquet as pq

TASKS = [
    (10, "cube_stacking", "robot arm stacking a cube on top of another cube"),
    (12, "stack_three",   "robot arm stacking three blocks"),
]
NUM_DEMOS    = 10
NUM_FRAMES   = 8   # one generation window only
FPS_SBS      = 4   # slow so you can read each generated frame
FPS_GEN      = 8   # generated-only video


def load_frame(task_id: int, demo_idx: int, step: int = 0) -> np.ndarray:
    ep_idx  = task_id * 250 + demo_idx
    pq_path = MIMICGEN_250 / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    table   = pq.read_table(str(pq_path))
    step    = min(step, len(table) - 1)
    raw     = table["observation.image"][step].as_py()
    return np.array(Image.open(io.BytesIO(raw["bytes"])).convert("RGB").resize((256, 256)))


def load_hummingbird(device: str):
    from inference_lora import load_i2v_model
    from lora_utils     import load_lora_weights

    model = load_i2v_model(
        str(HB_ROOT / "configs/inference_i2v_512_v2.0_distil.yaml"),
        str(HB_ROOT / "hum_infer/checkpoints/stage_1.ckpt"),
        str(HB_ROOT / "hum_infer/checkpoints/unet.pt"),
        str(HB_ROOT / "hum_infer/checkpoints/img_proj.pt"),
        device,
    )
    if hasattr(model, "model"):
        model.model = load_lora_weights(model.model, LORA_CKPT)
    else:
        model = load_lora_weights(model, LORA_CKPT)

    reneg = str(HB_ROOT / "hum_infer/checkpoints/reneg_checkpoint.bin")
    reneg = reneg if Path(reneg).exists() else None
    return model, reneg


@torch.no_grad()
def generate(hb_model, reneg, frame_np: np.ndarray, prompt: str, device: str) -> np.ndarray:
    """Returns (NUM_FRAMES, 256, 256, 3) uint8."""
    from inference_lora import generate_future_frames
    pil   = Image.fromarray(frame_np).resize((256, 256))
    video = generate_future_frames(
        model=hb_model, image=pil, prompt=prompt,
        height=256, width=256, video_length=NUM_FRAMES,
        ddim_steps=16, unconditional_guidance_scale=7.5,
        device=device, reneg_path=reneg,
    )
    return (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def save_sidebyside(start_frame: np.ndarray, gen: np.ndarray, path: Path):
    """Left = start frame held, right = each generated frame. 3 lead frames then 8 gen frames."""
    sep = np.ones((256, 6, 3), dtype=np.uint8) * 60
    frames = []
    # 3 lead frames so viewer sees the input clearly
    for _ in range(3):
        frames.append(np.concatenate([start_frame, sep, start_frame], axis=1))
    # 8 generated frames
    for gf in gen:
        frames.append(np.concatenate([start_frame, sep, gf], axis=1))
    w = imageio.get_writer(str(path), fps=FPS_SBS)
    for f in frames:
        w.append_data(f)
    w.close()


def main():
    device  = "cuda:0"
    out_dir = REPO / "ablation_results" / "hb_stack_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LoRA checkpoint: {LORA_CKPT}")
    print("Loading Hummingbird stack model...")
    hb_model, hb_reneg = load_hummingbird(device)
    print("Ready.\n")

    for task_id, task_label, prompt in TASKS:
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_label}  |  {NUM_DEMOS} demos")
        print(f"  prompt: \"{prompt}\"")
        print(f"{'='*60}")

        for demo_idx in range(NUM_DEMOS):
            try:
                start_frame = load_frame(task_id, demo_idx, step=0)
            except Exception as e:
                print(f"  demo {demo_idx:02d}: SKIP ({e})")
                continue

            print(f"  demo {demo_idx:02d}: generating {NUM_FRAMES} frames ...", end=" ", flush=True)
            gen = generate(hb_model, hb_reneg, start_frame, prompt, device)
            print("done")

            sbs_path = out_dir / f"task{task_id}_{task_label}_demo{demo_idx:02d}_start_vs_gen.mp4"
            save_sidebyside(start_frame, gen, sbs_path)
            print(f"           saved: {sbs_path.name}")

    print(f"\nDone. All videos in:")
    print(f"  {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
