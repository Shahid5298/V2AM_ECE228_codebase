"""
Run I2V LoRA inference across an entire LIBERO episode.

Workflow
--------
1. Pick a random episode from the LIBERO dataset.
2. Save the FULL ground-truth video (all frames) as a GIF.
3. Select 10 equally-spaced ground-truth frames as conditioning images.
4. For each conditioning frame, generate 8 future predicted frames using
   the I2V model with LoRA weights.
5. For each conditioning frame, also extract the 8-frame ground-truth clip
   (using the same frame_stride=2 used in training) so you can compare
   GT vs. prediction side by side.
6. Save everything into an organised output directory.

Output structure
-----------------
output_episode/
  full_ground_truth.gif          – every frame of the episode
  summary.txt                    – metadata
  sample_00/
    cond_frame.png               – the conditioning image
    gt_clip.gif                  – 8-frame GT clip starting at this frame
    predicted_clip.gif           – 8-frame model prediction
    predicted_clip.mp4           – same prediction as MP4
  sample_01/
    ...
  ...
"""

import os
import sys
import io
import json
import glob
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_lightning import seed_everything
import torchvision
import torchvision.transforms as transforms

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hum_infer"))

# # ── Configuration ──────────────────────────────────────────────────────
# SEED = 42
# OUTPUT_DIR = Path(__file__).parent / "output_episode"
# BASE_CACHE = Path.home() / ".cache/huggingface/hub/datasets--physical-intelligence--libero"
# SNAPSHOT = BASE_CACHE / "snapshots/a4336d589d589045d1c56423ffdf3b88a0e19b1f"
# DATA_DIR = SNAPSHOT / "data"
# META_DIR = SNAPSHOT / "meta"


# ── Configuration ──────────────────────────────────────────────────────
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "output_episode_mimicgen_full_data"
BASE_DIR = Path.home() / "mimicgen_training_data_100"

DATA_DIR = BASE_DIR / "data"
META_DIR = BASE_DIR / "meta"


# Model paths
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "../configs/inference_i2v_512_v2.0_distil.yaml"
CKPT_PATH = SCRIPT_DIR / "../hum_infer/checkpoints/stage_1.ckpt"
UNET_PATH = SCRIPT_DIR / "../hum_infer/checkpoints/unet.pt"
IMG_PROJ_PATH = SCRIPT_DIR / "../hum_infer/checkpoints/img_proj.pt"
RENEG_PATH = SCRIPT_DIR / "../hum_infer/checkpoints/reneg_checkpoint.bin"

# LoRA weights
# LORA_PATH = SCRIPT_DIR / "checkpoints_mimicgen_t10/checkpoint-3500"
LORA_PATH = SCRIPT_DIR / "checkpoints_full_data/epoch-9"

# Video settings (must match training config)
VIDEO_LENGTH = 8        # frames to predict
FRAME_STRIDE = 2        # stride used during training
RESOLUTION = (256, 256)
NUM_COND_FRAMES = 10    # number of equally-spaced conditioning frames
DDIM_STEPS = 16
GUIDANCE_SCALE = 7.5
FPS_OUT = 8

# ── Helpers ────────────────────────────────────────────────────────────

def decode_image(image_data):
    """Decode image from various formats."""
    if isinstance(image_data, dict) and 'bytes' in image_data:
        return Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
    elif isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    elif isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data.astype(np.uint8)).convert('RGB')
    elif isinstance(image_data, Image.Image):
        return image_data.convert('RGB')
    else:
        raise ValueError(f"Unknown image format: {type(image_data)}")


RESIZE_TRANSFORM = transforms.Compose([
    transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(RESOLUTION),
])


def save_gif(frames, path, duration=125):
    """Save list of PIL Images as GIF (125 ms/frame = 8 fps)."""
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:],
        duration=duration, loop=0,
    )


def tensor_to_pil_frames(video_tensor):
    """Convert (T, C, H, W) tensor in [0,1] to list of PIL Images."""
    frames = []
    for t in range(video_tensor.shape[0]):
        frame = video_tensor[t].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))
    return frames


# ── Step 1: Load entire episode ──────────────────────────────────────

def load_episode(task_ids=None):
    """Load a random episode, optionally filtered by task_ids.
    
    Args:
        task_ids: Optional list of task_index values to filter by.
                  If None, picks from all episodes.
    
    Returns:
        all_frames, task_annotation, task_idx, episode_path
    """
    # Task prompts
    tasks_file = META_DIR / "tasks.jsonl"
    task_prompts = {}
    with open(tasks_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            task_prompts[entry['task_index']] = entry['task']

    # Find all episodes
    episode_files = sorted(glob.glob(str(DATA_DIR / "chunk-*" / "*.parquet")))
    if not episode_files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    # Filter by task_ids if specified
    if task_ids is not None:
        task_ids_set = set(task_ids)
        filtered = []
        print(f"  Filtering episodes to task_ids {task_ids}...")
        for ep_path in episode_files:
            try:
                df_meta = pd.read_parquet(ep_path, columns=['task_index'])
                if 'task_index' in df_meta.columns and len(df_meta) > 0:
                    t_idx = int(df_meta.iloc[0]['task_index'])
                    if t_idx in task_ids_set:
                        filtered.append(ep_path)
            except Exception:
                pass
        print(f"  Found {len(filtered)} episodes matching task_ids {task_ids}")
        if not filtered:
            raise ValueError(f"No episodes found for task_ids {task_ids}")
        episode_files = filtered

    # Pick random episode
    episode_path = random.choice(episode_files)
    print(f"Episode: {episode_path}")

    df = pd.read_parquet(episode_path)
    num_frames = len(df)
    print(f"  Total frames: {num_frames}")

    # Detect image key
    image_key = None
    for key in ["observation.image", "image", "observation.images.top", "agentview_image"]:
        if key in df.columns:
            image_key = key
            break
    if image_key is None:
        for col in df.columns:
            if 'image' in col.lower():
                image_key = col
                break
    print(f"  Image key: {image_key}")

    # Task annotation
    if 'task_index' in df.columns:
        task_idx = int(df.iloc[0]['task_index'])
        task_annotation = task_prompts.get(task_idx, "robot arm manipulating objects on a table")
    else:
        task_idx = 0
        task_annotation = task_prompts.get(0, "robot arm manipulating objects on a table")

    # Decode ALL frames
    print(f"  Decoding all {num_frames} frames...")
    all_frames = []
    for i in range(num_frames):
        img = decode_image(df.iloc[i][image_key])
        img = RESIZE_TRANSFORM(img)
        all_frames.append(img)

    return all_frames, task_annotation, task_idx, episode_path


# ── Step 2: Select conditioning frames & extract GT clips ────────────

def select_conditioning_frames(all_frames, num_cond=10):
    """
    Pick `num_cond` equally-spaced frame indices such that each has
    enough room for an 8-frame GT clip (with FRAME_STRIDE) after it.
    Returns list of (cond_index, gt_clip_indices).
    """
    total = len(all_frames)
    required_after = (VIDEO_LENGTH - 1) * FRAME_STRIDE  # frames needed after cond
    max_start = total - 1 - required_after  # latest valid conditioning index

    if max_start < 0:
        raise ValueError(
            f"Episode too short ({total} frames) for video_length={VIDEO_LENGTH}, "
            f"frame_stride={FRAME_STRIDE}."
        )

    # Equally-spaced indices in [0, max_start]
    cond_indices = np.linspace(0, max_start, num_cond, dtype=int).tolist()

    samples = []
    for ci in cond_indices:
        gt_indices = list(range(ci, ci + VIDEO_LENGTH * FRAME_STRIDE, FRAME_STRIDE))
        # Clamp to valid range
        gt_indices = [min(idx, total - 1) for idx in gt_indices]
        samples.append((ci, gt_indices))

    return samples


# ── Step 3: Load model once ──────────────────────────────────────────

def load_model():
    """Load I2V base model + LoRA weights (done once)."""
    from inference_lora import load_i2v_model
    from lora_utils import load_lora_weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    print("  Loading I2V base model...")
    model = load_i2v_model(
        config_path=str(CONFIG_PATH),
        ckpt_path=str(CKPT_PATH),
        unet_path=str(UNET_PATH),
        img_proj_path=str(IMG_PROJ_PATH),
        device=device,
    )

    lora_path = str(LORA_PATH)
    print(f"  Loading LoRA weights from: {lora_path}")
    if hasattr(model, 'model'):
        model.model = load_lora_weights(model.model, lora_path)
    else:
        model = load_lora_weights(model, lora_path)

    return model, device


# ── Step 4: Generate predictions ─────────────────────────────────────

def generate_for_frame(model, cond_image, prompt, device):
    """Run I2V inference for a single conditioning image."""
    from inference_lora import generate_future_frames

    video = generate_future_frames(
        model=model,
        image=cond_image,
        prompt=prompt,
        height=RESOLUTION[0],
        width=RESOLUTION[1],
        video_length=VIDEO_LENGTH,
        ddim_steps=DDIM_STEPS,
        unconditional_guidance_scale=GUIDANCE_SCALE,
        device=device,
        reneg_path=str(RENEG_PATH),
    )
    return video  # (T, C, H, W) in [0,1]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run episode inference with optional task filtering")
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to filter by, e.g. '10,11'. Default: all tasks.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory. Default: checkpoints_mimicgen_stack/final",
    )
    args = parser.parse_args()

    # Parse task_ids
    task_ids = None
    if args.task_ids is not None:
        task_ids = [int(x.strip()) for x in args.task_ids.split(',')]

    # Override LoRA path if provided
    global LORA_PATH
    if args.lora_path is not None:
        LORA_PATH = Path(args.lora_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use system-time seed for random episode selection
    random.seed()

    # === Step 1: Load episode ===
    print("=" * 60)
    if task_ids:
        print(f"STEP 1: Loading random episode (filtered to task_ids {task_ids})")
    else:
        print("STEP 1: Loading random episode (all tasks)")
    print("=" * 60)
    all_frames, task_annotation, task_idx, episode_path = load_episode(task_ids=task_ids)
    total_frames = len(all_frames)

    # Now set deterministic seed for model inference
    seed_everything(SEED)

    # === Step 2: Save full ground truth ===
    print("\n" + "=" * 60)
    print("STEP 2: Saving full ground truth video")
    print("=" * 60)
    gt_full_path = OUTPUT_DIR / "full_ground_truth.gif"
    save_gif(all_frames, gt_full_path, duration=80)  # slightly faster for long episodes
    print(f"  Saved {total_frames}-frame GT GIF: {gt_full_path}")

    # === Step 3: Select conditioning frames ===
    print("\n" + "=" * 60)
    print(f"STEP 3: Selecting {NUM_COND_FRAMES} equally-spaced conditioning frames")
    print("=" * 60)
    samples = select_conditioning_frames(all_frames, NUM_COND_FRAMES)
    for i, (ci, gt_idx) in enumerate(samples):
        print(f"  Sample {i:02d}: cond_frame={ci}, gt_clip={gt_idx}")

    # === Step 4: Load model ===
    print("\n" + "=" * 60)
    print("STEP 4: Loading I2V model + LoRA")
    print("=" * 60)
    model, device = load_model()

    # === Step 5: Generate predictions for each conditioning frame ===
    print("\n" + "=" * 60)
    print("STEP 5: Running inference for each conditioning frame")
    print("=" * 60)

    for i, (ci, gt_indices) in enumerate(samples):
        sample_dir = OUTPUT_DIR / f"sample_{i:02d}"
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n--- Sample {i:02d}/{NUM_COND_FRAMES - 1} (frame {ci}) ---")

        # Save conditioning frame
        cond_frame = all_frames[ci]
        cond_path = sample_dir / "cond_frame.png"
        cond_frame.save(str(cond_path))

        # Save GT clip for this conditioning frame
        gt_clip_frames = [all_frames[idx] for idx in gt_indices]
        gt_clip_path = sample_dir / "gt_clip.gif"
        save_gif(gt_clip_frames, gt_clip_path)
        print(f"  GT clip saved: {gt_clip_path}")

        # Run inference
        video_tensor = generate_for_frame(model, cond_frame, task_annotation, device)

        # Save predicted clip as GIF
        pred_frames = tensor_to_pil_frames(video_tensor)
        pred_gif_path = sample_dir / "predicted_clip.gif"
        save_gif(pred_frames, pred_gif_path)
        print(f"  Predicted GIF saved: {pred_gif_path}")

        # Save predicted clip as MP4
        pred_mp4_path = sample_dir / "predicted_clip.mp4"
        video_uint8 = (video_tensor * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
        torchvision.io.write_video(str(pred_mp4_path), video_uint8, fps=FPS_OUT, video_codec='h264')
        print(f"  Predicted MP4 saved: {pred_mp4_path}")

    # === Step 6: Write summary ===
    summary = f"""Episode Inference Summary
{'=' * 40}
Episode:           {episode_path}
Task Index:        {task_idx}
Task Annotation:   {task_annotation}
Total Frames:      {total_frames}
Conditioning Pts:  {NUM_COND_FRAMES}
Video Length:      {VIDEO_LENGTH} frames
Frame Stride:      {FRAME_STRIDE}
Resolution:        {RESOLUTION[0]}x{RESOLUTION[1]}
LoRA Checkpoint:   {LORA_PATH}
DDIM Steps:        {DDIM_STEPS}
Guidance Scale:    {GUIDANCE_SCALE}
Seed:              {SEED}

Conditioning Frame Indices:
"""
    for i, (ci, gt_idx) in enumerate(samples):
        summary += f"  sample_{i:02d}: frame {ci} -> GT clip {gt_idx}\n"

    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)

    # Final output
    print("\n" + "=" * 60)
    print("DONE — OUTPUT FILES:")
    print("=" * 60)
    print(f"  Full GT GIF:  {gt_full_path}")
    print(f"  Summary:      {summary_path}")
    for i in range(NUM_COND_FRAMES):
        d = OUTPUT_DIR / f"sample_{i:02d}"
        print(f"  {d.name}/  cond_frame.png | gt_clip.gif | predicted_clip.gif | predicted_clip.mp4")
    print(f"\n  Task: \"{task_annotation}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
