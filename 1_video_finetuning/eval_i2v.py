"""
Evaluate I2V LoRA quality against ground-truth clips.

For each task_id × seed combination:
  1. Load a random episode (determined by seed)
  2. Select 10 equally-spaced conditioning frames
  3. Generate 8-frame predicted clips via the I2V model
  4. Compare each predicted clip to the corresponding GT clip
     using PSNR, SSIM, and LPIPS
  5. Average metrics across all conditioning frames → one score-set per seed

Finally, compute mean ± stddev across seeds for each task_id and print a table.
"""

import os
import sys
import json
import glob
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_lightning import seed_everything

# ── Metrics ──────────────────────────────────────────────────────────
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# Add paths (same as run_episode_inference.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hum_infer"))

# ── Reuse helpers from run_episode_inference ─────────────────────────
from run_episode_inference import (
    load_episode,
    select_conditioning_frames,
    load_model,
    generate_for_frame,
    tensor_to_pil_frames,
    RESOLUTION,
    VIDEO_LENGTH,
    FRAME_STRIDE,
    DDIM_STEPS,
    GUIDANCE_SCALE,
    NUM_COND_FRAMES,
)

# ── Metric helpers ───────────────────────────────────────────────────

_lpips_net = None


def get_lpips_net(device="cuda"):
    """Lazy-load LPIPS network (AlexNet, fastest)."""
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net="alex").to(device)
        _lpips_net.eval()
    return _lpips_net


def compute_frame_metrics(pred_pil: Image.Image, gt_pil: Image.Image, device="cuda"):
    """Compute PSNR, SSIM, LPIPS between two PIL images.

    Returns dict with keys: psnr, ssim, lpips
    """
    pred_np = np.array(pred_pil).astype(np.float64)
    gt_np = np.array(gt_pil).astype(np.float64)

    # PSNR
    psnr_val = psnr_fn(gt_np, pred_np, data_range=255.0)

    # SSIM (multichannel)
    ssim_val = ssim_fn(gt_np, pred_np, data_range=255.0, channel_axis=2)

    # LPIPS (expects [-1, 1] tensors, shape (1, 3, H, W))
    def pil_to_lpips_tensor(img):
        t = torch.from_numpy(np.array(img)).float() / 255.0  # [0,1]
        t = t * 2.0 - 1.0  # [-1,1]
        t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return t.to(device)

    net = get_lpips_net(device)
    with torch.no_grad():
        lpips_val = net(pil_to_lpips_tensor(pred_pil), pil_to_lpips_tensor(gt_pil))
    lpips_val = lpips_val.item()

    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def compute_clip_metrics(pred_frames, gt_frames, device="cuda"):
    """Average per-frame metrics across a clip pair.

    Args:
        pred_frames: list of PIL Images (predicted clip)
        gt_frames:   list of PIL Images (ground truth clip)

    Returns dict with averaged psnr, ssim, lpips.
    """
    assert len(pred_frames) == len(gt_frames), (
        f"Clip length mismatch: {len(pred_frames)} vs {len(gt_frames)}"
    )
    all_metrics = [compute_frame_metrics(p, g, device) for p, g in zip(pred_frames, gt_frames)]
    return {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0]
    }


# ── Main evaluation loop ────────────────────────────────────────────

def evaluate_single_run(model, device, task_id, seed, num_cond_frames=10):
    """Run evaluation for a single (task_id, seed) pair.

    Returns dict with averaged metrics across all conditioning frames.
    """
    # Seed everything for reproducibility  
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)

    # Load episode
    all_frames, task_annotation, task_idx, episode_path = load_episode(task_ids=[task_id])
    print(f"    Episode: {Path(episode_path).name}, frames: {len(all_frames)}, "
          f"task: \"{task_annotation}\"")

    # Select conditioning frames
    samples = select_conditioning_frames(all_frames, num_cond_frames)

    # Seed model inference deterministically
    seed_everything(seed)

    clip_metrics = []
    cond_bar = tqdm(enumerate(samples), total=len(samples),
                    desc="      Cond frames", leave=False)
    for i, (ci, gt_indices) in cond_bar:
        # GT clip
        gt_clip_frames = [all_frames[idx] for idx in gt_indices]

        # Generate predicted clip
        cond_frame = all_frames[ci]
        video_tensor = generate_for_frame(model, cond_frame, task_annotation, device)
        pred_frames = tensor_to_pil_frames(video_tensor)

        # Compute metrics
        metrics = compute_clip_metrics(pred_frames, gt_clip_frames, device)
        clip_metrics.append(metrics)
        cond_bar.set_postfix(PSNR=f"{metrics['psnr']:.1f}",
                             SSIM=f"{metrics['ssim']:.3f}",
                             LPIPS=f"{metrics['lpips']:.3f}")

    # Average across all conditioning frames
    avg = {k: np.mean([m[k] for m in clip_metrics]) for k in clip_metrics[0]}
    return avg


def main():
    parser = argparse.ArgumentParser(description="Evaluate I2V LoRA vs ground truth")
    parser.add_argument(
        "--task_ids", type=str, default="0,5,7,11,14,17",
        help="Comma-separated task IDs (default: 0,5,7,11,14,17)",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5,
        help="Number of random seeds per task (default: 5)",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results CSV (default: same dir as script)",
    )
    args = parser.parse_args()

    task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    num_seeds = args.num_seeds
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    # Override LoRA path if provided
    if args.lora_path is not None:
        import run_episode_inference
        run_episode_inference.LORA_PATH = Path(args.lora_path)

    # Generate reproducible random seeds
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 100000, size=num_seeds).tolist()
    print(f"Seeds: {seeds}")

    # Load model once
    print("=" * 60)
    print("Loading I2V model + LoRA ...")
    print("=" * 60)
    model, device = load_model()

    # ── Collect results ──────────────────────────────────────────────
    rows = []  # each row: {task_id, seed, psnr, ssim, lpips}

    total_runs = len(task_ids) * num_seeds
    overall_bar = tqdm(total=total_runs, desc="Overall progress")

    for tid in task_ids:
        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"TASK {tid}")
        tqdm.write(f"{'=' * 60}")
        for si, seed in enumerate(seeds):
            tqdm.write(f"  Seed {si+1}/{num_seeds} (seed={seed})")
            metrics = evaluate_single_run(model, device, tid, seed)
            rows.append({
                "task_id": tid,
                "seed": seed,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
            })
            tqdm.write(f"    => PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
                       f"LPIPS={metrics['lpips']:.4f}")
            overall_bar.update(1)

    overall_bar.close()

    # ── Build results table ──────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Per-task stats (ddof=0 so single-seed runs give 0 instead of NaN)
    summary_rows = []
    for tid in task_ids:
        sub = df[df["task_id"] == tid]
        summary_rows.append({
            "task_id": tid,
            "PSNR (mean)": sub["psnr"].mean(),
            "PSNR (std)": sub["psnr"].std(ddof=0),
            "SSIM (mean)": sub["ssim"].mean(),
            "SSIM (std)": sub["ssim"].std(ddof=0),
            "LPIPS (mean)": sub["lpips"].mean(),
            "LPIPS (std)": sub["lpips"].std(ddof=0),
        })

    # Overall
    summary_rows.append({
        "task_id": "ALL",
        "PSNR (mean)": df["psnr"].mean(),
        "PSNR (std)": df["psnr"].std(ddof=0),
        "SSIM (mean)": df["ssim"].mean(),
        "SSIM (std)": df["ssim"].std(ddof=0),
        "LPIPS (mean)": df["lpips"].mean(),
        "LPIPS (std)": df["lpips"].std(ddof=0),
    })

    summary_df = pd.DataFrame(summary_rows)

    # ── Print table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    header = (
        f"{'Task':>6}  |  {'PSNR':^16}  |  {'SSIM':^16}  |  {'LPIPS':^16}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for _, row in summary_df.iterrows():
        tid = row["task_id"]
        tid_str = f"{tid:>6}" if isinstance(tid, int) else f"{tid:>6}"
        print(
            f"{tid_str}  |  "
            f"{row['PSNR (mean)']:7.2f} ± {row['PSNR (std)']:5.2f}  |  "
            f"{row['SSIM (mean)']:7.4f} ± {row['SSIM (std)']:5.4f}  |  "
            f"{row['LPIPS (mean)']:7.4f} ± {row['LPIPS (std)']:5.4f}"
        )
    print(sep)

    # ── Save CSV ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    raw_csv = output_dir / "eval_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\nRaw results saved to: {raw_csv}")

    summary_csv = output_dir / "eval_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to:     {summary_csv}")


if __name__ == "__main__":
    main()
