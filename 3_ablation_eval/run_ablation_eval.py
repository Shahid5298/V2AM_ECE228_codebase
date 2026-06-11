"""
V2AM Ablation Evaluation Script
================================
Loads a trained V2AM baseline checkpoint and evaluates three inference variants
on the validation set without any retraining.

Modes
-----
  euler         -- baseline: fixed 10-step Euler ODE (baseline)
  ensemble      -- flow ensembling: k Euler samples averaged (Exp 1)
  dopri5        -- adaptive Neural ODE solver from torchdiffeq (Exp 2)
  dopri5-ensemble -- both: dopri5 integrated + k-ensemble (Exp 4 proxy)

Usage
-----
  conda run -n ml python run_ablation_eval.py \\
      --checkpoint checkpoint/model_task0_coffeeinsert_history1_future_epoch26_R2_0723.pt \\
      --action-stats checkpoint/action_stats_task0_coffeeinsert_1task.pt \\
      --mode euler

  # run all modes back to back:
  conda run -n ml python run_ablation_eval.py --mode all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor

# ── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.flow_matching.config import FlowMatchingConfig
from src.flow_matching.model import FlowMatchingActionHead
from src.flow_matching.train import build_video_features
from src.mimicgen_dataset import make_mimicgen_dataloaders
from src.videomae_encoder import VideoMAEFeatureExtractor


# ── Config that exactly matches the saved checkpoint ────────────────────────
# The checkpoint was trained with:
#   --task-id 0  (coffee pod insertion, task_index=0 only)
#   --history-num-frames 1
#   (future video enabled by default)
#   hidden_dim=384, num_layers=4, num_heads=6  (inferred from state dict)

def build_config(task_id: int = 0) -> FlowMatchingConfig:
    cfg = FlowMatchingConfig()
    cfg.task_filter_indices = (task_id,)
    cfg.history_num_frames = 1
    cfg.include_history_frames = True
    cfg.condition_on_future_video = True
    cfg.hidden_dim = 384
    cfg.num_layers = 4
    cfg.num_heads = 6
    cfg.num_denoising_steps = 10   # original inference step count
    cfg.num_workers = 4
    return cfg


# ── Evaluation helpers ───────────────────────────────────────────────────────

def compute_metrics(
    all_preds: torch.Tensor,   # (N, C, 7) — continuous + gripper
    all_targets: torch.Tensor, # (N, C, 7)
) -> dict[str, float]:
    mse = F.mse_loss(all_preds, all_targets).item()
    per_dim_mse = ((all_preds - all_targets) ** 2).mean(dim=(0, 1))  # (7,)
    target_var = all_targets.var(dim=(0, 1)).sum().item()
    r2 = 1.0 - (mse * 7) / max(target_var, 1e-8)
    gripper_pred = (all_preds[:, :, 6] > 0).float() * 2.0 - 1.0
    gripper_acc = (gripper_pred == all_targets[:, :, 6]).float().mean().item()

    # Jerk = second finite difference of the continuous action chunk (dims 0:6)
    cont_preds = all_preds[:, :, :6]                                     # (N, C, 6)
    jerk = cont_preds[:, 2:] - 2 * cont_preds[:, 1:-1] + cont_preds[:, :-2]  # (N, C-2, 6)
    mean_jerk = (jerk ** 2).mean().item()

    return {
        "mse": mse,
        "r2": r2,
        "gripper_accuracy": gripper_acc,
        "per_dim_mse": per_dim_mse.tolist(),
        "mean_jerk": mean_jerk,
    }


@torch.no_grad()
def run_euler(model, videomae, val_loader, config, a_mean, a_std) -> dict:
    """Baseline Euler ODE — exact replica of baseline inference."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in val_loader:
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)
        video_features = build_video_features(batch, videomae, config)

        cont_norm, grip_logit = model.sample_actions(video_features, proprio)

        cont = cont_norm * a_std + a_mean
        grip = (torch.sigmoid(grip_logit) > 0.5).float() * 2.0 - 1.0
        pred = torch.cat([cont, grip], dim=-1)

        all_preds.append(pred.cpu())
        all_targets.append(actions.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["nfe"] = config.num_denoising_steps
    metrics["nfe_std"] = 0.0
    metrics["per_step_variance"] = None
    return metrics


@torch.no_grad()
def run_ensemble(model, videomae, val_loader, config, a_mean, a_std, k: int = 5) -> dict:
    """Flow ensembling: k Euler samples averaged, per-step variance collected."""
    model.eval()
    all_preds, all_targets = [], []
    all_variances = []

    for batch in val_loader:
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)
        video_features = build_video_features(batch, videomae, config)

        mean_cont_norm, mean_grip_logit, step_var = model.sample_actions_ensemble(
            video_features, proprio, k=k,
        )

        cont = mean_cont_norm * a_std + a_mean
        grip = (torch.sigmoid(mean_grip_logit) > 0.5).float() * 2.0 - 1.0
        pred = torch.cat([cont, grip], dim=-1)

        all_preds.append(pred.cpu())
        all_targets.append(actions.cpu())
        all_variances.append(step_var.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["nfe"] = config.num_denoising_steps * k
    metrics["nfe_std"] = 0.0

    # Mean per-step variance across all batches
    mean_variance = torch.stack(all_variances).mean(dim=0)  # (C,)
    metrics["per_step_variance"] = mean_variance.tolist()
    return metrics


@torch.no_grad()
def run_dopri5(model, videomae, val_loader, config, a_mean, a_std,
               rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    """Adaptive Neural ODE with dopri5 solver. Tracks NFE per sample."""
    model.eval()
    all_preds, all_targets = [], []
    all_nfe = []

    for batch in val_loader:
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)
        video_features = build_video_features(batch, videomae, config)

        cont_norm, grip_logit, nfe = model.sample_actions_dopri5(
            video_features, proprio, rtol=rtol, atol=atol,
        )

        cont = cont_norm * a_std + a_mean
        grip = (torch.sigmoid(grip_logit) > 0.5).float() * 2.0 - 1.0
        pred = torch.cat([cont, grip], dim=-1)

        all_preds.append(pred.cpu())
        all_targets.append(actions.cpu())
        all_nfe.append(nfe)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["nfe"] = float(np.mean(all_nfe))
    metrics["nfe_std"] = float(np.std(all_nfe))
    metrics["per_step_variance"] = None
    return metrics


@torch.no_grad()
def run_dopri5_ensemble(model, videomae, val_loader, config, a_mean, a_std,
                        k: int = 5, rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    """dopri5 integration + k-ensemble averaging (Exp 4)."""
    model.eval()
    all_preds, all_targets = [], []
    all_nfe = []
    all_variances = []

    for batch in val_loader:
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)
        video_features = build_video_features(batch, videomae, config)

        batch_continuous = []
        batch_gripper = []
        batch_nfe = 0

        for _ in range(k):
            cont_norm, grip_logit, nfe = model.sample_actions_dopri5(
                video_features, proprio, rtol=rtol, atol=atol,
            )
            batch_continuous.append(cont_norm)
            batch_gripper.append(grip_logit)
            batch_nfe += nfe

        stacked = torch.stack(batch_continuous)           # (k, B, C, cont_dim)
        mean_cont_norm = stacked.mean(dim=0)
        step_var = stacked.var(dim=0).mean(dim=(0, 2))    # (C,)
        mean_grip = torch.stack(batch_gripper).mean(dim=0)

        cont = mean_cont_norm * a_std + a_mean
        grip = (torch.sigmoid(mean_grip) > 0.5).float() * 2.0 - 1.0
        pred = torch.cat([cont, grip], dim=-1)

        all_preds.append(pred.cpu())
        all_targets.append(actions.cpu())
        all_nfe.append(batch_nfe)
        all_variances.append(step_var.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["nfe"] = float(np.mean(all_nfe))
    metrics["nfe_std"] = float(np.std(all_nfe))
    metrics["per_step_variance"] = torch.stack(all_variances).mean(dim=0).tolist()
    return metrics


# ── Pretty printer ───────────────────────────────────────────────────────────

def print_row(label: str, m: dict):
    var_str = "✓" if m["per_step_variance"] is not None else "—"
    nfe_str = f"{m['nfe']:.1f}±{m['nfe_std']:.1f}" if m["nfe_std"] > 0 else f"{int(m['nfe'])}"
    print(f"  {label:<30} | {m['mse']:.5f}  | {m['r2']:.4f} | {m['gripper_accuracy']*100:.2f}%  | {nfe_str:<12} | {m['mean_jerk']:.6f} | {var_str}")


def print_dim_breakdown(label: str, m: dict):
    dim_names = ["pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z", "gripper"]
    print(f"\n  Per-dim MSE ({label}):")
    for name, val in zip(dim_names, m["per_dim_mse"]):
        print(f"    {name:8s}: {val:.6f}")


def print_variance(label: str, m: dict):
    if m["per_step_variance"] is None:
        return
    print(f"\n  Per-step variance ({label}):")
    for i, v in enumerate(m["per_step_variance"]):
        bar = "█" * int(v * 500)
        print(f"    t+{i:02d}: {v:.6f}  {bar}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint/model_task0_coffeeinsert_history1_future_epoch26_R2_0723.pt"),
    )
    parser.add_argument(
        "--action-stats",
        type=Path,
        default=Path("checkpoint/action_stats_task0_coffeeinsert_1task.pt"),
    )
    parser.add_argument(
        "--mode",
        choices=["euler", "ensemble", "dopri5", "dopri5-ensemble", "all"],
        default="all",
    )
    parser.add_argument("--ensemble-k", type=int, default=5)
    parser.add_argument("--dopri5-rtol", type=float, default=1e-3)
    parser.add_argument("--dopri5-atol", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Task index to evaluate on (must match training task)")
    args = parser.parse_args()

    config = build_config(task_id=args.task_id)
    config.device = args.device

    task_labels = {0:"coffee pod insertion",10:"cube stacking",12:"stack three cubes"}
    task_label = task_labels.get(args.task_id, f"task_{args.task_id}")
    print("\n" + "=" * 80)
    print(f"V2AM ABLATION — task {args.task_id} ({task_label})")
    print("=" * 80)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Action stats: {args.action_stats}")
    print(f"  Mode       : {args.mode}")
    print(f"  Device     : {args.device}")
    print(f"  Task ID    : {args.task_id} ({task_label})")

    # ── Load action normalization stats ──────────────────────────────────
    stats = torch.load(args.action_stats, weights_only=True)
    a_mean = stats["mean"].to(args.device)
    a_std = stats["std"].to(args.device)

    # ── Load model ───────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = FlowMatchingActionHead(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.2f}M params | epoch {ckpt['epoch']} | "
          f"saved MSE={ckpt['metrics']['mse']:.5f} R²={ckpt['metrics']['r2']:.4f}")

    # ── Load VideoMAE ────────────────────────────────────────────────────
    print("\n[2/4] Loading frozen VideoMAE encoder...")
    processor = VideoMAEImageProcessor.from_pretrained(config.model_dir)
    videomae = VideoMAEFeatureExtractor(
        config.model_dir,
        layer_idx=config.videomae_layer,
        num_frames_expected=config.num_frames,
        device=args.device,
    ).to(args.device)

    # ── Build val dataloader ─────────────────────────────────────────────
    print("\n[3/4] Building validation dataloader...")
    _, val_loader = make_mimicgen_dataloaders(config, processor)
    print(f"  Val windows: {len(val_loader.dataset)}")

    # ── Run ablations ────────────────────────────────────────────────────
    print("\n[4/4] Running inference...\n")
    results = {}

    modes_to_run = (
        ["euler", "ensemble", "dopri5", "dopri5-ensemble"]
        if args.mode == "all" else [args.mode]
    )

    for mode in modes_to_run:
        print(f"  → {mode}...", end=" ", flush=True)
        t0 = time.time()
        if mode == "euler":
            m = run_euler(model, videomae, val_loader, config, a_mean, a_std)
        elif mode == "ensemble":
            m = run_ensemble(model, videomae, val_loader, config, a_mean, a_std, k=args.ensemble_k)
        elif mode == "dopri5":
            m = run_dopri5(model, videomae, val_loader, config, a_mean, a_std,
                           rtol=args.dopri5_rtol, atol=args.dopri5_atol)
        elif mode == "dopri5-ensemble":
            m = run_dopri5_ensemble(model, videomae, val_loader, config, a_mean, a_std,
                                    k=args.ensemble_k, rtol=args.dopri5_rtol, atol=args.dopri5_atol)
        results[mode] = m
        print(f"done ({time.time()-t0:.1f}s)")

    # ── Print table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"ABLATION TABLE — task {args.task_id}, {task_label}")
    print("=" * 80)
    print(f"  {'Variant':<30} | {'Val MSE':>8} | {'R²':>6} | {'Grip Acc':>8} | {'NFE':>12} | {'Mean Jerk':>10} | Variance")
    print("  " + "-" * 90)

    label_map = {
        "euler":           "V2AM baseline (Euler-10)",
        "ensemble":        f"+ Ensembling (k={args.ensemble_k})",
        "dopri5":          "+ Neural ODE (dopri5)",
        "dopri5-ensemble": f"V2AM full (dopri5 + k={args.ensemble_k})",
    }

    for mode in modes_to_run:
        print_row(label_map[mode], results[mode])

    # ── Per-dimension breakdown for each completed mode ──────────────────
    for mode in modes_to_run:
        print_dim_breakdown(label_map[mode], results[mode])

    # ── Per-step variance for ensemble modes ────────────────────────────
    for mode in modes_to_run:
        print_variance(label_map[mode], results[mode])

    # ── Save results to disk ─────────────────────────────────────────────
    out_path = Path("ablation_results.pt")
    torch.save(results, out_path)
    print(f"\n  Results saved to {out_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
