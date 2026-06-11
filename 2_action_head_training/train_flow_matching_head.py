"""Train flow matching action head on LIBERO pick-and-place with frozen VideoMAE features."""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Make the repo root importable so that `src` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import wandb
from transformers import VideoMAEImageProcessor

from src.flow_matching.config import FlowMatchingConfig
from src.flow_matching.model import FlowMatchingActionHead
from src.flow_matching.train import train_one_epoch, build_video_features
from src.flow_matching.evaluate import evaluate_per_timestep, compute_baselines
from src.mimicgen_dataset import make_mimicgen_dataloaders
from src.videomae_encoder import VideoMAEFeatureExtractor
from src.utils import set_seed, save_checkpoint, load_checkpoint, get_cosine_schedule_with_warmup


def compute_action_stats_continuous(dataset, continuous_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std over continuous action dims only."""
    all_actions = []
    for i in range(len(dataset)):
        all_actions.append(dataset[i]["actions"][:, :continuous_dim])

    stacked = np.concatenate(all_actions, axis=0)
    mean = torch.from_numpy(stacked.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(stacked.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=1e-6)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true",
                        help="Overfit 1 batch for 200 steps to verify pipeline")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Train on a specific task ID only (e.g., 0)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Train on a specific set of task IDs (e.g., 0 1 2)")
    parser.add_argument("--no-history-frames", action="store_true",
                        help="Disable the past-to-current history video stream")
    parser.add_argument("--history-num-frames", type=int, default=None,
                        help="Override number of history frames when history is enabled")
    parser.add_argument("--no-future-video", action="store_true",
                        help="Disable the future-conditioned video stream")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="v2am-flow-matching",
                        help="Weights & Biases project name")
    parser.add_argument("--smooth-loss-weight", type=float, default=None,
                        help="λ for jerk penalty on reconstructed actions (0.1 recommended)")
    args = parser.parse_args()

    config = FlowMatchingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.smooth_loss_weight is not None:
        config.smooth_loss_weight = args.smooth_loss_weight
    if args.no_history_frames:
        config.include_history_frames = False
    if args.history_num_frames is not None:
        config.history_num_frames = args.history_num_frames
    if args.no_future_video:
        config.condition_on_future_video = False
    if not config.include_history_frames and not config.condition_on_future_video:
        raise ValueError("At least one visual conditioning stream must be enabled.")
    selected_task_ids = None
    if args.task_ids is not None:
        selected_task_ids = tuple(args.task_ids)
    elif args.task_id is not None:
        selected_task_ids = (args.task_id,)
    if selected_task_ids is not None:
        config.task_filter_indices = selected_task_ids
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if selected_task_ids is None:
        task_label = "all_tasks"
    else:
        task_label = "tasks_" + "_".join(str(tid) for tid in selected_task_ids)
    visual_parts = []
    if config.include_history_frames:
        visual_parts.append(f"history{config.history_num_frames}")
    if config.condition_on_future_video:
        visual_parts.append("future")
    visual_label = "_".join(visual_parts)
    run_name = f"{task_label}_{visual_label}_{timestamp}"
    config.output_dir = config.output_dir / run_name
    config.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    print(f"Config: {config}")
    print(f"Device: {config.device}")

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                **config.__dict__,
                "output_dir": str(config.output_dir),
            },
            dir=str(config.output_dir),
        )

    # 1. Build dataloaders
    print("\n[1/7] Building dataloaders...")
    processor = VideoMAEImageProcessor.from_pretrained(str(config.model_dir))
    train_loader, val_loader = make_mimicgen_dataloaders(config, processor)
    print(f"  Train: {len(train_loader.dataset)} windows, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} windows, {len(val_loader)} batches")

    # 2. Compute action normalization stats (continuous dims only)
    print("\n[2/7] Computing action normalization stats (continuous dims only)...")
    num_tasks = len(train_loader.dataset.task_names)
    stats_path = config.output_dir / f"action_stats_continuous_{num_tasks}tasks.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, weights_only=True)
        action_mean, action_std = stats["mean"], stats["std"]
        print(f"  Loaded cached stats from {stats_path}")
    else:
        action_mean, action_std = compute_action_stats_continuous(
            train_loader.dataset, config.continuous_action_dim,
        )
        torch.save({"mean": action_mean, "std": action_std}, stats_path)
        print(f"  Saved stats to {stats_path}")
    print(f"  Action mean: {action_mean.numpy()}")
    print(f"  Action std:  {action_std.numpy()}")

    # 3. Load frozen VideoMAE
    print("\n[3/7] Loading frozen VideoMAE encoder...")
    videomae = VideoMAEFeatureExtractor(
        config.model_dir, layer_idx=config.videomae_layer, num_frames_expected=config.num_frames, device=config.device,
    ).to(config.device)
    print(f"  Layer {config.videomae_layer} features: ({config.videomae_seq_len}, {config.videomae_hidden_dim})")

    # 4. Build flow matching action head
    print("\n[4/7] Building flow matching action head...")
    model = FlowMatchingActionHead(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Trainable params: {n_params:.1f}M")

    # 5. Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"\n[5/7] Optimizer: AdamW (lr={config.lr}, wd={config.weight_decay})")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")

    # --- Sanity check mode ---
    if args.sanity_check:
        print("\n=== SANITY CHECK: Overfitting 1 batch for 200 steps ===")
        batch = next(iter(train_loader))
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)

        a_mean = action_mean.to(config.device)
        a_std = action_std.to(config.device)
        continuous_actions = (actions[:, :, :config.continuous_action_dim] - a_mean) / a_std
        gripper_actions = (actions[:, :, config.continuous_action_dim:] + 1.0) / 2.0

        video_features = build_video_features(batch, videomae, config)

        model.train()
        sanity_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(200):
            losses = model.compute_losses(
                video_features, proprio, continuous_actions, gripper_actions,
            )
            loss = losses["loss"]
            sanity_opt.zero_grad()
            loss.backward()
            sanity_opt.step()
            if step % 20 == 0 or step == 199:
                print(
                    f"  Step {step:3d}: loss={loss.item():.6f} "
                    f"(flow={losses['flow_loss'].item():.6f} "
                    f"grip={losses['gripper_loss'].item():.6f})"
                )

        print(f"\n  Final loss: {loss.item():.6f}")
        if loss.item() < 0.1:
            print("  ✓ PASS: Loss dropped sufficiently. Flow matching pipeline is working.")
        else:
            print("  ✗ WARNING: Loss did not drop enough. Check for issues.")
        if wandb_run is not None:
            wandb.log({"sanity/final_loss": loss.item()})
            wandb.finish()
        return

    # --- Full training ---
    print(f"\n[6/7] Training for {config.epochs} epochs...")
    best_val_mse = float("inf")
    history = {"train_loss": [], "val_mse": [], "val_r2": [], "val_gripper_acc": []}

    for epoch in range(config.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, videomae, train_loader, optimizer, scheduler,
            config, action_mean, action_std,
        )

        val_metrics = evaluate_per_timestep(
            model, videomae, val_loader, config, action_mean, action_std,
        )

        elapsed = time.time() - t0
        history["train_loss"].append(train_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_gripper_acc"].append(val_metrics["gripper_accuracy"])

        smooth_str = f" smooth={train_metrics['smooth_loss']:.6f}" if config.smooth_loss_weight > 0 else ""
        print(
            f"  Epoch {epoch+1:3d}/{config.epochs} | "
            f"loss={train_metrics['loss']:.6f} "
            f"(flow={train_metrics['flow_loss']:.6f} "
            f"grip={train_metrics['gripper_loss']:.6f}{smooth_str}) | "
            f"val_mse={val_metrics['mse']:.6f} R²={val_metrics['r2']:.3f} "
            f"grip_acc={val_metrics['gripper_accuracy']:.3f} | "
            f"lr={train_metrics['lr']:.2e} | {elapsed:.1f}s"
        )

        if wandb_run is not None:
            wandb_metrics = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/flow_loss": train_metrics["flow_loss"],
                "train/gripper_loss": train_metrics["gripper_loss"],
                "train/smooth_loss": train_metrics["smooth_loss"],
                "train/lr": train_metrics["lr"],
                "val/mse": val_metrics["mse"],
                "val/r2": val_metrics["r2"],
                "val/gripper_accuracy": val_metrics["gripper_accuracy"],
                "time/epoch_seconds": elapsed,
            }
            for step_idx, mse_val in enumerate(val_metrics["per_timestep_mse"]):
                wandb_metrics[f"val_per_timestep/mse_t{step_idx:02d}"] = mse_val
            for step_idx, acc_val in enumerate(val_metrics["per_timestep_gripper_accuracy"]):
                wandb_metrics[f"val_per_timestep/gripper_acc_t{step_idx:02d}"] = acc_val
            wandb.log(wandb_metrics)

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                action_mean, action_std, config.output_dir / "best.pt",
            )
            print(f"    --> New best val MSE: {best_val_mse:.6f}")
            if wandb_run is not None:
                wandb.log({
                    "best/epoch": epoch + 1,
                    "best/val_mse": best_val_mse,
                    "best/val_r2": val_metrics["r2"],
                    "best/gripper_accuracy": val_metrics["gripper_accuracy"],
                })

    save_checkpoint(
        model, optimizer, scheduler, config.epochs - 1, val_metrics,
        action_mean, action_std, config.output_dir / "final.pt",
    )
    torch.save(history, config.output_dir / "history.pt")
    print(f"\n  Training complete. Best val MSE: {best_val_mse:.6f}")

    # 7. Baselines
    print("\n[7/7] Computing baselines...")
    all_actions = []
    for i in range(len(train_loader.dataset)):
        all_actions.append(train_loader.dataset[i]["actions"])
    all_actions_np = np.stack(all_actions, axis=0)
    global_mean_action = all_actions_np.mean(axis=(0, 1))

    task_means: dict[int, np.ndarray] = {}
    task_actions: dict[int, list] = {}
    for i in range(len(train_loader.dataset)):
        sample = train_loader.dataset[i]
        tid = sample["task_id"]
        if tid not in task_actions:
            task_actions[tid] = []
        task_actions[tid].append(sample["actions"])
    for tid, acts in task_actions.items():
        task_means[tid] = np.stack(acts, axis=0).mean(axis=(0, 1))

    baselines = compute_baselines(val_loader, global_mean_action, task_means)

    load_checkpoint(config.output_dir / "best.pt", model)
    best_metrics = evaluate_per_timestep(
        model, videomae, val_loader, config, action_mean, action_std,
    )

    print("\n" + "=" * 70)
    print(f"RESULTS: Flow Matching Action Head on LIBERO Pick-and-Place ({num_tasks} tasks)")
    print("=" * 70)
    print(f"{'Method':<30} | {'MSE':>10} | {'R²':>8} | {'Grip Acc':>8}")
    print("-" * 62)
    print(f"{'Global mean baseline':<30} | {baselines['global_mean']['mse']:>10.6f} | {'N/A':>8} | {'N/A':>8}")
    print(f"{'Per-task mean baseline':<30} | {baselines['per_task_mean']['mse']:>10.6f} | {'N/A':>8} | {'N/A':>8}")
    print(f"{'Flow matching head (ours)':<30} | {best_metrics['mse']:>10.6f} | {best_metrics['r2']:>8.3f} | {best_metrics['gripper_accuracy']:>8.3f}")

    print("\nPer-dimension MSE:")
    dim_names = ["pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z", "gripper"]
    for name, mse_val in zip(dim_names, best_metrics["per_dim_mse"]):
        print(f"  {name}: {mse_val:.6f}")

    print("\nPer-timestep metrics:")
    for step_idx, (mse_val, acc_val) in enumerate(zip(
        best_metrics["per_timestep_mse"], best_metrics["per_timestep_gripper_accuracy"],
    )):
        print(f"  t+{step_idx:02d}: mse={mse_val:.6f} grip_acc={acc_val:.3f}")

    if wandb_run is not None:
        summary = {
            "final/best_val_mse": best_val_mse,
            "final/best_val_r2": best_metrics["r2"],
            "final/best_gripper_accuracy": best_metrics["gripper_accuracy"],
            "baseline/global_mean_mse": baselines["global_mean"]["mse"],
            "baseline/per_task_mean_mse": baselines["per_task_mean"]["mse"],
        }
        for name, mse_val in zip(dim_names, best_metrics["per_dim_mse"]):
            summary[f"final/per_dim_mse/{name}"] = mse_val
        for step_idx, mse_val in enumerate(best_metrics["per_timestep_mse"]):
            summary[f"final/per_timestep_mse/t{step_idx:02d}"] = mse_val
        for step_idx, acc_val in enumerate(best_metrics["per_timestep_gripper_accuracy"]):
            summary[f"final/per_timestep_gripper_accuracy/t{step_idx:02d}"] = acc_val
        wandb.log(summary)
        wandb.finish()

    print(f"\nCheckpoints saved to: {config.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
