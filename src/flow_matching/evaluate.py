"""Evaluation for the flow matching action head (ODE-based inference)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import FlowMatchingConfig
from .model import FlowMatchingActionHead
from .train import build_video_features
from ..videomae_encoder import VideoMAEFeatureExtractor


@torch.no_grad()
def evaluate_per_timestep(
    model: FlowMatchingActionHead,
    videomae: VideoMAEFeatureExtractor,
    dataloader: DataLoader,
    config: FlowMatchingConfig,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, float]:
    """Evaluate using flow sampling for continuous actions and BCE gripper logits."""
    model.eval()

    a_mean = action_mean.to(config.device)
    a_std = action_std.to(config.device)

    all_preds = []
    all_targets = []

    for batch in dataloader:
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)

        video_features = build_video_features(batch, videomae, config)

        pred_cont_norm, gripper_logit = model.sample_actions(video_features, proprio)

        pred_cont = pred_cont_norm * a_std + a_mean
        gripper_pred = (torch.sigmoid(gripper_logit) > 0.5).float() * 2.0 - 1.0
        pred = torch.cat([pred_cont, gripper_pred], dim=-1)

        all_preds.append(pred.cpu())
        all_targets.append(actions.cpu())

    all_preds = torch.cat(all_preds, dim=0)      # (N, C, 7)
    all_targets = torch.cat(all_targets, dim=0)  # (N, C, 7)

    # Overall MSE
    mse = F.mse_loss(all_preds, all_targets).item()

    # Per-dimension MSE
    per_dim_mse = ((all_preds - all_targets) ** 2).mean(dim=(0, 1))  # (7,)

    # Per-timestep metrics
    per_timestep_mse = ((all_preds - all_targets) ** 2).mean(dim=(0, 2))  # (C,)

    # Gripper accuracy: threshold at 0 (since gripper is in {-1, +1})
    gripper_pred_binary = (all_preds[:, :, 6] > 0).float() * 2.0 - 1.0
    gripper_acc = (gripper_pred_binary == all_targets[:, :, 6]).float().mean().item()
    per_timestep_gripper_acc = (
        gripper_pred_binary == all_targets[:, :, 6]
    ).float().mean(dim=0)  # (C,)

    # R²
    target_var = all_targets.var(dim=(0, 1)).sum().item()
    r2 = 1.0 - (mse * 7) / max(target_var, 1e-8)

    model.train()

    return {
        "mse": mse,
        "per_dim_mse": per_dim_mse.tolist(),
        "per_timestep_mse": per_timestep_mse.tolist(),
        "per_timestep_gripper_accuracy": per_timestep_gripper_acc.tolist(),
        "gripper_accuracy": gripper_acc,
        "r2": r2,
    }


def compute_baselines(
    dataloader: DataLoader,
    action_mean_global: np.ndarray,
    task_action_means: dict[int, np.ndarray],
) -> dict[str, dict]:
    """Compute global-mean and per-task-mean baselines."""
    all_targets = []
    all_task_ids = []

    for batch in dataloader:
        all_targets.append(batch["actions"])
        all_task_ids.append(batch["task_ids"])

    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_task_ids = torch.cat(all_task_ids, dim=0).numpy()

    results = {}

    global_pred = np.broadcast_to(action_mean_global, all_targets.shape)
    results["global_mean"] = {"mse": float(np.mean((global_pred - all_targets) ** 2))}

    task_pred = np.zeros_like(all_targets)
    for i, tid in enumerate(all_task_ids):
        task_pred[i] = task_action_means[int(tid)]
    results["per_task_mean"] = {"mse": float(np.mean((task_pred - all_targets) ** 2))}

    return results
