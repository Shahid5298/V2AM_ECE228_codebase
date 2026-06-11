"""Training loop for the flow matching action head."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .config import FlowMatchingConfig
from .model import FlowMatchingActionHead
from ..videomae_encoder import VideoMAEFeatureExtractor


def build_video_features(
    batch: dict[str, torch.Tensor],
    videomae: VideoMAEFeatureExtractor,
    config: FlowMatchingConfig,
) -> torch.Tensor:
    video_feature_chunks = []
    if config.include_history_frames:
        video_feature_chunks.extend([
            videomae(batch["pixel_values_history"].to(config.device)),
            videomae(batch["pixel_values_wrist_history"].to(config.device)),
        ])
    if config.condition_on_future_video:
        video_feature_chunks.extend([
            videomae(batch["pixel_values"].to(config.device)),
            videomae(batch["pixel_values_wrist"].to(config.device)),
        ])
    if not video_feature_chunks:
        raise ValueError("At least one visual conditioning stream must be enabled.")
    return torch.cat(video_feature_chunks, dim=1)


def train_one_epoch(
    model: FlowMatchingActionHead,
    videomae: VideoMAEFeatureExtractor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: FlowMatchingConfig,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, float]:
    """One epoch of flow matching training.

    Continuous action dims are trained with flow matching; the gripper
    command is trained with BCE on a separate binary head.
    """
    model.train()

    total_loss = 0.0
    total_flow_loss = 0.0
    total_gripper_loss = 0.0
    total_smooth_loss = 0.0
    n_batches = 0

    a_mean = action_mean.to(config.device)
    a_std = action_std.to(config.device)

    for batch in dataloader:
        proprio = batch["proprio"].to(config.device)
        actions = batch["actions"].to(config.device)

        continuous_actions = (actions[:, :, :config.continuous_action_dim] - a_mean) / a_std
        gripper_actions = (actions[:, :, config.continuous_action_dim:] + 1.0) / 2.0

        video_features = build_video_features(batch, videomae, config)

        losses = model.compute_losses(
            video_features, proprio, continuous_actions, gripper_actions,
        )
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_flow_loss += losses["flow_loss"].item()
        total_gripper_loss += losses["gripper_loss"].item()
        total_smooth_loss += losses["smooth_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "flow_loss": total_flow_loss / n_batches,
        "gripper_loss": total_gripper_loss / n_batches,
        "smooth_loss": total_smooth_loss / n_batches,
        "lr": optimizer.param_groups[0]["lr"],
    }
