from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlowMatchingConfig
from .model import FlowMatchingActionHead


class HummingbirdLatentAdapter(nn.Module):
    """Project cached diffusion latents into a token sequence for the action head."""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.pool_spatial = config.hummingbird_pool_spatial
        self.proj_in = nn.Conv3d(
            config.hummingbird_latent_channels,
            config.hummingbird_adapter_dim,
            kernel_size=1,
        )
        self.norm = nn.GroupNorm(8, config.hummingbird_adapter_dim)
        self.proj_out = nn.Conv3d(
            config.hummingbird_adapter_dim,
            config.visual_feature_dim,
            kernel_size=1,
        )

    def forward(self, latent_video: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(latent_video)
        x = F.gelu(self.norm(x))
        x = self.proj_out(x)
        x = F.adaptive_avg_pool3d(
            x,
            output_size=(x.shape[2], self.pool_spatial, self.pool_spatial),
        )
        x = x.permute(0, 2, 3, 4, 1).reshape(latent_video.shape[0], -1, x.shape[1])
        return x


class HummingbirdLatentFlowMatchingPolicy(nn.Module):
    """Flow-matching action policy conditioned on cached Hummingbird latents."""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.latent_adapter = HummingbirdLatentAdapter(config)
        self.action_head = FlowMatchingActionHead(config)

    def compute_losses(
        self,
        latent_video: torch.Tensor,
        proprio: torch.Tensor,
        continuous_actions: torch.Tensor,
        gripper_actions: torch.Tensor,
        current_video_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent_features = self.latent_adapter(latent_video)
        if current_video_features is not None:
            latent_features = torch.cat([current_video_features, latent_features], dim=1)
        return self.action_head.compute_losses(
            latent_features, proprio, continuous_actions, gripper_actions,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        latent_video: torch.Tensor,
        proprio: torch.Tensor,
        num_steps: int | None = None,
        current_video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_features = self.latent_adapter(latent_video)
        if current_video_features is not None:
            latent_features = torch.cat([current_video_features, latent_features], dim=1)
        return self.action_head.sample_actions(latent_features, proprio, num_steps=num_steps)
