from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import VideoMAEModel


class VideoMAEFeatureExtractor(nn.Module):
    """Frozen VideoMAE wrapper that extracts features from a specific layer via hook."""

    def __init__(self, model_dir: str | Path, layer_idx: int = 6, num_frames_expected: int = 8, device: str = "cuda:0"):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(str(model_dir), torch_dtype=torch.float16)
        self.model.eval()
        self.layer_idx = layer_idx

        # ── Interpolate positional embeddings if num_frames differs ──────────
        model_num_frames = self.model.config.num_frames
        if num_frames_expected != model_num_frames:
            print(f"Adapting VideoMAE positional embeddings from {model_num_frames} to {num_frames_expected} frames.")
            self._interpolate_pos_encoding(num_frames_expected, model_num_frames)

        for param in self.model.parameters():
            param.requires_grad = False

        # Hook on the target layer (0-indexed: layer_idx=6 → encoder.layer[5])
        self._features: torch.Tensor | None = None
        self.model.encoder.layer[layer_idx - 1].register_forward_hook(self._hook_fn)

    def _interpolate_pos_encoding(self, target_frames: int, source_frames: int):
        """Interpolate the 1D positional embeddings to match the new number of frames."""
        # VideoMAE patches video into a 3D grid: (T, H, W)
        # Sequence length = (num_frames // tubelet_size) * (image_size // patch_size)^2
        tubelet_size = self.model.config.tubelet_size
        patch_size = self.model.config.patch_size
        image_size = self.model.config.image_size
        
        T_orig = source_frames // tubelet_size
        T_new = target_frames // tubelet_size
        N_spatial = (image_size // patch_size) ** 2
        
        orig_pos_embed = self.model.embeddings.position_embeddings  # (1, T_orig * N_spatial, D)
        D = orig_pos_embed.shape[-1]
        
        # Reshape to (1, D, T_orig, N_spatial) for interpolation
        orig_pos_embed = orig_pos_embed.reshape(1, T_orig, N_spatial, D)
        orig_pos_embed = orig_pos_embed.permute(0, 3, 1, 2)  # (1, D, T_orig, N_spatial)
        
        # Interpolate along the temporal dimension (T_orig -> T_new)
        # mode "bilinear" expects 4D input (N, C, H, W) where we treat (T, N_spatial) as spatial dims
        new_pos_embed = torch.nn.functional.interpolate(
            orig_pos_embed,
            size=(T_new, N_spatial),
            mode="bilinear",
            align_corners=False,
        )  # (1, D, T_new, N_spatial)
        
        # Reshape back to (1, T_new * N_spatial, D)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Replace the embedding in the model
        self.model.embeddings.position_embeddings = nn.Parameter(new_pos_embed, requires_grad=False)
        self.model.embeddings.num_patches = T_new * N_spatial

    def _hook_fn(self, module, input, output):
        # VideoMAE encoder layers return a tuple (hidden_states, ...)
        self._features = output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 16, 3, 224, 224) float32 from processor
        Returns:
            features: (B, 1568, 768) float32
        """
        pixel_values = pixel_values.to(dtype=torch.float16)
        self.model(pixel_values)
        features = self._features.float()
        self._features = None
        return features
