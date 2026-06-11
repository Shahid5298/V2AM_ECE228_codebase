"""
Flow Matching Action Head for VideoMAE features.

Adapted from SmolVLA (HuggingFace LeRobot) — extracts the core flow matching
components (sinusoidal time embedding, action-time MLP fusion, transformer
denoiser, Euler ODE sampling) and plugs them into VideoMAE features instead
of a VLM backbone.

Architecture:
    VideoMAE features (B, S, 768)
            │
            ▼
    prefix_proj  →  (B, S, hidden_dim)   ← conditioning context
            │
    ┌───────┴───────────────────────────┐
    │                                   │
    │   noisy_actions (B, C, action_dim)│
    │        + sinusoidal time embed    │
    │        → action_time_mlp          │
    │        → (B, C, hidden_dim)       │
    │             │                     │
    │   Transformer Decoder Blocks      │
    │   (cross-attn to prefix,          │
    │    self-attn among actions,       │
    │    MLP)                           │
    │             │                     │
    │   action_out_proj → (B,C,act_dim) │
    └───────────────────────────────────┘
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import FlowMatchingConfig


# ─── Sinusoidal positional embedding (from SmolVLA) ──────────────────────────

def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device: str = "cpu",
) -> Tensor:
    """Sine-cosine positional embedding for scalar timesteps.

    Args:
        time: (batch_size,) scalar timesteps
        dimension: embedding dimension (must be even)
        min_period / max_period: frequency range
    Returns:
        (batch_size, dimension)
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    dtype = torch.float64 if device == "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb.float()


# ─── Transformer decoder block ──────────────────────────────────────────────

class DenoiserBlock(nn.Module):
    """Pre-norm transformer decoder block: cross-attn → self-attn → MLP."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )

        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # Cross-attention: action tokens attend to VideoMAE context
        x_norm = self.norm_cross(x)
        ctx_norm = self.norm_kv(context)
        x = x + self.cross_attn(x_norm, ctx_norm, ctx_norm)[0]

        # Self-attention among action tokens
        x_norm = self.norm_self(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]

        # Feed-forward
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ─── Flow Matching Action Head ──────────────────────────────────────────────

class FlowMatchingActionHead(nn.Module):
    """
    Conditional Flow Matching action head conditioned on VideoMAE features.

    Training:
        - Sample t ~ Beta(1.5, 1.0), noise ~ N(0,1)
        - x_t = t * noise + (1-t) * actions  (flow interpolation)
        - u_t = noise - actions               (target velocity)
        - Predict v_t via denoiser, loss = MSE(u_t, v_t)

    Inference:
        - Start from x_1 = noise
        - Euler integrate: x_{t+dt} = x_t + dt * v_t  for num_steps
        - Return x_0 ≈ predicted actions
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        cont_dim = config.continuous_action_dim

        # ── Projections ──────────────────────────────────────────────────
        # Project conditioning features → denoiser hidden dim
        self.prefix_proj = nn.Linear(config.visual_feature_dim, d)

        # Proprio state → hidden dim (added to prefix context)
        self.proprio_proj = nn.Sequential(
            nn.Linear(config.proprio_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Project noisy continuous actions → hidden dim
        self.action_in_proj = nn.Linear(cont_dim, d)

        # Time-action fusion MLP (from SmolVLA)
        self.action_time_mlp_in = nn.Linear(d * 2, d)
        self.action_time_mlp_out = nn.Linear(d, d)

        # Output heads
        self.continuous_out_proj = nn.Linear(d, cont_dim)
        self.gripper_head = nn.Linear(d, 1)

        # ── Transformer denoiser ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DenoiserBlock(
                hidden_dim=d,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

    # ── Flow matching utilities (from SmolVLA) ───────────────────────────

    def sample_noise(self, shape, device):
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device):
        beta_dist = torch.distributions.Beta(
            self.config.beta_concentration1, self.config.beta_concentration0,
        )
        t = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        t = t * 0.999 + 0.001  # clamp to [0.001, 1.0]
        return t

    # ── Embed context (VideoMAE + proprio) ───────────────────────────────

    def embed_context(self, video_features: Tensor, proprio: Tensor) -> Tensor:
        """Build the conditioning context sequence.

        Args:
            video_features: (B, S, 768) from frozen VideoMAE
            proprio: (B, H, proprio_dim) where H = proprio_history_size
        Returns:
            context: (B, S + H, hidden_dim)
        """
        vid_emb = self.prefix_proj(video_features)           # (B, S, d)
        pro_emb = self.proprio_proj(proprio)                  # (B, H, d)
        return torch.cat([vid_emb, pro_emb], dim=1)           # (B, S+H, d)

    # ── Embed suffix (noisy actions + time) ──────────────────────────────

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> Tensor:
        """Fuse noisy actions with sinusoidal time embedding.

        Args:
            noisy_actions: (B, C, continuous_action_dim)
            timestep: (B,)
        Returns:
            (B, C, hidden_dim)
        """
        action_emb = self.action_in_proj(noisy_actions)       # (B, C, d)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.hidden_dim,
            self.config.min_period, self.config.max_period,
            device=action_emb.device,
        )                                                     # (B, d)
        time_emb = time_emb[:, None, :].expand_as(action_emb) # (B, C, d)

        fused = torch.cat([action_emb, time_emb], dim=2)      # (B, C, 2d)
        fused = self.action_time_mlp_in(fused)
        fused = F.silu(fused)
        fused = self.action_time_mlp_out(fused)                # (B, C, d)
        return fused

    # ── Denoiser forward ─────────────────────────────────────────────────

    def denoise(self, context: Tensor, noisy_actions: Tensor, timestep: Tensor) -> Tensor:
        """Run the denoising transformer trunk over noisy continuous actions."""
        x = self.embed_suffix(noisy_actions, timestep)  # (B, C, d)
        for block in self.blocks:
            x = block(x, context)
        return self.final_norm(x)

    def predict_continuous_velocity(self, hidden: Tensor) -> Tensor:
        return self.continuous_out_proj(hidden.float())

    def predict_gripper_logits(self, hidden: Tensor) -> Tensor:
        return self.gripper_head(hidden.float())

    # ── Training forward ─────────────────────────────────────────────────

    def compute_losses(
        self,
        video_features: Tensor,
        proprio: Tensor,
        continuous_actions: Tensor,
        gripper_actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute joint flow-matching and gripper BCE losses.

        Args:
            video_features: (B, S, 768)
            proprio: (B, C, proprio_dim)
            continuous_actions: (B, C, continuous_action_dim) normalized continuous targets
            gripper_actions: (B, C, 1) binary gripper targets in {0, 1}
            noise: optional pre-sampled noise
            time: optional pre-sampled timesteps
        Returns:
            dict containing total, flow, and gripper losses
        """
        B = continuous_actions.shape[0]
        device = continuous_actions.device

        if noise is None:
            noise = self.sample_noise(continuous_actions.shape, device)
        if time is None:
            time = self.sample_time(B, device)

        # Flow interpolation
        t = time[:, None, None]                    # (B, 1, 1)
        x_t = t * noise + (1 - t) * continuous_actions
        u_t = noise - continuous_actions

        context = self.embed_context(video_features, proprio)
        hidden = self.denoise(context, x_t, time)
        v_t = self.predict_continuous_velocity(hidden)
        gripper_logit = self.predict_gripper_logits(hidden)

        flow_loss = F.mse_loss(u_t, v_t)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logit, gripper_actions)
        loss = (
            self.config.flow_loss_weight * flow_loss
            + self.config.gripper_loss_weight * gripper_loss
        )

        smooth_loss = torch.tensor(0.0, device=device)
        if self.config.smooth_loss_weight > 0:
            # Reconstruct the predicted clean action from the noisy observation.
            # From the flow interpolation: x_t = (1-t)*x_0 + t*noise
            # and v_t ≈ noise - x_0, so: x_0_pred = x_t - t * v_t ≈ x_0
            x0_pred = x_t - t * v_t                               # (B, C, cont_dim)
            # Second finite difference across the chunk axis = discrete jerk
            jerk = x0_pred[:, 2:] - 2 * x0_pred[:, 1:-1] + x0_pred[:, :-2]
            smooth_loss = (jerk ** 2).mean()
            loss = loss + self.config.smooth_loss_weight * smooth_loss

        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "gripper_loss": gripper_loss,
            "smooth_loss": smooth_loss,
        }

    def forward(
        self,
        video_features: Tensor,
        proprio: Tensor,
        continuous_actions: Tensor,
        gripper_actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        return self.compute_losses(
            video_features, proprio, continuous_actions, gripper_actions, noise=noise, time=time,
        )["loss"]

    # ── Inference (Euler ODE) ────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        video_features: Tensor,
        proprio: Tensor,
        num_steps: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate normalized continuous actions and gripper logits.

        Args:
            video_features: (B, S, 768)
            proprio: (B, C, proprio_dim)
            num_steps: override for number of Euler steps
        Returns:
            continuous_actions: (B, C, continuous_action_dim)
            gripper_logits: (B, C, 1)
        """
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        B = video_features.shape[0]
        device = video_features.device

        # Build context once (cached across steps)
        context = self.embed_context(video_features, proprio)

        # Start from pure noise at t=1
        x_t = self.sample_noise(
            (B, self.config.chunk_size, self.config.continuous_action_dim), device,
        )

        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.full((B,), t, dtype=torch.float32, device=device)
            hidden = self.denoise(context, x_t, t_tensor)
            v_t = self.predict_continuous_velocity(hidden)
            x_t = x_t + dt * v_t

        t_zero = torch.zeros((B,), dtype=torch.float32, device=device)
        hidden = self.denoise(context, x_t, t_zero)
        gripper_logit = self.predict_gripper_logits(hidden)
        return x_t, gripper_logit

    # ── Inference (Flow Ensembling) ──────────────────────────────────────

    @torch.no_grad()
    def sample_actions_ensemble(
        self,
        video_features: Tensor,
        proprio: Tensor,
        k: int = 5,
        num_steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run Euler ODE k times from different noise seeds and average.

        Averaging over multiple noise samples reduces variance and gives a free
        uncertainty estimate: high per-step variance = model is uncertain at
        that action timestep (e.g. at grasp/release transitions).

        Args:
            video_features: (B, S, 768)
            proprio: (B, H, proprio_dim)
            k: number of noise samples to average
            num_steps: Euler steps per sample (default from config)
        Returns:
            mean_continuous: (B, C, continuous_action_dim) averaged actions
            mean_gripper_logit: (B, C, 1) mean gripper logit across samples
            per_step_variance: (C,) mean action variance per chunk timestep
        """
        all_continuous = []
        all_gripper = []

        for _ in range(k):
            cont, grip = self.sample_actions(video_features, proprio, num_steps=num_steps)
            all_continuous.append(cont)
            all_gripper.append(grip)

        stacked_cont = torch.stack(all_continuous, dim=0)   # (k, B, C, cont_dim)
        mean_continuous = stacked_cont.mean(dim=0)           # (B, C, cont_dim)

        # Variance per chunk timestep: average over batch dimension and action dims
        per_step_variance = stacked_cont.var(dim=0).mean(dim=(0, 2))  # (C,)

        stacked_grip = torch.stack(all_gripper, dim=0)       # (k, B, C, 1)
        mean_gripper_logit = stacked_grip.mean(dim=0)        # (B, C, 1)

        return mean_continuous, mean_gripper_logit, per_step_variance

    # ── Inference (Neural ODE, dopri5 adaptive solver) ───────────────────

    @torch.no_grad()
    def sample_actions_dopri5(
        self,
        video_features: Tensor,
        proprio: Tensor,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> tuple[Tensor, Tensor, int]:
        """Integrate the learned velocity field with an adaptive dopri5 solver.

        Uses torchdiffeq. The ODE is reparameterized so s = 1-t goes from 0→1
        while the physical flow time t goes 1→0 (noise → actions). This lets
        torchdiffeq receive an increasing time span.

        The adaptive solver takes large steps where the field is smooth and
        small steps where it curves — same or better accuracy than Euler-50
        at lower average NFE.

        Args:
            video_features: (B, S, 768)
            proprio: (B, H, proprio_dim)
            rtol: relative ODE tolerance
            atol: absolute ODE tolerance
        Returns:
            continuous_actions: (B, C, continuous_action_dim)
            gripper_logits: (B, C, 1)
            nfe: number of function evaluations used
        """
        from torchdiffeq import odeint

        B = video_features.shape[0]
        device = video_features.device

        context = self.embed_context(video_features, proprio)

        x_noise = self.sample_noise(
            (B, self.config.chunk_size, self.config.continuous_action_dim), device,
        )

        nfe_count = [0]

        def ode_func(s: Tensor, x: Tensor) -> Tensor:
            # s goes 0→1, physical time t = 1-s goes 1→0
            nfe_count[0] += 1
            t = 1.0 - s.item()
            t_tensor = torch.full((B,), t, dtype=torch.float32, device=device)
            hidden = self.denoise(context, x, t_tensor)
            v = self.predict_continuous_velocity(hidden)
            return -v  # dx/ds = -v_t  (sign from chain rule: ds = -dt)

        s_span = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
        solution = odeint(ode_func, x_noise, s_span, method="dopri5", rtol=rtol, atol=atol)
        x_clean = solution[-1]  # (B, C, cont_dim) at s=1 i.e. t=0

        t_zero = torch.zeros((B,), dtype=torch.float32, device=device)
        hidden = self.denoise(context, x_clean, t_zero)
        gripper_logit = self.predict_gripper_logits(hidden)

        return x_clean, gripper_logit, nfe_count[0]
