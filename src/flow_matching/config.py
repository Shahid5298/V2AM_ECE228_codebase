import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FlowMatchingConfig:
    """Configuration for VideoMAE + Flow Matching Action Head.

    The default setup is inverse-dynamics style:
      - actions start at timestep t
      - proprio is taken at timestep t
      - video conditioning frames start at t + future_frame_offset
    """

    # ── Paths ────────────────────────────────────────────────────────────
    data_root: Path = Path("./data")
    model_dir: str = "MCG-NJU/videomae-base-finetuned-ssv2"
    output_dir: Path = Path("./outputs/flow_matching_future_video")

    # Suites
    suites: tuple = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")

    dataset_type: str = "mimicgen"
    # Location of the MimicGen demonstration data (override with the MIMICGEN_DATA env var).
    mimicgen_data_root: Path = Path(os.environ.get("MIMICGEN_DATA", str(Path.home() / "mimicgen_training_data")))

    # Task filter
    task_filter_keywords: tuple = ("pick_up", "put_the", "put_both", "place_it")
    task_filter_exclude: tuple = ("open", "close", "turn", "push", "stack")
    task_filter_indices: tuple | None = None  # e.g. (12,) to train on stack_three_d0 only

    # ── Data ─────────────────────────────────────────────────────────────
    num_frames: int = 8        # number of future video frames fed to VideoMAE
    frame_stride: int = 2      # subsample every 2nd raw step for video
    window_stride: int = 8
    train_demos_per_task: int = 40
    val_demos_per_task: int = 10
    img_key: str = "agentview_rgb"
    proprio_keys: tuple = ("ee_states", "gripper_states")
    proprio_dim: int = 8  # ee_states(6) + gripper_states(2)

    # ── Actions ──────────────────────────────────────────────────────────
    action_dim: int = 7   # 3 pos + 3 ori + 1 gripper
    continuous_action_dim: int = 6  # 3 pos + 3 ori
    chunk_size: int = 16  # number of future action steps to predict
    action_stride: int = 1  # stride for action sampling (1 = every raw step)
    future_frame_offset: int = 1  # first video frame is one step after the action start
    include_history_frames: bool = True  # prepend past-to-current visual context as a separate stream
    history_num_frames: int = 4  # number of past->current frames to include in the history stream
    condition_on_future_video: bool = True  # include the future-conditioned video stream

    # ── Proprio ──────────────────────────────────────────────────────────
    proprio_history_size: int = 4  # short proprio history for motion / velocity cues

    # ── VideoMAE encoder (frozen) ────────────────────────────────────────
    visual_feature_dim: int = 768  # input feature dim for the action head prefix projection
    videomae_hidden_dim: int = 768
    videomae_layer: int = 6
    videomae_seq_len: int = 784  # (8/2) * (224/16)^2 = 4 * 196

    # ── Hummingbird latent adapter ───────────────────────────────────────
    hummingbird_latent_channels: int = 4
    hummingbird_adapter_dim: int = 128
    hummingbird_pool_spatial: int = 8

    # ── Flow Matching Action Head ────────────────────────────────────────
    hidden_dim: int = 384       # denoiser transformer hidden size
    num_layers: int = 4         # number of transformer decoder layers
    num_heads: int = 6          # attention heads (384/6 = 64 dim/head)
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Sinusoidal time embedding parameters (from SmolVLA)
    min_period: float = 4e-3
    max_period: float = 4.0

    # ODE solver
    num_denoising_steps: int = 50  # Euler steps at inference (used for val checkpointing)

    # Flow matching time sampling
    beta_concentration1: float = 1.5
    beta_concentration0: float = 1.0

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    seed: int = 42
    device: str = "cuda:0"
    grad_clip: float = 1.0
    flow_loss_weight: float = 1.0
    gripper_loss_weight: float = 1.0
    smooth_loss_weight: float = 0.0  # λ for jerk penalty on reconstructed actions (0 = off)
