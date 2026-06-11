import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Paths
    data_root: Path = Path("./data")
    model_dir: str = "MCG-NJU/videomae-base-finetuned-ssv2"
    output_dir: Path = Path("./outputs")

    # Suites to use (scans data_root/<suite>/*.hdf5)
    suites: tuple = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")

    dataset_type: str = "mimicgen"
    # Location of the MimicGen demonstration data (override with the MIMICGEN_DATA env var).
    mimicgen_data_root: Path = Path(os.environ.get("MIMICGEN_DATA", str(Path.home() / "mimicgen_training_data")))

    # Task filter: only use tasks whose name contains one of these keywords
    # Set to None to use all tasks
    task_filter_keywords: tuple = ("pick_up", "put_the", "put_both", "place_it")
    task_filter_exclude: tuple = ("open", "close", "turn", "push", "stack")
    task_filter_indices: Optional[tuple[int, ...]] = None

    # Data
    num_frames: int = 8
    frame_stride: int = 2
    window_stride: int = 8
    train_demos_per_task: int = 40
    val_demos_per_task: int = 10
    img_key: str = "agentview_rgb"
    proprio_keys: tuple = ("ee_states", "gripper_states")
    proprio_dim: int = 8  # ee_states(6) + gripper_states(2)
    proprio_history_size: int = 1
    history_num_frames: int = 1
    future_frame_offset: int = 1
    video_source: str = "future"

    # Actions
    action_dim: int = 7  # 3 pos + 3 ori + 1 gripper
    action_chunk_size: int = 8

    # VideoMAE
    videomae_hidden_dim: int = 768
    videomae_layer: int = 6
    videomae_seq_len: int = 784  # (8/2) * (224/16)^2

    # Decoder
    decoder_hidden_dim: int = 768
    decoder_num_layers: int = 2
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1

    # Training
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    seed: int = 42
    device: str = "cuda:0"
    grad_clip: float = 1.0

    # Loss weights
    continuous_loss_weight: float = 1.0
    gripper_loss_weight: float = 1.0
