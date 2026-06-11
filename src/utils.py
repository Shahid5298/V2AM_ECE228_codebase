from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_action_stats(
    dataset,
    continuous_dims: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of continuous action dims from the dataset.

    Returns (mean, std) each of shape (6,).
    """
    all_actions = []
    for i in tqdm(range(len(dataset)), desc="Computing action stats", unit="win"):
        sample = dataset[i]
        all_actions.append(sample["actions"][:, :continuous_dims])

    stacked = np.concatenate(all_actions, axis=0)  # (N*16, 6)
    mean = torch.from_numpy(stacked.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(stacked.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=1e-6)  # prevent division by zero
    return mean, std


def compute_action_stats_fast(
    demo_paths: list[str],
    continuous_dims: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast Parquet-only action stats computation (no image decoding).

    Reads only the 'action' column from each episode file.
    Returns (mean, std) each of shape (continuous_dims,).
    """
    import pyarrow.parquet as pq
    all_actions = []
    for pq_path in tqdm(demo_paths, desc="Computing action stats", unit="ep"):
        table = pq.read_table(pq_path, columns=["action"])
        actions = np.stack(table["action"].to_numpy(zero_copy_only=False))  # (T, 7)
        all_actions.append(actions[:, :continuous_dims])

    stacked = np.concatenate(all_actions, axis=0)  # (N_total_steps, 6)
    mean = torch.from_numpy(stacked.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(stacked.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=1e-6)
    return mean, std


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "action_mean": action_mean,
        "action_std": action_std,
    }, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: str | None = None,
) -> dict:
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
