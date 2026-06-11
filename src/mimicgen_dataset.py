from __future__ import annotations

import functools
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor
from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm

from .config import Config

# Per-worker Parquet file handle cache (not shared across processes)
_pq_cache: dict[str, pq.ParquetFile] = {}


def _discover_parquet_episodes(config: Config) -> list[Path]:
    """Discover Parquet files in the MimicGen dataset chunks directory."""
    parquet_files = []
    mimicgen_dir = Path(config.mimicgen_data_root)
    if not mimicgen_dir.exists():
        print(f"Warning: MimicGen root {mimicgen_dir} does not exist.")
        return parquet_files
        
    for chunk_dir in sorted((mimicgen_dir / "data").glob("chunk-*")):
        for pq_path in sorted(chunk_dir.glob("*.parquet")):
            parquet_files.append(pq_path)
    return parquet_files

def _load_tasks_jsonl(config: Config) -> dict[int, str]:
    """Load task index to task name mapping."""
    tasks = {}
    tasks_path = Path(config.mimicgen_data_root) / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                data = json.loads(line)
                tasks[data["task_index"]] = data["task"]
    return tasks


def _get_pq(path: str) -> pq.ParquetFile:
    if path not in _pq_cache:
        _pq_cache[path] = pq.ParquetFile(path)
    return _pq_cache[path]


def _worker_init_fn(worker_id: int):
    """Clear Parquet cache so each worker starts fresh."""
    _pq_cache.clear()


def _temporal_resample_sequence(frames: np.ndarray, target_len: int) -> list[np.ndarray]:
    """Nearest-neighbor resampling along the time axis to match VideoMAE input length."""
    if len(frames) == target_len:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[i] for i in idx]


def _aggregate_actions(actions: np.ndarray, stride: int) -> np.ndarray:
    """Aggregate relative actions over non-overlapping windows of size `stride`.

    For OSC_POSE actions (7-dim):
      - dims 0:6 (delta pose): summed over the window to preserve total displacement
      - dim 6 (gripper): take the last value in each window (binary command)

    Args:
        actions: (T, D) array of raw actions
        stride: window size for aggregation

    Returns:
        (T // stride, D) array of aggregated actions
    """
    if stride <= 1:
        return actions

    T, D = actions.shape
    n_windows = T // stride
    # Trim to exact multiple of stride
    trimmed = actions[:n_windows * stride]  # (n_windows * stride, D)
    reshaped = trimmed.reshape(n_windows, stride, D)  # (n_windows, stride, D)

    # Sum delta-pose dims, take last gripper value
    agg = np.empty((n_windows, D), dtype=actions.dtype)
    agg[:, :6] = reshaped[:, :, :6].sum(axis=1)   # sum relative deltas
    agg[:, 6] = reshaped[:, -1, 6]                  # last gripper command
    return agg


class MimicGenWindowDataset(Dataset):
    """Sliding-window dataset over MimicGen Parquet episodes."""

    def __init__(self, config: Config, demo_paths: list[str] | None = None):
        """
        Args:
            config: hyperparameters
            demo_paths: List of specific parquet file paths to use. If None, use all.
        """
        self.config = config
        self.num_frames = config.num_frames
        self.frame_stride = getattr(config, "frame_stride", 1)
        self.action_stride = getattr(config, "action_stride", 1)
        self.chunk_size = getattr(config, "chunk_size", getattr(config, "action_chunk_size", self.num_frames))
        self.proprio_history_size = getattr(config, "proprio_history_size", self.num_frames)
        self.history_num_frames = getattr(config, "history_num_frames", 1)
        self.future_frame_offset = getattr(config, "future_frame_offset", 0)
        self.stride = config.window_stride

        # The sample must contain enough raw steps for both the action chunk
        # and the future video conditioning window.
        self.max_action_idx = self.action_stride * (self.chunk_size - 1)
        self.max_video_idx = self.future_frame_offset + self.frame_stride * (self.num_frames - 1)
        self.raw_window_len = max(self.max_action_idx, self.max_video_idx) + 1

        # Build flat index: [(parquet_path, start_frame, task_id)]
        self.index: list[tuple[str, int, int]] = []
        
        task_map = _load_tasks_jsonl(config)
        
        # Deduplicate task names for the global list to preserve compatibility
        self.task_names = list(sorted(set(task_map.values()))) if task_map else ["Unknown Task"]
        
        # If no explicit subset is provided, discover all
        parquet_files = _discover_parquet_episodes(config)
        use_files = [str(p) for p in parquet_files] if demo_paths is None else demo_paths
        self.demo_paths = use_files  # expose for fast action stat computation

        # Optional task index filter
        task_filter = getattr(config, "task_filter_indices", None)

        for pq_str in tqdm(use_files, desc="Indexing episodes", unit="ep"):
            md = pq.read_metadata(pq_str)
            T = md.num_rows
            
            # Read just the task index for the episode (it's constant per file)
            table = pq.read_table(pq_str, columns=['task_index'])
            task_id = table['task_index'][0].as_py()

            # Skip if task not in filter set
            if task_filter is not None and task_id not in task_filter:
                continue
            
            for start in range(0, T - self.raw_window_len + 1, self.stride):
                self.index.append((pq_str, start, task_id))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        pq_path, start, task_id = self.index[idx]
        
        pf = _get_pq(pq_path)
        full_table = pf.read()  # Read entire episode (~200 rows)
        total_rows = len(full_table)

        # ── Shared slice: [start, start + raw_window_len) ─────────────────
        window_table = full_table.slice(offset=start, length=self.raw_window_len)

        full_main_dicts = full_table['observation.image'].to_pylist()
        full_wrist_dicts = full_table['observation.image_wrist'].to_pylist()
        history_indices = [
            max(0, start - (self.history_num_frames - 1 - j) * self.frame_stride)
            for j in range(self.history_num_frames)
        ]
        history_main = []
        history_wrist = []
        for hist_idx in history_indices:
            hist_main = full_main_dicts[hist_idx]
            hist_wrist = full_wrist_dicts[hist_idx]
            history_main.append(np.array(Image.open(io.BytesIO(hist_main['bytes'])).convert('RGB')))
            history_wrist.append(np.array(Image.open(io.BytesIO(hist_wrist['bytes'])).convert('RGB')))
        history_frames_main = np.stack(history_main)
        history_frames_wrist = np.stack(history_wrist)

        # Video frames: future-conditioned window starting at start + offset
        images_main = []
        images_wrist = []
        main_dicts = window_table['observation.image'].to_pylist()
        wrist_dicts = window_table['observation.image_wrist'].to_pylist()

        frame_indices = [
            self.future_frame_offset + j * self.frame_stride
            for j in range(self.num_frames)
        ]
        for frame_idx in frame_indices:
            main_d = main_dicts[frame_idx]
            wrist_d = wrist_dicts[frame_idx]
            img_m = Image.open(io.BytesIO(main_d['bytes'])).convert('RGB')
            img_w = Image.open(io.BytesIO(wrist_d['bytes'])).convert('RGB')
            images_main.append(np.array(img_m))
            images_wrist.append(np.array(img_w))

        frames_main_np = np.stack(images_main)    # (num_frames, H, W, 3)
        frames_wrist_np = np.stack(images_wrist)  # (num_frames, H, W, 3)

        # Actions: predict the chunk starting exactly at `start`
        raw_actions = np.stack(
            window_table['action'].to_numpy(zero_copy_only=False)
        ).astype(np.float32)  # (raw_window_len, 7)
        action_indices = [j * self.action_stride for j in range(self.chunk_size)]
        actions = raw_actions[action_indices]  # (chunk_size, 7)

        # ── Proprio history: before and up to first video frame ──────────
        # We want proprio_history_size steps ending at index `start` (inclusive)
        proprio_start = max(0, start - self.proprio_history_size + 1)
        proprio_end = start + 1  # exclusive, so we include `start`
        proprio_table = full_table.slice(offset=proprio_start, length=proprio_end - proprio_start)
        
        pos = np.stack(proprio_table['observation.robot0_eef_pos'].to_numpy(zero_copy_only=False))
        quat = np.stack(proprio_table['observation.robot0_eef_quat'].to_numpy(zero_copy_only=False))
        gripper = np.stack(proprio_table['observation.robot0_gripper_qpos'].to_numpy(zero_copy_only=False))
        euler = R.from_quat(quat).as_euler('xyz', degrees=False)
        proprio = np.concatenate([pos, euler, gripper], axis=-1).astype(np.float32)
        
        # Left-pad with earliest value if near episode start
        actual_len = proprio.shape[0]
        if actual_len < self.proprio_history_size:
            pad_len = self.proprio_history_size - actual_len
            pad = np.repeat(proprio[:1], pad_len, axis=0)
            proprio = np.concatenate([pad, proprio], axis=0)

        return {
            "history_frames": history_frames_main,
            "history_frames_wrist": history_frames_wrist,
            "frames": frames_main_np,
            "frames_wrist": frames_wrist_np,
            "proprio": proprio,          # (proprio_history_size, 8)
            "actions": actions,           # (chunk_size, 7)
            "task_id": task_id,
        }


def collate_fn(batch: list[dict], processor: VideoMAEImageProcessor) -> dict:
    """Collate batch and run VideoMAE preprocessing on both frame streams."""
    videos_main = [list(sample["frames"]) for sample in batch]
    videos_wrist = [list(sample["frames_wrist"]) for sample in batch]
    num_frames = len(batch[0]["frames"])
    history_videos_main = [_temporal_resample_sequence(sample["history_frames"], num_frames) for sample in batch]
    history_videos_wrist = [_temporal_resample_sequence(sample["history_frames_wrist"], num_frames) for sample in batch]
    
    pixel_values_main = processor(videos_main, return_tensors="pt")["pixel_values"]
    pixel_values_wrist = processor(videos_wrist, return_tensors="pt")["pixel_values"]
    pixel_values_history_main = processor(history_videos_main, return_tensors="pt")["pixel_values"]
    pixel_values_history_wrist = processor(history_videos_wrist, return_tensors="pt")["pixel_values"]

    proprio = torch.stack([torch.from_numpy(s["proprio"]) for s in batch])
    actions = torch.stack([torch.from_numpy(s["actions"]) for s in batch])
    task_ids = torch.tensor([s["task_id"] for s in batch], dtype=torch.long)

    return {
        "pixel_values_history": pixel_values_history_main,
        "pixel_values_wrist_history": pixel_values_history_wrist,
        "pixel_values": pixel_values_main,
        "pixel_values_wrist": pixel_values_wrist,
        "proprio": proprio,
        "actions": actions,
        "task_ids": task_ids,
    }


def make_mimicgen_dataloaders(
    config: Config,
    processor: VideoMAEImageProcessor,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders with stratified demo split."""
    train_files, val_files = split_mimicgen_demo_paths(config)

    train_ds = MimicGenWindowDataset(config, train_files)
    val_ds = MimicGenWindowDataset(config, val_files)

    collate = functools.partial(collate_fn, processor=processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def split_mimicgen_demo_paths(config: Config) -> tuple[list[str], list[str]]:
    """Return stratified train / val episode file lists."""
    parquet_files = _discover_parquet_episodes(config)

    train_files = []
    val_files = []

    # Map files by task to perform a stratified split
    task_to_files = {}
    for pq_path in tqdm(parquet_files, desc="Splitting episodes", unit="ep"):
        pq_str = str(pq_path)
        table = pq.read_table(pq_str, columns=['task_index'])
        task_id = table['task_index'][0].as_py()
        if task_id not in task_to_files:
            task_to_files[task_id] = []
        task_to_files[task_id].append(pq_str)
        
    for task_id, files in task_to_files.items():
        total_demos = len(files)
        num_train = min(config.train_demos_per_task, int(0.8 * total_demos))
        num_val = min(config.val_demos_per_task, total_demos - num_train)
        
        train_files.extend(files[:num_train])
        val_files.extend(files[num_train:num_train + num_val])
    return train_files, val_files
