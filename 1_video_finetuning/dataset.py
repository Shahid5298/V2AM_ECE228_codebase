"""
LIBERO Dataset Loader for I2V LoRA Fine-tuning

Loads robot manipulation videos from the LIBERO dataset and prepares them
for Image-to-Video training. The first frame is always the conditioning image.
"""

import os
import sys
import json
import glob
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


def load_task_prompts(meta_dir: str) -> Dict[int, str]:
    """Load task descriptions from LIBERO meta/tasks.jsonl."""
    tasks_file = os.path.join(meta_dir, "tasks.jsonl")
    prompts = {}
    if os.path.exists(tasks_file):
        with open(tasks_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                prompts[entry['task_index']] = entry['task']
    return prompts


DEFAULT_PROMPT = "robot arm manipulating objects on a table"


class LiberoI2VDataset(Dataset):
    """
    Dataset for loading LIBERO robot manipulation videos for I2V training.

    Each sample returns:
      - first_frame: (C, H, W) — the conditioning image (always frame 0)
      - video: (C, T, H, W) — the full video clip starting from the conditioning frame
      - caption: text description of the task
    """

    def __init__(
        self,
        data_dir: str = None,
        meta_dir: str = None,
        video_length: int = 4,
        frame_stride: int = 2,
        resolution: Tuple[int, int] = (256, 256),
        max_episodes: Optional[int] = None,
        image_key: str = "observation.image",
        random_start: bool = True,
        task_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            data_dir: Path to LIBERO parquet data directory
            meta_dir: Path to LIBERO meta directory (for task descriptions)
            video_length: Number of frames per video clip (including first frame)
            frame_stride: Sample every N frames
            resolution: Output resolution (H, W)
            max_episodes: Maximum episodes to load
            image_key: Column name for image data in parquet
            random_start: Randomly sample start frame within episode
            task_ids: Optional list of task_index values to filter by
        """
        super().__init__()

        base_path = os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--physical-intelligence--libero/"
            "snapshots/a4336d589d589045d1c56423ffdf3b88a0e19b1f"
        )
        if data_dir is None:
            data_dir = os.path.join(base_path, "data")
        if meta_dir is None:
            meta_dir = os.path.join(base_path, "meta")

        self.data_dir = Path(data_dir)
        self.video_length = video_length
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.image_key = image_key
        self.random_start = random_start

        # Load task prompts from metadata
        self.task_prompts = load_task_prompts(meta_dir)
        if self.task_prompts:
            print(f"Loaded {len(self.task_prompts)} task descriptions from metadata")
        else:
            print("Warning: No task descriptions found, using default prompt")

        # Find all episode files
        self.episode_files = self._find_episodes(max_episodes)

        # Filter by task_ids if specified
        if task_ids is not None:
            self.episode_files = self._filter_by_task_ids(task_ids)
            print(f"Filtered to {len(self.episode_files)} episodes for task_ids {task_ids}")
        else:
            print(f"Found {len(self.episode_files)} episodes")

        # Determine actual image key by peeking at first episode
        if len(self.episode_files) > 0:
            self.image_key = self._detect_image_key(self.episode_files[0])
            print(f"Using image key: '{self.image_key}'")

        # Build transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Cache for loaded episodes
        self._episode_cache = {}

    def _detect_image_key(self, episode_path: Path) -> str:
        """Auto-detect the image column name from a parquet file."""
        df = pd.read_parquet(episode_path, columns=None)
        cols = df.columns.tolist()
        # Try common keys
        for key in ["observation.image", "image", "observation.images.top", "agentview_image"]:
            if key in cols:
                return key
        # Fall back to first column containing 'image'
        for col in cols:
            if 'image' in col.lower():
                return col
        raise ValueError(f"No image column found in {cols}")

    def _find_episodes(self, max_episodes: Optional[int] = None) -> List[Path]:
        """Find all parquet episode files."""
        pattern = str(self.data_dir / "chunk-*" / "*.parquet")
        files = sorted(glob.glob(pattern))

        if not files:
            pattern = str(self.data_dir / "*.parquet")
            files = sorted(glob.glob(pattern))

        if max_episodes:
            files = files[:max_episodes]

        return [Path(f) for f in files]

    def _filter_by_task_ids(self, task_ids: List[int]) -> List[Path]:
        """Filter episode files to only include those matching the given task IDs."""
        filtered = []
        task_ids_set = set(task_ids)
        for ep_path in self.episode_files:
            try:
                df = pd.read_parquet(ep_path, columns=['task_index'])
                if 'task_index' in df.columns and len(df) > 0:
                    task_idx = int(df.iloc[0]['task_index'])
                    if task_idx in task_ids_set:
                        filtered.append(ep_path)
            except Exception as e:
                print(f"Warning: Could not read task_index from {ep_path}: {e}")
        return filtered

    def _load_episode(self, episode_path: Path) -> pd.DataFrame:
        """Load a single episode from parquet file."""
        if episode_path in self._episode_cache:
            return self._episode_cache[episode_path]

        df = pd.read_parquet(episode_path)

        if len(self._episode_cache) < 50:
            self._episode_cache[episode_path] = df

        return df

    def _decode_image(self, image_data) -> Image.Image:
        """Decode image from various formats."""
        import io
        if isinstance(image_data, dict):
            if 'bytes' in image_data:
                return Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif 'path' in image_data:
                return Image.open(image_data['path']).convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, np.ndarray):
            return Image.fromarray(image_data.astype(np.uint8)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            return image_data.convert('RGB')
        else:
            raise ValueError(f"Unknown image format: {type(image_data)}")

    def _get_task_prompt(self, df: pd.DataFrame) -> str:
        """Get text prompt for this episode."""
        # Try to get task_index from the episode
        if 'task_index' in df.columns:
            task_idx = int(df.iloc[0]['task_index'])
            if task_idx in self.task_prompts:
                return self.task_prompts[task_idx]
        return DEFAULT_PROMPT

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video clip from an episode. Frame 0 is always the conditioning image.

        Returns:
            dict with:
                - video: (C, T, H, W) — full clip including first frame
                - caption: text description
                - fps: frames per second metadata
        """
        episode_path = self.episode_files[idx]
        df = self._load_episode(episode_path)

        num_frames = len(df)
        required_frames = self.video_length * self.frame_stride

        # Determine start frame
        if num_frames <= required_frames:
            start_idx = 0
        else:
            if self.random_start:
                start_idx = random.randint(0, num_frames - required_frames)
            else:
                start_idx = 0

        # Extract frames with stride
        frame_indices = list(range(
            start_idx,
            min(start_idx + required_frames, num_frames),
            self.frame_stride
        ))

        # Pad if needed by repeating last frame
        while len(frame_indices) < self.video_length:
            frame_indices.append(frame_indices[-1] if frame_indices else 0)
        frame_indices = frame_indices[:self.video_length]

        # Load and transform frames
        frames = []
        for fi in frame_indices:
            row = df.iloc[fi]
            image_data = row[self.image_key]
            image = self._decode_image(image_data)
            image_tensor = self.transform(image)
            frames.append(image_tensor)

        # Stack to (C, T, H, W) — frame 0 is always the conditioning image
        video = torch.stack(frames, dim=1)

        # Get task prompt
        caption = self._get_task_prompt(df)

        return {
            'video': video,         # (C, T, H, W)
            'caption': caption,
            'fps': torch.tensor(10),  # Default FPS for robot data
        }


def create_dataloader(
    data_dir: str = None,
    meta_dir: str = None,
    batch_size: int = 1,
    video_length: int = 4,
    frame_stride: int = 2,
    resolution: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    shuffle: bool = True,
    max_episodes: Optional[int] = None,
    task_ids: Optional[List[int]] = None,
) -> DataLoader:
    """Create a DataLoader for LIBERO I2V dataset."""

    dataset = LiberoI2VDataset(
        data_dir=data_dir,
        meta_dir=meta_dir,
        video_length=video_length,
        frame_stride=frame_stride,
        resolution=resolution,
        max_episodes=max_episodes,
        task_ids=task_ids,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    print("Testing LiberoI2VDataset...")

    dataset = LiberoI2VDataset(
        video_length=4,
        frame_stride=2,
        resolution=(256, 256),
        max_episodes=3,
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Video shape: {sample['video'].shape}")  # Should be (3, 4, 256, 256)
        print(f"Caption: {sample['caption']}")
        print(f"FPS: {sample['fps']}")
        print(f"First frame (conditioning) range: [{sample['video'][:,0].min():.2f}, {sample['video'][:,0].max():.2f}]")
    else:
        print("No episodes found. Check data_dir path.")
