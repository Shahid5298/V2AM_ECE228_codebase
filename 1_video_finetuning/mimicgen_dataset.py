"""
MimicGen Dataset Loader for I2V LoRA Fine-tuning

Loads robot manipulation videos from MimicGen HDF5 datasets or pre-rendered 
high-resolution image directories.
"""

import os
import h5py
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Mapping from MimicGen task labels (often nested in HDF5) to text prompts
TASK_PROMPTS = {
    "Coffee": "a robot arm preparing coffee by placing a mug and a pod",
    "Coffee_D0": "a robot arm placing a mug on a coffee machine",
    "Coffee_D1": "a robot arm placing a mug on a coffee machine with varied positions",
    "Coffee_Preparation": "a robot arm preparing a coffee machine",
    "NutAssembly": "a robot arm assembling a nut on a bolt",
    "PickPlace": "a robot arm picking up an object and placing it in a container",
}

DEFAULT_PROMPT = "a robot arm performing a manipulation task on a tabletop"

class MimicGenI2VDataset(Dataset):
    """
    Dataset for loading MimicGen robot manipulation data for I2V training.
    
    Supports loading from:
    1. Pre-rendered image directories (organized by episode).
    2. HDF5 files (if high-res images were stored there).
    """

    def __init__(
        self,
        data_path: str,
        video_length: int = 8,
        frame_stride: int = 2,
        resolution: Tuple[int, int] = (256, 256),
        image_type: str = "agentview_image", # or "concat"
        task_prompt: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to rendered images directory or HDF5 file
            video_length: Number of frames per clip
            frame_stride: Stride between frames
            resolution: (H, W) output resolution
            image_type: Key for image data
            task_prompt: Override task prompt if provided
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.video_length = video_length
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.image_type = image_type
        
        # Determine if we are loading from HDF5 or Directory
        self.is_hdf5 = self.data_path.suffix == ".hdf5"
        
        if self.is_hdf5:
            self.f = h5py.File(self.data_path, "r")
            self.episodes = sorted(list(self.f["data"].keys()))
            # Auto-detect prompt from filename if not provided
            if task_prompt is None:
                self.task_prompt = self._infer_prompt_from_path()
            else:
                self.task_prompt = task_prompt
        else:
            # Assume directory of episode folders
            self.episodes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
            self.task_prompt = task_prompt or DEFAULT_PROMPT

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _infer_prompt_from_path(self) -> str:
        name = self.data_path.stem.lower()
        for key, prompt in TASK_PROMPTS.items():
            if key.lower() in name:
                return prompt
        return DEFAULT_PROMPT

    def __len__(self) -> int:
        return len(self.episodes)

    def _load_frame_from_hdf5(self, ep_name: str, frame_idx: int) -> Image.Image:
        # Note: MimicGen HDF5 usually stores images as uint8 arrays
        img_data = self.f[f"data/{ep_name}/obs/{self.image_type}"][frame_idx]
        return Image.fromarray(img_data)

    def _load_frame_from_dir(self, ep_name: str, frame_idx: int) -> Image.Image:
        # Expecting files like episode_folder/frame_0000.png
        # This structure depends on how we render the high-res images
        ep_dir = self.data_path / ep_name
        img_path = ep_dir / f"frame_{frame_idx:04d}.png"
        if not img_path.exists():
            # Fallback for different naming conventions
            img_path = ep_dir / f"{frame_idx}.png"
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_name = self.episodes[idx]
        
        if self.is_hdf5:
            num_frames = self.f[f"data/{ep_name}/obs/{self.image_type}"].shape[0]
        else:
            # Simple count of files in dir
            num_frames = len(list((self.data_path / ep_name).glob("*.png")))

        required_frames = self.video_length * self.frame_stride
        
        # For I2V LoRA, we often want the first frame to be the conditioning one
        # but we can sample a clip starting from anywhere if we want diversity.
        # However, LiberoI2VDataset in this repo seems to prefer start_idx=0 or random
        if num_frames <= required_frames:
            start_idx = 0
        else:
            start_idx = random.randint(0, num_frames - required_frames)

        frame_indices = [
            start_idx + i * self.frame_stride 
            for i in range(self.video_length)
        ]
        
        # Clamp indices to num_frames - 1
        frame_indices = [min(i, num_frames - 1) for i in frame_indices]

        frames = []
        for fi in frame_indices:
            try:
                if self.is_hdf5:
                    img = self._load_frame_from_hdf5(ep_name, fi)
                else:
                    img = self._load_frame_from_dir(ep_name, fi)
                frames.append(self.transform(img))
            except Exception as e:
                print(f"Error loading frame {fi} from {ep_name}: {e}")
                # Fallback to black image or repeat last
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, *self.resolution))

        video = torch.stack(frames, dim=1) # (C, T, H, W)
        
        return {
            "video": video,
            "caption": self.task_prompt,
            "fps": torch.tensor(10), # standard for robosuite
        }

def create_mimicgen_dataloader(
    data_path: str,
    batch_size: int = 1,
    video_length: int = 8,
    frame_stride: int = 2,
    resolution: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MimicGenI2VDataset(
        data_path=data_path,
        video_length=video_length,
        frame_stride=frame_stride,
        resolution=resolution,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to HDF5 or images dir")
    args = parser.parse_args()
    
    dataset = MimicGenI2VDataset(args.path)
    print(f"Dataset loaded with {len(dataset)} episodes.")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample video shape: {sample['video'].shape}")
        print(f"Sample caption: {sample['caption']}")
