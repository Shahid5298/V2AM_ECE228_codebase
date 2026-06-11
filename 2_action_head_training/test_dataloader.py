"""Quick test to verify MimicGenWindowDataset shapes with decoupled video/action strides."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.flow_matching.config import FlowMatchingConfig
from src.mimicgen_dataset import MimicGenWindowDataset

config = FlowMatchingConfig()
config.num_workers = 0  # single process for debugging

print(f"Config:")
print(f"  num_frames={config.num_frames}, frame_stride={config.frame_stride}")
print(f"  chunk_size={config.chunk_size}, action_stride={config.action_stride}")
print(f"  proprio_history_size={config.proprio_history_size}")
print(f"  raw window = {config.frame_stride * config.num_frames} timesteps")

print("\nBuilding MimicGenWindowDataset (parquet)...")
ds = MimicGenWindowDataset(config)
print(f"Total windows: {len(ds)}")
print(f"Task names ({len(ds.task_names)}): {ds.task_names[:3]}...")

sample = ds[0]
print(f"\nSample[0]:")
print(f"  frames (main):  {sample['frames'].shape} dtype={sample['frames'].dtype}")
print(f"  frames (wrist): {sample['frames_wrist'].shape} dtype={sample['frames_wrist'].dtype}")
print(f"  proprio:        {sample['proprio'].shape} dtype={sample['proprio'].dtype}")
print(f"  actions:        {sample['actions'].shape} dtype={sample['actions'].dtype}")
print(f"  task_id:        {sample['task_id']}")

# Verify expected shapes
assert sample['frames'].shape[0] == config.num_frames, f"Expected {config.num_frames} video frames"
assert sample['actions'].shape[0] == config.chunk_size, f"Expected {config.chunk_size} action steps"
assert sample['proprio'].shape[0] == config.proprio_history_size, f"Expected {config.proprio_history_size} proprio steps"

# Test a sample from near episode start (to verify proprio padding)
sample_early = ds[0]  # start=0, should pad proprio
assert sample_early['proprio'].shape[0] == config.proprio_history_size, "Proprio padding failed"

print("\n✓ All shape assertions passed!")
print("Dataloader test PASSED!")
