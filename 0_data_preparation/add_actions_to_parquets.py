"""
Add missing actions and proprioceptive state from HDF5 to pre-rendered Parquet files.

This avoids having to re-render the images, which is very slow.
It matches the episode number in the Parquet filename to the original HDF5 file
and episode, extracts the required data, and saves the Parquet back to disk.

Usage:
  python add_actions_to_parquets.py \
    --core_dir ~/.cache/huggingface/hub/datasets--amandlek--mimicgen_datasets/snapshots/.../core \
    --data_dir ~/mimicgen_training_data_250/data/chunk-000 \
    --max_episodes 250
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_hdf5_files(core_dir: Path):
    """Get the sorted list of HDF5 files to match the original prepare script order."""
    hdf5_files = sorted([
        f for f in core_dir.glob("*.hdf5")
        if "unused" not in str(f)
    ])
    return hdf5_files


def get_episode_data(h5_path: Path, episode_idx: int):
    """
    Given an HDF5 path and the specific episode index (e.g. 0 to 249),
    extract the action and proprioception data.
    """
    f = h5py.File(h5_path, "r")
    demos = sorted(list(f["data"].keys()))
    
    if episode_idx >= len(demos):
        f.close()
        return None, None
        
    ep = demos[episode_idx]
    
    # Extract actions (T, action_dim)
    actions = f[f"data/{ep}/actions"][()]
    
    # Extract proprioception from obs
    obs_group = f[f"data/{ep}/obs"]
    
    # We want to build a flat 'observation.state' vector similar to LIBERO
    # Typically includes: eef_pos (3), eef_quat (4), gripper_qpos (2), joint_pos (7)
    eef_pos = obs_group["robot0_eef_pos"][()]
    eef_quat = obs_group["robot0_eef_quat"][()]
    gripper_qpos = obs_group["robot0_gripper_qpos"][()]
    joint_pos = obs_group["robot0_joint_pos"][()]
    
    # Concatenate along feature dimension -> (T, feature_dim)
    states = np.concatenate([eef_pos, eef_quat, gripper_qpos, joint_pos], axis=1)
    
    f.close()
    return actions, states


def process_parquet_file(parquet_path: Path, h5_path: Path, h5_episode_idx: int):
    """
    Load the Parquet, append the action and state columns, and overwrite it.
    """
    df = pd.read_parquet(parquet_path)
    
    actions, states = get_episode_data(h5_path, h5_episode_idx)
    
    if actions is None or states is None:
        print(f"  Warning: Cannot extract data from {h5_path.name} episode {h5_episode_idx}")
        return False
        
    # Check if lengths match
    if len(df) != len(actions):
        print(f"  Warning: Length mismatch in {parquet_path.name}. Parquet: {len(df)}, HDF5: {len(actions)}")
        # Sometimes there's an off-by-one difference (e.g. states vs actions in some datasets)
        # We truncate to the min length
        min_len = min(len(df), len(actions))
        df = df.iloc[:min_len].copy()
        actions = actions[:min_len]
        states = states[:min_len]
        
    # Add columns. They need to be stored as 1D numpy arrays per row for PyArrow to serialize them.
    df["action"] = [act for act in actions]
    df["observation.state"] = [st for st in states]
    
    # Also add individual proprioception columns just in case
    with h5py.File(h5_path, "r") as f:
        demos = sorted(list(f["data"].keys()))
        ep_name = demos[h5_episode_idx]
        obs_group = f[f"data/{ep_name}/obs"]
        
        df["observation.robot0_eef_pos"] = [obs for obs in obs_group["robot0_eef_pos"][()][:len(df)]]
        df["observation.robot0_eef_quat"] = [obs for obs in obs_group["robot0_eef_quat"][()][:len(df)]]
        df["observation.robot0_gripper_qpos"] = [obs for obs in obs_group["robot0_gripper_qpos"][()][:len(df)]]
        df["observation.robot0_joint_pos"] = [obs for obs in obs_group["robot0_joint_pos"][()][:len(df)]]
    
    # Overwrite
    df.to_parquet(parquet_path, engine="pyarrow")
    return True


def main(args):
    core_dir = Path(os.path.expanduser(args.core_dir))
    data_dir = Path(os.path.expanduser(args.data_dir))
    max_episodes = args.max_episodes
    
    hdf5_files = get_hdf5_files(core_dir)
    print(f"Found {len(hdf5_files)} HDF5 files.")
    
    parquet_files = sorted(glob.glob(str(data_dir / "episode_*.parquet")))
    print(f"Found {len(parquet_files)} Parquet files to update.")
    
    if not parquet_files:
        return
        
    success_count = 0
    
    for pq_file in tqdm(parquet_files, desc="Updating Parquets"):
        pq_path = Path(pq_file)
        
        # The filename is something like episode_000000.parquet
        # We extract the number to figure out which HDF5 and which episode it belongs to
        try:
            ep_num = int(pq_path.stem.split("_")[1])
        except ValueError:
            print(f"Skipping {pq_path.name} (invalid format)")
            continue
            
        task_idx = ep_num // max_episodes
        h5_episode_idx = ep_num % max_episodes
        
        if task_idx >= len(hdf5_files):
            print(f"task_idx {task_idx} out of bounds for {pq_path.name}")
            continue
            
        h5_path = hdf5_files[task_idx]
        
        if process_parquet_file(pq_path, h5_path, h5_episode_idx):
            success_count += 1
            
    print(f"\nSuccessfully updated {success_count} / {len(parquet_files)} Parquet files with actions and proprioception.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core_dir", type=str, required=True, help="Path to core HDF5 directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to chunk-000 Parquet directory")
    parser.add_argument("--max_episodes", type=int, default=250, help="Max episodes originally rendered per task")
    args = parser.parse_args()
    main(args)
