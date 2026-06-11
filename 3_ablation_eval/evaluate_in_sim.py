"""Evaluate ground truth or trained policy actions in robosuite simulation.

Usage:
  # Ground truth playback at different strides (auto-resolves HDF5+Parquets from task ID):
  python evaluate_in_sim.py --mode gt --task-ids 0 \
      --strides 2 4 8 --num-demos 5 --render

  # Trained policy evaluation (semi-open-loop with GT frames):
  python evaluate_in_sim.py --mode policy --task-ids 0 \
      --checkpoint outputs/flow_matching_task_0/best.pt \
      --num-demos 1 --render

Requires: conda activate ml  (MuJoCo + robosuite + mimicgen + torch installed)
"""

import argparse
import io
import json
import os
import sys
from collections import deque
from types import SimpleNamespace
from pathlib import Path

import h5py
import imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation as R

# External dependency: set MIMICGEN_REPO to the directory containing `mimicgen/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
MIMICGEN_PKG_ROOT = Path(os.environ.get("MIMICGEN_REPO", str(Path.home() / "mimicgen_sim")))
if str(MIMICGEN_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMICGEN_PKG_ROOT))

import mimicgen.envs.robosuite  # load env classes
import robosuite
from robosuite.environments.base import register_env

# Explicitly register MimicGen custom environments
from mimicgen.envs.robosuite.stack import Stack_D0, Stack_D1, StackThree, StackThree_D0, StackThree_D1
from mimicgen.envs.robosuite.coffee import Coffee, Coffee_D0, Coffee_D1, Coffee_D2
from mimicgen.envs.robosuite.mug_cleanup import MugCleanup, MugCleanup_D0, MugCleanup_D1
from mimicgen.envs.robosuite.nut_assembly import Square_D0, Square_D1, Square_D2
from mimicgen.envs.robosuite.threading import Threading, Threading_D0, Threading_D1, Threading_D2
from mimicgen.envs.robosuite.three_piece_assembly import ThreePieceAssembly, ThreePieceAssembly_D0, ThreePieceAssembly_D1, ThreePieceAssembly_D2

for cls in [
    Stack_D0, Stack_D1, StackThree, StackThree_D0, StackThree_D1,
    Coffee, Coffee_D0, Coffee_D1, Coffee_D2,
    MugCleanup, MugCleanup_D0, MugCleanup_D1,
    Square_D0, Square_D1, Square_D2,
    Threading, Threading_D0, Threading_D1, Threading_D2,
    ThreePieceAssembly, ThreePieceAssembly_D0, ThreePieceAssembly_D1, ThreePieceAssembly_D2
]:
    register_env(cls)

# Default location for MimicGen core HDF5 files (for env XML/states)
DEFAULT_CORE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--amandlek--mimicgen_datasets/"
    "snapshots/9bcacb0446b0f895dd352164bd39938710df4a1e/core"
)
# Default location for the training dataset metadata and parquets (100 ep subset)
DEFAULT_MIMICGEN_ROOT = os.path.expanduser(
    "~/mimicgen_training_data_100_with_actions"
)
# 250 ep dataset with hi-res images + states (used for GT frame conditioning)
DEFAULT_MIMICGEN_250 = os.path.expanduser(
    "~/mimicgen_training_data_250"
)

# Number of episodes per source HDF5 file in the unified parquet dataset
TAKE_PER_FILE = {
    "coffee_d0": 33, "coffee_d1": 33, "coffee_d2": 34,
    "coffee_preparation_d0": 50, "coffee_preparation_d1": 50,
    "mug_cleanup_d0": 50, "mug_cleanup_d1": 50,
    "square_d0": 33, "square_d1": 33, "square_d2": 34,
    "stack_d0": 50, "stack_d1": 50,
    "stack_three_d0": 50, "stack_three_d1": 50,
    "threading_d0": 33, "threading_d1": 33, "threading_d2": 34,
    "three_piece_assembly_d0": 33, "three_piece_assembly_d1": 33, "three_piece_assembly_d2": 34,
}
# Ordered list of files in the unified dataset
FILES_ORDER = [
    "coffee_d0", "coffee_d1", "coffee_d2",
    "coffee_preparation_d0", "coffee_preparation_d1",
    "mug_cleanup_d0", "mug_cleanup_d1",
    "square_d0", "square_d1", "square_d2",
    "stack_d0", "stack_d1",
    "stack_three_d0", "stack_three_d1",
    "threading_d0", "threading_d1", "threading_d2",
    "three_piece_assembly_d0", "three_piece_assembly_d1", "three_piece_assembly_d2",
]


def resolve_h5_paths_from_task_ids(task_ids, mimicgen_root, core_dir):
    """Map task IDs to HDF5 file paths using tasks.jsonl.

    Returns:
        list of (task_id, h5_path, task_description, ep_start_idx) tuples
    """
    tasks_jsonl = Path(mimicgen_root) / "meta" / "tasks.jsonl"
    if not tasks_jsonl.exists():
        raise FileNotFoundError(f"tasks.jsonl not found at {tasks_jsonl}")

    task_map = {}
    with open(tasks_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            task_map[entry["task_index"]] = entry

    # Compute global episode offsets
    ep_offsets = {}
    current_offset = 0
    for fname in FILES_ORDER:
        ep_offsets[fname] = current_offset
        current_offset += TAKE_PER_FILE[fname]

    results = []
    for tid in task_ids:
        if tid not in task_map:
            print(f"WARNING: task_index {tid} not found in tasks.jsonl, skipping")
            continue
        source_file = task_map[tid]["source_file"]
        h5_path = Path(core_dir) / source_file
        if not h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {h5_path}\n"
                f"Looked for source_file='{source_file}' from task_index={tid}\n"
                f"Use --core-dir to specify the directory containing the HDF5 files."
            )
        
        base_name = source_file.replace(".hdf5", "")
        ep_start = ep_offsets.get(base_name, 0)
        
        results.append((tid, str(h5_path), task_map[tid]["task"], ep_start))
        
    return results


def create_env_from_h5(h5_file, render_hw=256):
    """Create a robosuite env from the HDF5 metadata."""
    env_meta = json.loads(h5_file["data"].attrs["env_args"])
    env_name = env_meta["env_name"]
    env_kwargs = env_meta["env_kwargs"]

    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_heights"] = render_hw
    env_kwargs["camera_widths"] = render_hw

    env = robosuite.make(env_name, **env_kwargs)
    return env


def reset_env_to_demo(env, h5_file, demo_key):
    """Reset the environment to the initial state of a specific demo."""
    model_xml = h5_file[f"data/{demo_key}"].attrs["model_file"]
    states = h5_file[f"data/{demo_key}/states"][()]

    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()

    # Set to initial state
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    return states


def render_frame(env, camera_name="agentview", height=256, width=256):
    """Render a frame from the environment."""
    frame = env.sim.render(height=height, width=width, camera_name=camera_name)
    return frame[::-1]  # flip vertically (MuJoCo convention)


def load_parquet_actions(mimicgen_root, global_ep_idx):
    """Load Ground Truth actions from the unified parquet dataset."""
    pq_path = Path(mimicgen_root) / "data" / "chunk-000" / f"episode_{global_ep_idx:06d}.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {pq_path}")
    
    table = pq.read_table(str(pq_path), columns=["action"])
    actions = np.stack(table["action"].to_numpy(zero_copy_only=False))
    return actions


def load_parquet_episode(mimicgen_root, global_ep_idx):
    """Load a full episode from the parquet dataset (frames, states, proprio, actions).
    
    Returns dict with:
        frames_main: list of (H,W,3) uint8 numpy arrays (agentview)
        frames_wrist: list of (H,W,3) uint8 numpy arrays (wrist)
        states: (T, state_dim) float array for sim reset
        proprio: (T, 8) float array [pos(3), euler(3), gripper(2)]
        actions: (T, 7) float array
    """
    pq_path = Path(mimicgen_root) / "data" / "chunk-000" / f"episode_{global_ep_idx:06d}.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {pq_path}")
    
    table = pq.read_table(str(pq_path))
    T = len(table)
    
    # Decode images
    frames_main = []
    frames_wrist = []
    main_dicts = table['observation.image'].to_pylist()
    wrist_dicts = table['observation.image_wrist'].to_pylist()
    for md, wd in zip(main_dicts, wrist_dicts):
        frames_main.append(np.array(Image.open(io.BytesIO(md['bytes'])).convert('RGB')))
        frames_wrist.append(np.array(Image.open(io.BytesIO(wd['bytes'])).convert('RGB')))
    
    # States for sim reset
    states = np.stack(table['observation.state'].to_numpy(zero_copy_only=False)).astype(np.float64)
    
    # Proprio
    pos = np.stack(table['observation.robot0_eef_pos'].to_numpy(zero_copy_only=False))
    quat = np.stack(table['observation.robot0_eef_quat'].to_numpy(zero_copy_only=False))
    gripper = np.stack(table['observation.robot0_gripper_qpos'].to_numpy(zero_copy_only=False))
    euler = R.from_quat(quat).as_euler('xyz', degrees=False)
    proprio = np.concatenate([pos, euler, gripper], axis=-1).astype(np.float32)
    
    # Actions
    actions = np.stack(table['action'].to_numpy(zero_copy_only=False)).astype(np.float32)
    
    return {
        'frames_main': frames_main,
        'frames_wrist': frames_wrist,
        'states': states,
        'proprio': proprio,
        'actions': actions,
        'num_steps': T,
    }


def load_hummingbird_latent_episode(latent_root, task_id, global_ep_idx):
    """Load cached Hummingbird latent windows for one parquet episode."""
    import torch

    cache_path = (
        Path(latent_root)
        / f"task_{task_id}"
        / "chunk-000"
        / f"episode_{global_ep_idx:06d}.pt"
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached latent file not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload["cache_path"] = str(cache_path)
    return payload


def get_live_proprio(obs):
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    euler = R.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)
    return np.concatenate([pos, euler, gripper], axis=-1).astype(np.float32)


def render_observation_pair(env, hw=256):
    main = env.sim.render(height=hw, width=hw, camera_name="agentview")[::-1]
    wrist = env.sim.render(height=hw, width=hw, camera_name="robot0_eye_in_hand")[::-1]
    return main, wrist


def aggregate_actions(actions, stride):
    """Aggregate relative actions over non-overlapping windows of size `stride`.

    For OSC_POSE actions (7-dim):
      - dims 0:6 (delta pose): summed to preserve total displacement
      - dim 6 (gripper): take the last value per window
    """
    if stride <= 1:
        return actions
    T, D = actions.shape
    n_windows = T // stride
    trimmed = actions[:n_windows * stride]
    reshaped = trimmed.reshape(n_windows, stride, D)
    agg = np.empty((n_windows, D), dtype=actions.dtype)
    agg[:, :6] = reshaped[:, :, :6].sum(axis=1)
    agg[:, 6] = reshaped[:, -1, 6]
    return agg


def _scale_controller_limits(env, scale_factor):
    """Scale the OSC controller's input/output limits by a factor.

    This allows aggregated actions (summed over stride windows) to pass
    through without being clipped to [-1, 1].
    """
    ctrl = env.robots[0].controller
    ctrl.input_max *= scale_factor
    ctrl.input_min *= scale_factor
    ctrl.output_max *= scale_factor
    ctrl.output_min *= scale_factor


def evaluate_gt_playback(env, h5_file, demo_key, actions, stride=1,
                         render=False, video_writer=None, camera_name="agentview", hw=256):
    """Replay ground truth actions at a given stride and check success.

    At stride S, we aggregate S consecutive actions (sum deltas, last gripper)
    and execute the aggregated action once. The controller's input/output limits
    are scaled by S so the summed action is not clipped.

    Returns:
        dict with keys: success, num_steps, total_gt_steps
    """
    # Reset env using HDF5 metadata
    states = reset_env_to_demo(env, h5_file, demo_key)

    # Scale controller limits so aggregated actions don't get clipped
    if stride > 1:
        _scale_controller_limits(env, stride)

    # Aggregate actions over stride windows (sum deltas, take last gripper)
    agg_actions = aggregate_actions(actions, stride)

    success = False
    frames = []

    for i, action in enumerate(agg_actions):
        if render or video_writer is not None:
            frame = render_frame(env, camera_name, hw, hw)
            frames.append(frame)

        obs, reward, done, info = env.step(action)

        if env._check_success():
            success = True

    # Capture final frame
    if render or video_writer is not None:
        frame = render_frame(env, camera_name, hw, hw)
        frames.append(frame)

    # Write video
    if video_writer is not None and frames:
        for f in frames:
            video_writer.append_data(f)

    # Restore original controller limits
    if stride > 1:
        _scale_controller_limits(env, 1.0 / stride)

    return {
        "success": success,
        "num_steps": len(agg_actions),
        "total_gt_steps": len(actions),
    }


def evaluate_policy(env, h5_file, demo_key, episode_data, model, videomae, config,
                    action_mean, action_std, render=False, video_writer=None,
                    camera_name="agentview", hw=256,
                    reset_mode="always", reset_threshold=1.0,
                    execute_steps=4):
    """Semi-open-loop policy evaluation with GT frames.
    
    Every replanning step:
      1. Optionally reset sim to GT state (based on reset_mode)
      2. Extract 8 GT frames (stride 2) from both cameras
      3. Get GT proprio at window start
      4. Model predicts 16 actions
      5. Execute only the first `execute_steps` actions, then replan
    
    Args:
        reset_mode: 'always' = reset every window (semi-open-loop)
                    'never'  = free flow (open-loop, no reset after initial)
                    'adaptive' = reset only if sim state diverges from GT
        reset_threshold: L2 distance threshold for adaptive mode
        execute_steps: number of predicted actions to execute before replanning
    """
    import torch
    from transformers import VideoMAEImageProcessor

    # Reset env to initial demo state (for model XML) and get full sim states
    states_h5 = reset_env_to_demo(env, h5_file, demo_key)
    # states_h5 shape: (T+1, state_dim) — full MuJoCo flattened states from HDF5

    processor = VideoMAEImageProcessor.from_pretrained(str(config.model_dir))

    device = config.device
    a_mean = action_mean.to(device)
    a_std = action_std.to(device)

    num_frames = config.num_frames      # 8
    frame_stride = config.frame_stride   # 2
    future_frame_offset = getattr(config, "future_frame_offset", 0)
    chunk_size = config.chunk_size       # 16
    execute_steps = min(execute_steps, chunk_size)

    gt_frames_main = episode_data['frames_main']   # list of (H,W,3)
    gt_frames_wrist = episode_data['frames_wrist'] # list of (H,W,3)
    gt_proprio = episode_data['proprio']           # (T, 8)
    T = episode_data['num_steps']
    # Use HDF5 states for sim reset (full MuJoCo state, not parquet observation.state)
    # states_h5 has T+1 entries (initial + one per action step)
    gt_sim_states = states_h5

    success = False
    frames_for_video = []
    step_count = 0
    num_windows = 0
    num_resets = 0

    # Receding-horizon rollout: plan chunk_size actions, execute execute_steps.
    for win_start in range(0, T, execute_steps):
        # --- 1. Optionally reset sim to GT state at window start ---
        should_reset = False
        if reset_mode == "always":
            should_reset = True
        elif reset_mode == "never":
            should_reset = (win_start == 0)  # only reset at the very start
        elif reset_mode == "adaptive":
            if win_start == 0:
                should_reset = True  # always reset at start
            elif win_start < len(gt_sim_states):
                # Compare current sim state with GT state
                current_state = env.sim.get_state().flatten()
                gt_state = gt_sim_states[win_start]
                # Compare qpos portion (most meaningful for divergence)
                min_len = min(len(current_state), len(gt_state))
                state_dist = np.linalg.norm(current_state[:min_len] - gt_state[:min_len])
                should_reset = state_dist > reset_threshold
                if should_reset:
                    print(f"    [adaptive] Reset at step {win_start}: state_dist={state_dist:.4f} > {reset_threshold}")

        if should_reset and win_start < len(gt_sim_states):
            env.sim.set_state_from_flattened(gt_sim_states[win_start])
            env.sim.forward()
            num_resets += 1

        # Render the GT-reset frame for video
        if render or video_writer is not None:
            frame = render_frame(env, camera_name, hw, hw)
            frames_for_video.append(frame)

        # --- 2. Extract future GT frames for action-conditioned rollout ---
        frame_indices = [
            win_start + future_frame_offset + j * frame_stride
            for j in range(num_frames)
        ]
        gt_main_window = [gt_frames_main[idx] for idx in frame_indices if idx < T]
        gt_wrist_window = [gt_frames_wrist[idx] for idx in frame_indices if idx < T]

        # Pad if we don't have enough frames at the end
        while len(gt_main_window) < num_frames:
            gt_main_window.append(gt_main_window[-1])
            gt_wrist_window.append(gt_wrist_window[-1])

        # --- 3. Get GT proprio at window start ---
        proprio_vec = gt_proprio[win_start]  # (8,)
        proprio = torch.from_numpy(proprio_vec).unsqueeze(0).unsqueeze(0)  # (1,1,8)
        proprio = proprio.expand(1, config.proprio_history_size, -1).to(device)

        history_num_frames = getattr(config, "history_num_frames", 1)
        history_indices = [
            max(0, win_start - (history_num_frames - 1 - j) * frame_stride)
            for j in range(history_num_frames)
        ]
        history_main_window = [gt_frames_main[idx] for idx in history_indices]
        history_wrist_window = [gt_frames_wrist[idx] for idx in history_indices]
        history_resample_idx = np.linspace(0, len(history_main_window) - 1, num_frames).round().astype(int)
        history_main_window = [history_main_window[idx] for idx in history_resample_idx]
        history_wrist_window = [history_wrist_window[idx] for idx in history_resample_idx]

        if not config.include_history_frames and not getattr(config, "condition_on_future_video", True):
            raise ValueError("At least one visual conditioning stream must be enabled.")

        with torch.no_grad():
            video_feature_chunks = []
            if config.include_history_frames:
                pv_history_main = processor(history_main_window, return_tensors="pt")["pixel_values"].to(device)
                pv_history_wrist = processor(history_wrist_window, return_tensors="pt")["pixel_values"].to(device)
                video_feature_chunks.extend([
                    videomae(pv_history_main),
                    videomae(pv_history_wrist),
                ])
            if getattr(config, "condition_on_future_video", True):
                pv_main = processor(gt_main_window, return_tensors="pt")["pixel_values"].to(device)
                pv_wrist = processor(gt_wrist_window, return_tensors="pt")["pixel_values"].to(device)
                video_feature_chunks.extend([
                    videomae(pv_main),
                    videomae(pv_wrist),
                ])
            video_features = torch.cat(video_feature_chunks, dim=1)

            # --- 5. Predict actions ---
            pred_cont_norm, gripper_logit = model.sample_actions(video_features, proprio)
            pred_cont = pred_cont_norm * a_std + a_mean
            pred_gripper = (torch.sigmoid(gripper_logit) > 0.5).float() * 2.0 - 1.0
            pred_actions = torch.cat([pred_cont, pred_gripper], dim=-1).cpu().numpy()[0]

        # --- 6. Execute predicted actions in sim ---
        for act_idx, act in enumerate(pred_actions[:execute_steps]):
            if step_count >= T:
                break

            obs, reward, done, info = env.step(act)
            step_count += 1

            if render or video_writer is not None:
                frame = render_frame(env, camera_name, hw, hw)
                frames_for_video.append(frame)

            if env._check_success():
                success = True

        num_windows += 1

    # Write video
    if video_writer is not None and frames_for_video:
        for f in frames_for_video:
            video_writer.append_data(f)

    return {
        "success": success,
        "num_steps": step_count,
        "num_windows": num_windows,
        "num_resets": num_resets,
    }


def evaluate_policy_hummingbird(
    env,
    h5_file,
    demo_key,
    episode_data,
    latent_payload,
    model,
    videomae,
    config,
    action_mean,
    action_std,
    render=False,
    video_writer=None,
    camera_name="agentview",
    hw=256,
    reset_mode="always",
    reset_threshold=1.0,
    execute_steps=8,
):
    """Semi-open-loop policy evaluation using cached Hummingbird latents + current VideoMAE history."""
    import torch
    from transformers import VideoMAEImageProcessor

    states_h5 = reset_env_to_demo(env, h5_file, demo_key)
    device = config.device
    a_mean = action_mean.to(device)
    a_std = action_std.to(device)
    processor = VideoMAEImageProcessor.from_pretrained(str(config.model_dir))

    gt_frames_main = episode_data["frames_main"]
    gt_frames_wrist = episode_data["frames_wrist"]
    gt_proprio = episode_data["proprio"]
    T = episode_data["num_steps"]
    gt_sim_states = states_h5

    starts = [int(s) for s in latent_payload["starts"].tolist()]
    if not starts:
        raise ValueError(f"No cached latent windows found in {latent_payload.get('cache_path', '<unknown>')}")
    latent_by_start = {start: idx for idx, start in enumerate(starts)}
    latent_plan_stride = starts[1] - starts[0] if len(starts) > 1 else config.window_stride
    if execute_steps != latent_plan_stride:
        print(
            f"  [info] overriding execute_steps from {execute_steps} to cached latent stride "
            f"{latent_plan_stride} so replanning aligns with available windows"
        )
        execute_steps = latent_plan_stride

    success = False
    frames_for_video = []
    step_count = 0
    num_windows = 0
    num_resets = 0

    for win_start in starts:
        if win_start >= T:
            break

        should_reset = False
        if reset_mode == "always":
            should_reset = True
        elif reset_mode == "never":
            should_reset = (win_start == starts[0])
        elif reset_mode == "adaptive":
            if win_start == starts[0]:
                should_reset = True
            elif win_start < len(gt_sim_states):
                current_state = env.sim.get_state().flatten()
                gt_state = gt_sim_states[win_start]
                min_len = min(len(current_state), len(gt_state))
                state_dist = np.linalg.norm(current_state[:min_len] - gt_state[:min_len])
                should_reset = state_dist > reset_threshold
                if should_reset:
                    print(f"    [adaptive] Reset at step {win_start}: state_dist={state_dist:.4f} > {reset_threshold}")

        if should_reset and win_start < len(gt_sim_states):
            env.sim.set_state_from_flattened(gt_sim_states[win_start])
            env.sim.forward()
            num_resets += 1

        if render or video_writer is not None:
            frame = render_frame(env, camera_name, hw, hw)
            frames_for_video.append(frame)

        proprio_hist = gt_proprio[max(0, win_start - config.proprio_history_size + 1):win_start + 1]
        if proprio_hist.shape[0] < config.proprio_history_size:
            pad = np.repeat(proprio_hist[:1], config.proprio_history_size - proprio_hist.shape[0], axis=0)
            proprio_hist = np.concatenate([pad, proprio_hist], axis=0)
        proprio = torch.from_numpy(proprio_hist).unsqueeze(0).to(device)

        history_num_frames = getattr(config, "history_num_frames", 1)
        frame_stride = getattr(config, "frame_stride", 1)
        num_frames = config.num_frames
        history_indices = [
            max(0, win_start - (history_num_frames - 1 - j) * frame_stride)
            for j in range(history_num_frames)
        ]
        history_main_window = [gt_frames_main[idx] for idx in history_indices]
        history_wrist_window = [gt_frames_wrist[idx] for idx in history_indices]
        history_resample_idx = np.linspace(0, len(history_main_window) - 1, num_frames).round().astype(int)
        history_main_window = [history_main_window[idx] for idx in history_resample_idx]
        history_wrist_window = [history_wrist_window[idx] for idx in history_resample_idx]

        latent_idx = latent_by_start[win_start]
        latent_video = latent_payload["latents"][latent_idx].unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pv_history_main = processor(history_main_window, return_tensors="pt")["pixel_values"].to(device)
            pv_history_wrist = processor(history_wrist_window, return_tensors="pt")["pixel_values"].to(device)
            current_video_features = torch.cat(
                [videomae(pv_history_main), videomae(pv_history_wrist)],
                dim=1,
            )
            pred_cont_norm, gripper_logit = model.sample_actions(
                latent_video, proprio, current_video_features=current_video_features,
            )
            pred_cont = pred_cont_norm * a_std + a_mean
            pred_gripper = (torch.sigmoid(gripper_logit) > 0.5).float() * 2.0 - 1.0
            pred_actions = torch.cat([pred_cont, pred_gripper], dim=-1).cpu().numpy()[0]

        max_exec = min(execute_steps, T - step_count)
        for act in pred_actions[:max_exec]:
            if step_count >= T:
                break

            env.step(act)
            step_count += 1

            if render or video_writer is not None:
                frame = render_frame(env, camera_name, hw, hw)
                frames_for_video.append(frame)

            if env._check_success():
                success = True

        num_windows += 1

    if video_writer is not None and frames_for_video:
        for f in frames_for_video:
            video_writer.append_data(f)

    return {
        "success": success,
        "num_steps": step_count,
        "num_windows": num_windows,
        "num_resets": num_resets,
        "latent_stride": latent_plan_stride,
    }


def evaluate_policy_hummingbird_online(
    env,
    h5_file,
    demo_key,
    task_desc,
    model,
    videomae,
    hummingbird_model,
    hummingbird_namespace,
    hummingbird_reneg_path,
    config,
    action_mean,
    action_std,
    render=False,
    video_writer=None,
    camera_name="agentview",
    hw=256,
    reset_mode="always",
    reset_threshold=1.0,
    execute_steps=4,
):
    """Closed-loop eval: live env obs -> Hummingbird + LoRA -> action head."""
    import torch
    from PIL import Image
    from transformers import VideoMAEImageProcessor
    from cache_hummingbird_latents import make_latent

    states_h5 = reset_env_to_demo(env, h5_file, demo_key)
    current_obs = env._get_observations(force_update=True)
    device = config.device
    a_mean = action_mean.to(device)
    a_std = action_std.to(device)
    processor = VideoMAEImageProcessor.from_pretrained(str(config.model_dir))

    T = len(states_h5) - 1
    gt_sim_states = states_h5
    success = False
    frames_for_video = []
    step_count = 0
    num_windows = 0
    num_resets = 0

    history_main = deque(maxlen=config.history_num_frames)
    history_wrist = deque(maxlen=config.history_num_frames)
    proprio_history = deque(maxlen=config.proprio_history_size)
    first_main, first_wrist = render_observation_pair(env, hw=hw)
    first_proprio = get_live_proprio(current_obs)
    for _ in range(config.history_num_frames):
        history_main.append(first_main.copy())
        history_wrist.append(first_wrist.copy())
    for _ in range(config.proprio_history_size):
        proprio_history.append(first_proprio.copy())

    while step_count < T:
        should_reset = False
        if reset_mode == "always":
            should_reset = True
        elif reset_mode == "never":
            should_reset = (step_count == 0)
        elif reset_mode == "adaptive":
            if step_count == 0:
                should_reset = True
            elif step_count < len(gt_sim_states):
                current_state = env.sim.get_state().flatten()
                gt_state = gt_sim_states[step_count]
                min_len = min(len(current_state), len(gt_state))
                state_dist = np.linalg.norm(current_state[:min_len] - gt_state[:min_len])
                should_reset = state_dist > reset_threshold
                if should_reset:
                    print(f"    [adaptive] Reset at step {step_count}: state_dist={state_dist:.4f} > {reset_threshold}")

        if should_reset and step_count < len(gt_sim_states):
            env.sim.set_state_from_flattened(gt_sim_states[step_count])
            env.sim.forward()
            current_obs = env._get_observations(force_update=True)
            main_frame, wrist_frame = render_observation_pair(env, hw=hw)
            current_proprio = get_live_proprio(current_obs)
            history_main.clear()
            history_wrist.clear()
            proprio_history.clear()
            for _ in range(config.history_num_frames):
                history_main.append(main_frame.copy())
                history_wrist.append(wrist_frame.copy())
            for _ in range(config.proprio_history_size):
                proprio_history.append(current_proprio.copy())
            num_resets += 1
        else:
            main_frame, wrist_frame = render_observation_pair(env, hw=hw)

        if render or video_writer is not None:
            frames_for_video.append(render_frame(env, camera_name, hw, hw))

        proprio = torch.from_numpy(np.stack(list(proprio_history))).unsqueeze(0).to(device)

        history_main_window = [frame.copy() for frame in history_main]
        history_wrist_window = [frame.copy() for frame in history_wrist]
        history_resample_idx = np.linspace(0, len(history_main_window) - 1, config.num_frames).round().astype(int)
        history_main_window = [history_main_window[idx] for idx in history_resample_idx]
        history_wrist_window = [history_wrist_window[idx] for idx in history_resample_idx]

        current_image = Image.fromarray(main_frame)
        latent_video = make_latent(
            hummingbird_model,
            torch.device(device),
            current_image,
            task_desc,
            hummingbird_namespace,
            hummingbird_reneg_path,
        ).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pv_history_main = processor(history_main_window, return_tensors="pt")["pixel_values"].to(device)
            pv_history_wrist = processor(history_wrist_window, return_tensors="pt")["pixel_values"].to(device)
            current_video_features = torch.cat(
                [videomae(pv_history_main), videomae(pv_history_wrist)],
                dim=1,
            )
            pred_cont_norm, gripper_logit = model.sample_actions(
                latent_video, proprio, current_video_features=current_video_features,
            )
            pred_cont = pred_cont_norm * a_std + a_mean
            pred_gripper = (torch.sigmoid(gripper_logit) > 0.5).float() * 2.0 - 1.0
            pred_actions = torch.cat([pred_cont, pred_gripper], dim=-1).cpu().numpy()[0]

        max_exec = min(execute_steps, T - step_count)
        for act in pred_actions[:max_exec]:
            current_obs, _, _, _ = env.step(act)
            step_count += 1
            main_frame, wrist_frame = render_observation_pair(env, hw=hw)
            proprio_history.append(get_live_proprio(current_obs).copy())
            history_main.append(main_frame.copy())
            history_wrist.append(wrist_frame.copy())

            if render or video_writer is not None:
                frames_for_video.append(render_frame(env, camera_name, hw, hw))

            if env._check_success():
                success = True

            if step_count >= T:
                break

        num_windows += 1

    if video_writer is not None and frames_for_video:
        for f in frames_for_video:
            video_writer.append_data(f)

    return {
        "success": success,
        "num_steps": step_count,
        "num_windows": num_windows,
        "num_resets": num_resets,
    }


def run_gt_eval(h5_path, task_desc, ep_start_idx, mimicgen_root, strides, demo_keys, args):
    """Run ground truth playback evaluation for one task."""
    h5_file = h5py.File(h5_path, "r")
    env = create_env_from_h5(h5_file, render_hw=args.render_hw)

    results_table = []

    for stride in strides:
        print(f"\n--- Stride {stride} ---")
        stride_dir = Path(args.output_dir) / f"gt_stride_{stride}"
        stride_dir.mkdir(parents=True, exist_ok=True)

        successes = []
        for i, demo_key in enumerate(demo_keys):
            # Resolve correct parquet episode index
            global_ep_idx = ep_start_idx + i
            actions = load_parquet_actions(mimicgen_root, global_ep_idx)

            video_writer = None
            if args.render:
                video_path = stride_dir / f"{demo_key}.mp4"
                video_writer = imageio.get_writer(str(video_path), fps=20)

            result = evaluate_gt_playback(
                env, h5_file, demo_key, actions,
                stride=stride,
                render=args.render,
                video_writer=video_writer,
                camera_name=args.camera,
                hw=args.render_hw,
            )

            if video_writer is not None:
                video_writer.close()

            status = "✓ SUCCESS" if result["success"] else "✗ FAIL"
            print(f"  {demo_key} (parquet {global_ep_idx:06d}): {status}  "
                  f"(steps: {result['num_steps']}/{result['total_gt_steps']})")
            successes.append(result["success"])

        success_rate = sum(successes) / len(successes) * 100
        results_table.append({
            "stride": stride,
            "num_demos": len(demo_keys),
            "successes": sum(successes),
            "success_rate": success_rate,
        })

    h5_file.close()
    env.close()
    return results_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate actions in robosuite simulation")
    parser.add_argument("--mode", choices=["gt", "policy"], required=True,
                        help="'gt' for ground truth replay, 'policy' for trained model")

    # Data source (pick one)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True,
                        help="Task indices (auto-resolves from tasks.jsonl)")
    parser.add_argument("--core-dir", type=str, default=DEFAULT_CORE_DIR,
                        help="Directory containing source HDF5 files")
    parser.add_argument("--mimicgen-root", type=str, default=DEFAULT_MIMICGEN_ROOT,
                        help="MimicGen training data root (Parquets and tasks.jsonl)")
    parser.add_argument("--mimicgen-250", type=str, default=DEFAULT_MIMICGEN_250,
                        help="MimicGen 250 dataset (hi-res images + states for GT conditioning)")

    # Evaluation settings
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 6],
                        help="Action strides to test (gt mode only)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (policy mode only)")
    parser.add_argument("--policy-input", choices=["videomae", "hummingbird_cached", "hummingbird_online"], default="videomae",
                        help="Condition policy on GT future VideoMAE features, cached Hummingbird latents, or online Hummingbird generation")
    parser.add_argument("--latent-root", type=str, default="outputs/hummingbird_latents",
                        help="Root directory for cached Hummingbird latents (policy-input=hummingbird_cached)")
    parser.add_argument("--hummingbird-root", type=str, default=None,
                        help="Path to /home/.../Hummingbird/i2v (policy-input=hummingbird_online)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to Hummingbird LoRA adapter directory (policy-input=hummingbird_online)")
    parser.add_argument("--hummingbird-ddim-steps", type=int, default=16,
                        help="DDIM steps for online Hummingbird generation")
    parser.add_argument("--hummingbird-guidance-scale", type=float, default=7.5,
                        help="CFG scale for online Hummingbird generation")
    parser.add_argument("--num-demos", type=int, default=5,
                        help="Number of demos to evaluate per task")
    parser.add_argument("--render", action="store_true",
                        help="Save rendered videos")
    parser.add_argument("--output-dir", type=str, default="outputs/sim_eval",
                        help="Output directory for videos and results")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Camera to use for rendering")
    parser.add_argument("--render-hw", type=int, default=256,
                        help="Render height/width")
    parser.add_argument("--reset-mode", choices=["always", "never", "adaptive"], default="always",
                        help="Sim state reset strategy: 'always'=every 16 steps, "
                             "'never'=free flow, 'adaptive'=reset if diverged")
    parser.add_argument("--reset-threshold", type=float, default=1.0,
                        help="L2 state distance threshold for adaptive reset mode")
    parser.add_argument("--execute-steps", type=int, default=4,
                        help="Number of predicted actions to execute before replanning (policy mode only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve HDF5 files and global parquet indexing
    h5_entries = resolve_h5_paths_from_task_ids(
        args.task_ids, args.mimicgen_root, args.core_dir
    )

    if args.mode == "gt":
        print(f"\n{'='*70}")
        print(f"GROUND TRUTH ACTION PLAYBACK (FROM PARQUETS)")
        print(f"Strides: {args.strides}")
        print(f"{'='*70}")

        all_results = []
        for task_id, h5_path, task_desc, ep_start_idx in h5_entries:
            print(f"\n>>> Task {task_id}: {task_desc}")
            print(f"    HDF5: {h5_path}")
            print(f"    Parquet Start Idx: {ep_start_idx}")

            h5_file = h5py.File(h5_path, "r")
            demo_keys = sorted([k for k in h5_file["data"].keys() if k.startswith("demo")])
            demo_keys = demo_keys[:args.num_demos]
            print(f"    Demos: {demo_keys}")
            h5_file.close()

            results = run_gt_eval(h5_path, task_desc, ep_start_idx, args.mimicgen_root, 
                                  args.strides, demo_keys, args)
            all_results.extend(results)

        # Print summary
        print(f"\n{'='*55}")
        print(f"{'Stride':>8} | {'Demos':>6} | {'Success':>8} | {'Rate':>8}")
        print(f"{'-'*55}")
        for r in all_results:
            print(f"{r['stride']:>8} | {r['num_demos']:>6} | "
                  f"{r['successes']:>8} | {r['success_rate']:>7.1f}%")
        print(f"{'='*55}")

    elif args.mode == "policy":
        if args.checkpoint is None:
            print("ERROR: --checkpoint is required for policy mode")
            return

        import torch
        from src.flow_matching.config import FlowMatchingConfig
        from src.flow_matching.model import FlowMatchingActionHead

        print(f"\n{'='*70}")
        print(f"TRAINED POLICY EVALUATION")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Policy input: {args.policy_input}")
        print(f"{'='*70}\n")

        config = FlowMatchingConfig()
        checkpoint_path = Path(args.checkpoint)
        videomae = None

        hummingbird_model = None
        hummingbird_namespace = None
        hummingbird_reneg_path = None

        if args.policy_input == "videomae":
            from src.videomae_encoder import VideoMAEFeatureExtractor

            model = FlowMatchingActionHead(config).to(config.device)
            videomae = VideoMAEFeatureExtractor(
                config.model_dir,
                layer_idx=config.videomae_layer,
                num_frames_expected=config.num_frames,
                device=config.device,
            ).to(config.device)
        else:
            from src.flow_matching.hummingbird_policy import HummingbirdLatentFlowMatchingPolicy
            from src.videomae_encoder import VideoMAEFeatureExtractor

            config.include_history_frames = True
            config.condition_on_future_video = False
            config.visual_feature_dim = config.videomae_hidden_dim
            model = HummingbirdLatentFlowMatchingPolicy(config).to(config.device)
            videomae = VideoMAEFeatureExtractor(
                config.model_dir,
                layer_idx=config.videomae_layer,
                num_frames_expected=config.num_frames,
                device=config.device,
            ).to(config.device)
            if args.policy_input == "hummingbird_online":
                if args.hummingbird_root is None or args.adapter_path is None:
                    raise ValueError("--hummingbird-root and --adapter-path are required for policy-input=hummingbird_online")
                from cache_hummingbird_latents import load_hummingbird_model

                hummingbird_namespace = SimpleNamespace(
                    hummingbird_root=Path(args.hummingbird_root),
                    adapter_path=Path(args.adapter_path),
                    base_config=None,
                    checkpoint=None,
                    unet_path=None,
                    img_proj_path=None,
                    reneg_path=None,
                    device=config.device,
                    height=args.render_hw,
                    width=args.render_hw,
                    video_length=config.num_frames,
                    ddim_steps=args.hummingbird_ddim_steps,
                    guidance_scale=args.hummingbird_guidance_scale,
                    fps_condition=10,
                )
                hummingbird_model, _, hummingbird_reneg_path = load_hummingbird_model(hummingbird_namespace)

        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=config.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        action_mean = ckpt["action_mean"]
        action_std = ckpt["action_std"]

        policy_name = checkpoint_path.parent.name or checkpoint_path.stem
        if args.policy_input == "videomae":
            policy_subdir = "policy"
        elif args.policy_input == "hummingbird_cached":
            policy_subdir = "policy_hummingbird_cached"
        else:
            policy_subdir = "policy_hummingbird_online"
        policy_dir = output_dir / policy_subdir / policy_name
        policy_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving policy eval artifacts to: {policy_dir}")

        # 250 dataset numbering: task_id * 250 + demo_index
        EPISODES_PER_TASK_250 = 250
        mimicgen_250_root = getattr(args, 'mimicgen_250', DEFAULT_MIMICGEN_250)

        for task_id, h5_path, task_desc, ep_start_idx in h5_entries:
            print(f"\n>>> Task {task_id}: {task_desc}")
            h5_file = h5py.File(h5_path, "r")
            env = create_env_from_h5(h5_file, render_hw=args.render_hw)
            demo_keys = sorted([k for k in h5_file["data"].keys() if k.startswith("demo")])
            demo_keys = demo_keys[:args.num_demos]

            successes = []
            for i, demo_key in enumerate(demo_keys):
                if args.policy_input == "videomae":
                    # 250 dataset: episode = task_id * 250 + demo_index
                    global_ep_idx = task_id * EPISODES_PER_TASK_250 + i
                    print(f"  Loading GT episode {global_ep_idx} from 250 dataset...")
                    episode_data = load_parquet_episode(mimicgen_250_root, global_ep_idx)
                    latent_payload = None
                elif args.policy_input == "hummingbird_cached":
                    global_ep_idx = ep_start_idx + i
                    print(f"  Loading cached latent episode {global_ep_idx} from 100 dataset...")
                    episode_data = load_parquet_episode(args.mimicgen_root, global_ep_idx)
                    latent_payload = load_hummingbird_latent_episode(args.latent_root, task_id, global_ep_idx)
                else:
                    global_ep_idx = ep_start_idx + i
                    print(f"  Running online Hummingbird rollout for episode {global_ep_idx}...")
                    episode_data = None
                    latent_payload = None
                
                video_writer = None
                if args.render:
                    video_path = policy_dir / f"task{task_id}_{demo_key}.mp4"
                    video_writer = imageio.get_writer(str(video_path), fps=20)

                if args.policy_input == "videomae":
                    result = evaluate_policy(
                        env, h5_file, demo_key, episode_data,
                        model, videomae, config,
                        action_mean, action_std,
                        render=args.render,
                        video_writer=video_writer,
                        camera_name=args.camera,
                        hw=args.render_hw,
                        reset_mode=args.reset_mode,
                        reset_threshold=args.reset_threshold,
                        execute_steps=args.execute_steps,
                    )
                elif args.policy_input == "hummingbird_cached":
                    result = evaluate_policy_hummingbird(
                        env, h5_file, demo_key, episode_data, latent_payload,
                        model, videomae, config,
                        action_mean, action_std,
                        render=args.render,
                        video_writer=video_writer,
                        camera_name=args.camera,
                        hw=args.render_hw,
                        reset_mode=args.reset_mode,
                        reset_threshold=args.reset_threshold,
                        execute_steps=args.execute_steps,
                    )
                else:
                    result = evaluate_policy_hummingbird_online(
                        env, h5_file, demo_key, task_desc,
                        model, videomae, hummingbird_model, hummingbird_namespace,
                        hummingbird_reneg_path, config,
                        action_mean, action_std,
                        render=args.render,
                        video_writer=video_writer,
                        camera_name=args.camera,
                        hw=args.render_hw,
                        reset_mode=args.reset_mode,
                        reset_threshold=args.reset_threshold,
                        execute_steps=args.execute_steps,
                    )

                if video_writer is not None:
                    video_writer.close()

                status = "✓ SUCCESS" if result["success"] else "✗ FAIL"
                print(f"  {demo_key}: {status}  (steps: {result['num_steps']}, "
                      f"windows: {result['num_windows']}, resets: {result['num_resets']})")
                successes.append(result["success"])

            success_rate = sum(successes) / len(successes) * 100
            print(f"\n  Task {task_id} Results: {sum(successes)}/{len(successes)} = {success_rate:.1f}%")

            h5_file.close()
            env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
