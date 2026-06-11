"""Evaluate the Flow Matching policy guided by I2V-generated frames via MimicGen playback.

This script runs a semi-open-loop evaluation mirroring `evaluate.py`.
At each replanning step (e.g. every `execute_steps` actions):
  1. Sim is optionally reset to the GT state (adaptive/always/never).
  2. The current GT agentview image is fed to the I2V LoRA model.
  3. I2V generates 8 future frames.
  4. These I2V frames + future GT wrist frames are passed to VideoMAE.
  5. The Flow Matching head predicts 16 consecutive actions.
  6. The robot executes `execute_steps` actions in the simulator.

The episode is rendered to an MP4 video (side-by-side with I2V debug).

Usage:
  python evaluate_i2v.py \\
      --task-ids 10 \\
      --flow-ckpt outputs/flow_matching_task_10_i2v/best.pt \\
      --lora-path $HUMMINGBIRD_I2V/lora/checkpoints_mimicgen_t10/checkpoint-3500 \\
      --num-demos 2 \\
      --render
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import h5py
import imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import VideoMAEImageProcessor

# ── locate the repo root so that `src` resolves ──────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# External dependency: set MIMICGEN_REPO to the directory containing `mimicgen/`.
MIMICGEN_PKG_ROOT = Path(os.environ.get("MIMICGEN_REPO", str(Path.home() / "mimicgen_sim")))
if str(MIMICGEN_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMICGEN_PKG_ROOT))

# ── stage-1 i2v LoRA scripts (1_video_finetuning/); override via HUMMINGBIRD_I2V ─
LORA_DIR    = Path(os.environ.get("HUMMINGBIRD_I2V", str(REPO_ROOT / "1_video_finetuning")))
HUMMINGBIRD = LORA_DIR
sys.path.insert(0, str(LORA_DIR))

import mimicgen.envs.robosuite
import robosuite
from robosuite.environments.base import register_env

# Explicitly register MimicGen custom environments
from mimicgen.envs.robosuite.stack import Stack_D0, Stack_D1, StackThree, StackThree_D0, StackThree_D1
for cls in [Stack_D0, Stack_D1, StackThree, StackThree_D0, StackThree_D1]:
    register_env(cls)

from src.flow_matching.config  import FlowMatchingConfig
from src.flow_matching.model   import FlowMatchingActionHead
from src.videomae_encoder      import VideoMAEFeatureExtractor
from src.utils                 import set_seed, load_checkpoint

# Re-use helpers from evaluate.py to ensure identical setup
from evaluate import (
    DEFAULT_CORE_DIR, DEFAULT_MIMICGEN_ROOT, DEFAULT_MIMICGEN_250,
    resolve_h5_paths_from_task_ids, create_env_from_h5,
    reset_env_to_demo, render_frame, load_parquet_episode
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. I2V Inference Logic
# ─────────────────────────────────────────────────────────────────────────────

def _load_i2v_model(lora_path: str, device: str, hummingbird_root: Path):
    from inference_lora import load_i2v_model
    from lora_utils    import load_lora_weights

    cfg_path      = str(hummingbird_root / "configs/inference_i2v_512_v2.0_distil.yaml")
    ckpt_path     = str(hummingbird_root / "hum_infer/checkpoints/stage_1.ckpt")
    unet_path     = str(hummingbird_root / "hum_infer/checkpoints/unet.pt")
    img_proj_path = str(hummingbird_root / "hum_infer/checkpoints/img_proj.pt")
    reneg_path_s  = str(hummingbird_root / "hum_infer/checkpoints/reneg_checkpoint.bin")

    model = load_i2v_model(cfg_path, ckpt_path, unet_path, img_proj_path, device)
    if hasattr(model, "model"):
        model.model = load_lora_weights(model.model, lora_path)
    else:
        model = load_lora_weights(model, lora_path)

    reneg_path = reneg_path_s if Path(reneg_path_s).exists() else None
    return model, reneg_path


@torch.no_grad()
def _run_i2v(i2v_model, reneg_path: str | None, pil_image: Image.Image,
             prompt: str, device: str,
             num_frames: int = 8, resolution: tuple[int, int] = (256, 256)
             ) -> np.ndarray:
    from inference_lora import generate_future_frames

    video = generate_future_frames(
        model=i2v_model,
        image=pil_image,
        prompt=prompt,
        height=resolution[0],
        width=resolution[1],
        video_length=num_frames,
        ddim_steps=16,
        unconditional_guidance_scale=7.5,
        device=device,
        reneg_path=reneg_path,
    )
    # Return (T, H, W, 3) uint8 directly
    frames_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return frames_np


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy_i2v(env, h5_file, demo_key, episode_data, 
                        model, videomae, i2v_model, reneg_path, prompt,
                        config, action_mean, action_std, args):
                        
    # ── 1. Init environment and states ──
    states_h5 = reset_env_to_demo(env, h5_file, demo_key)
    processor = VideoMAEImageProcessor.from_pretrained(str(config.model_dir))
    device = config.device
    a_mean = action_mean.to(device)
    a_std = action_std.to(device)

    num_frames = config.num_frames
    frame_stride = config.frame_stride
    future_offset = getattr(config, "future_frame_offset", 0)
    chunk_size = config.chunk_size
    execute_steps = min(args.execute_steps, chunk_size)

    gt_frames_main = episode_data['frames_main']   # (T, H, W, 3)
    gt_frames_wrist = episode_data['frames_wrist'] # (T, H, W, 3)
    gt_proprio = episode_data['proprio']           # (T, 8)
    T_total = episode_data['num_steps']
    gt_sim_states = states_h5

    success = False
    frames_for_video = []
    i2v_debug_frames = []
    step_count = 0
    num_windows = 0
    num_resets = 0

    # ── 2. Playback / Evaluation Loop ──
    for win_start in range(0, T_total, execute_steps):
        # ── Optional Sim Reset ──
        should_reset = False
        if args.reset_mode == "always":
            should_reset = True
        elif args.reset_mode == "never":
            should_reset = (win_start == 0)
        elif args.reset_mode == "adaptive":
            if win_start == 0:
                should_reset = True
            elif win_start < len(gt_sim_states):
                current_state = env.sim.get_state().flatten()
                gt_state = gt_sim_states[win_start]
                min_len = min(len(current_state), len(gt_state))
                state_dist = np.linalg.norm(current_state[:min_len] - gt_state[:min_len])
                should_reset = state_dist > args.reset_threshold
                if should_reset:
                    print(f"    [adaptive] Reset at step {win_start}: state_dist={state_dist:.4f}")

        if should_reset and win_start < len(gt_sim_states):
            env.sim.set_state_from_flattened(gt_sim_states[win_start])
            env.sim.forward()
            num_resets += 1

        if args.render:
            frame = render_frame(env, args.camera, args.render_hw, args.render_hw)
            frames_for_video.append(frame)

        # ── Generate I2V frames from the current agentview frame ──
        current_agentview = gt_frames_main[win_start]
        pil_agentview = Image.fromarray(current_agentview)

        # Output shape: (8, H, W, 3)
        i2v_frames_np = _run_i2v(i2v_model, reneg_path, pil_agentview, prompt, device, num_frames=num_frames)

        # ── Extract GT wrist frames (we do not have an I2V wrist model) ──
        wrist_indices = [win_start + future_offset + j * frame_stride for j in range(num_frames)]
        gt_wrist_window = [gt_frames_wrist[idx] for idx in wrist_indices if idx < T_total]
        while len(gt_wrist_window) < num_frames:
            gt_wrist_window.append(gt_wrist_window[-1])

        # Save I2V frames for debug video matching execution
        # i2v_frames_np has shape (num_frames, H, W, 3). It spans t .. t + frame_stride*(num_frames-1)
        for act_idx in range(execute_steps):
            frame_idx = min(act_idx // frame_stride, num_frames - 1)
            i2v_debug_frames.append(i2v_frames_np[frame_idx])

        # ── Get Proprio ──
        proprio_vec = gt_proprio[win_start]
        proprio = torch.from_numpy(proprio_vec).unsqueeze(0).unsqueeze(0)
        proprio = proprio.expand(1, config.proprio_history_size, -1).to(device)

        # ── Predict Actions ──
        with torch.no_grad():
            pv_main = processor(list(i2v_frames_np), return_tensors="pt")["pixel_values"].to(device)
            pv_wrist = processor(gt_wrist_window, return_tensors="pt")["pixel_values"].to(device)
            
            feats_main = videomae(pv_main)
            feats_wrist = videomae(pv_wrist)
            video_features = torch.cat([feats_main, feats_wrist], dim=1)

            pred_cont_norm, gripper_logit = model.sample_actions(video_features, proprio)
            pred_cont = pred_cont_norm * a_std + a_mean
            pred_gripper = (torch.sigmoid(gripper_logit) > 0.5).float() * 2.0 - 1.0
            pred_actions = torch.cat([pred_cont, pred_gripper], dim=-1).cpu().numpy()[0]

        # ── Execute Actions ──
        for act_idx, act in enumerate(pred_actions[:execute_steps]):
            if step_count >= T_total:
                break

            obs, reward, done, info = env.step(act)
            step_count += 1

            if args.render and (win_start + act_idx < T_total - 1): # Prevent double append on bounds
                frame = render_frame(env, args.camera, args.render_hw, args.render_hw)
                frames_for_video.append(frame)

            if env._check_success():
                success = True

        num_windows += 1

    return {
        "success": success,
        "num_steps": step_count,
        "num_windows": num_windows,
        "num_resets": num_resets,
        "frames_for_video": frames_for_video,
        "i2v_debug_frames": i2v_debug_frames,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate I2V + Flow Matching playback in MimicGen")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--flow-ckpt", type=str, default=None, help="Path to fine-tuned flow matching checkpoint. If omitted, uses base model.")
    parser.add_argument("--lora-path", type=str, required=True)
    
    parser.add_argument("--core-dir", type=str, default=DEFAULT_CORE_DIR)
    parser.add_argument("--mimicgen-root", type=str, default=DEFAULT_MIMICGEN_ROOT)
    parser.add_argument("--mimicgen-250", type=str, default=DEFAULT_MIMICGEN_250)
    
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/sim_eval_i2v")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--render-hw", type=int, default=256)
    parser.add_argument("--reset-mode", choices=["always", "never", "adaptive"], default="always")
    parser.add_argument("--reset-threshold", type=float, default=1.0)
    parser.add_argument("--execute-steps", type=int, default=16) 
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Config & Models ──
    config = FlowMatchingConfig()
    
    model = FlowMatchingActionHead(config).to(config.device)
    
    if args.flow_ckpt:
        print(f"\nLoading fine-tuned Flow Matching checkpoint: {args.flow_ckpt}")
        ckpt_path = args.flow_ckpt
    else:
        # Load the base model
        base_ckpt_dir = Path("outputs") / f"flow_matching_task_{args.task_ids[0]}"
        ckpt_path = base_ckpt_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Base checkpoint not found at {ckpt_path}. Please provide --flow-ckpt or ensure the base model is trained.")
        print(f"\nLoading BASE Flow Matching checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    action_mean = ckpt["action_mean"]
    action_std = ckpt["action_std"]

    print("Loading VideoMAE...")
    videomae = VideoMAEFeatureExtractor(
        config.model_dir, layer_idx=config.videomae_layer, 
        num_frames_expected=config.num_frames, device=config.device
    ).to(config.device)

    print(f"Loading I2V LoRA model from: {args.lora_path}")
    hummingbird_root = HUMMINGBIRD 
    i2v_model, reneg_path = _load_i2v_model(args.lora_path, config.device, hummingbird_root)

    # ── Data & Tasks ──
    h5_entries = resolve_h5_paths_from_task_ids(args.task_ids, args.mimicgen_root, args.core_dir)
    mimicgen_250_root = getattr(args, 'mimicgen_250', DEFAULT_MIMICGEN_250)

    for task_id, h5_path, task_desc, ep_start_idx in h5_entries:
        print(f"\n{'='*70}\nTask {task_id}: {task_desc}\n{'='*70}")
        
        h5_file = h5py.File(h5_path, "r")
        env = create_env_from_h5(h5_file, render_hw=args.render_hw)
        
        demo_keys = sorted([k for k in h5_file["data"].keys() if k.startswith("demo")])[:args.num_demos]
        successes = []

        for i, demo_key in enumerate(demo_keys):
            global_ep_idx = task_id * 250 + i
            print(f"  Evaluating {demo_key} (GT EP {global_ep_idx})...")
            
            episode_data = load_parquet_episode(mimicgen_250_root, global_ep_idx)

            result = evaluate_policy_i2v(
                env, h5_file, demo_key, episode_data, 
                model, videomae, i2v_model, reneg_path, task_desc,
                config, action_mean, action_std, args
            )

            status = "✓ SUCCESS" if result["success"] else "✗ FAIL"
            print(f"    {status}  (steps: {result['num_steps']}, windows: {result['num_windows']}, resets: {result['num_resets']})")
            successes.append(result["success"])

            # ── Rendering side-by-side ──
            if args.render and result["frames_for_video"]:
                vid_path = out_dir / f"task{task_id}_{demo_key}_i2v.mp4"
                
                # Combine sim output with GT frame we used to prompt I2V
                frames_sim = result["frames_for_video"]
                frames_gt = result["i2v_debug_frames"]
                
                # Make sure lengths match (they should based on logic, but pad if needed)
                min_len = min(len(frames_sim), len(frames_gt))
                
                combo_frames = []
                for s_idx in range(min_len):
                    sim_img = Image.fromarray(frames_sim[s_idx])
                    gt_img = Image.fromarray(frames_gt[s_idx]).resize(sim_img.size)
                    combo = np.concatenate([np.array(sim_img), np.array(gt_img)], axis=1)
                    combo_frames.append(combo)
                
                imageio.mimsave(str(vid_path), combo_frames, fps=20)
                print(f"    Saved video: {vid_path}")

        print(f"\n  Task {task_id} Success Rate: {sum(successes)}/{len(successes)} = {sum(successes)/len(successes)*100:.1f}%")
        h5_file.close()
        env.close()

    print("\nDone.")

if __name__ == "__main__":
    main()
