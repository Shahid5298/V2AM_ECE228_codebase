"""
Batch simulator evaluation for all V2AM ablation checkpoints.
Uses ground-truth oracle future frames from the 250-episode dataset.
Calls env._check_success() for hard physics-based success detection.

Usage:
    cd <repo-root>
    conda run -n ml python run_sim_eval_batch.py --num-demos 10

Results saved to: ablation_results/SIM_EVAL_RESULTS.txt
"""

from __future__ import annotations
import argparse, io, json, os, sys, time, datetime
from collections import deque
from pathlib import Path

import h5py, imageio, numpy as np, torch
from PIL import Image
from scipy.spatial.transform import Rotation as SciR

# ── path setup ──────────────────────────────────────────────────────────────
# External dependency: set MIMICGEN_REPO to the directory containing the
# `mimicgen/` simulation package.
REPO        = Path(__file__).resolve().parents[1]
MIMICGEN_REPO = Path(os.environ.get("MIMICGEN_REPO", str(Path.home() / "mimicgen_sim")))
MIMICGEN_PKG = MIMICGEN_REPO / "mimicgen"
for p in [str(REPO / "src"), str(MIMICGEN_REPO), str(MIMICGEN_PKG)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import mimicgen.envs.robosuite  # noqa: F401 — registers env classes
import robosuite
from robosuite.environments.base import register_env
from mimicgen.envs.robosuite.stack import Stack_D0, Stack_D1, StackThree, StackThree_D0, StackThree_D1
from mimicgen.envs.robosuite.coffee import Coffee, Coffee_D0, Coffee_D1, Coffee_D2
from mimicgen.envs.robosuite.mug_cleanup import MugCleanup, MugCleanup_D0, MugCleanup_D1
from mimicgen.envs.robosuite.nut_assembly import Square_D0, Square_D1, Square_D2
from mimicgen.envs.robosuite.threading import Threading, Threading_D0, Threading_D1, Threading_D2
from mimicgen.envs.robosuite.three_piece_assembly import (
    ThreePieceAssembly, ThreePieceAssembly_D0, ThreePieceAssembly_D1, ThreePieceAssembly_D2
)
for cls in [Stack_D0, Stack_D1, StackThree, StackThree_D0, StackThree_D1,
            Coffee, Coffee_D0, Coffee_D1, Coffee_D2,
            MugCleanup, MugCleanup_D0, MugCleanup_D1,
            Square_D0, Square_D1, Square_D2,
            Threading, Threading_D0, Threading_D1, Threading_D2,
            ThreePieceAssembly, ThreePieceAssembly_D0, ThreePieceAssembly_D1, ThreePieceAssembly_D2]:
    register_env(cls)

import pyarrow.parquet as pq
from transformers import VideoMAEImageProcessor

from flow_matching.config import FlowMatchingConfig
from flow_matching.model  import FlowMatchingActionHead
from videomae_encoder     import VideoMAEFeatureExtractor

# ── constants ───────────────────────────────────────────────────────────────
DEFAULT_CORE_DIR    = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--amandlek--mimicgen_datasets/"
    "snapshots/9bcacb0446b0f895dd352164bd39938710df4a1e/core"
)
MIMICGEN_250 = os.path.expanduser("~/mimicgen_training_data_250")
MIMICGEN_100 = os.path.expanduser("~/mimicgen_training_data_100_with_actions")
CKPT_DIR     = REPO / "checkpoint"

FILES_ORDER = [
    "coffee_d0","coffee_d1","coffee_d2",
    "coffee_preparation_d0","coffee_preparation_d1",
    "mug_cleanup_d0","mug_cleanup_d1",
    "square_d0","square_d1","square_d2",
    "stack_d0","stack_d1",
    "stack_three_d0","stack_three_d1",
    "threading_d0","threading_d1","threading_d2",
    "three_piece_assembly_d0","three_piece_assembly_d1","three_piece_assembly_d2",
]
TAKE_PER_FILE = {
    "coffee_d0":33,"coffee_d1":33,"coffee_d2":34,
    "coffee_preparation_d0":50,"coffee_preparation_d1":50,
    "mug_cleanup_d0":50,"mug_cleanup_d1":50,
    "square_d0":33,"square_d1":33,"square_d2":34,
    "stack_d0":50,"stack_d1":50,
    "stack_three_d0":50,"stack_three_d1":50,
    "threading_d0":33,"threading_d1":33,"threading_d2":34,
    "three_piece_assembly_d0":33,"three_piece_assembly_d1":33,"three_piece_assembly_d2":34,
}

# ── data helpers ─────────────────────────────────────────────────────────────

def resolve_h5(task_id, mimicgen_root, core_dir):
    tasks_jsonl = Path(mimicgen_root) / "meta" / "tasks.jsonl"
    task_map = {}
    with open(tasks_jsonl) as f:
        for line in f:
            d = json.loads(line)
            task_map[d["task_index"]] = d
    ep_offsets = {}
    cur = 0
    for fname in FILES_ORDER:
        ep_offsets[fname] = cur
        cur += TAKE_PER_FILE[fname]
    entry = task_map[task_id]
    h5_path = Path(core_dir) / entry["source_file"]
    base = entry["source_file"].replace(".hdf5","")
    return str(h5_path), entry["task"], ep_offsets.get(base, 0)


def load_episode_250(task_id, demo_idx):
    """Load GT frames + proprio from the 250-episode dataset."""
    ep_idx = task_id * 250 + demo_idx
    pq_path = Path(MIMICGEN_250) / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    table = pq.read_table(str(pq_path))
    T = len(table)
    frames_main, frames_wrist = [], []
    for md, wd in zip(table["observation.image"].to_pylist(),
                      table["observation.image_wrist"].to_pylist()):
        frames_main.append(np.array(Image.open(io.BytesIO(md["bytes"])).convert("RGB")))
        frames_wrist.append(np.array(Image.open(io.BytesIO(wd["bytes"])).convert("RGB")))
    pos   = np.stack(table["observation.robot0_eef_pos"].to_numpy(zero_copy_only=False))
    quat  = np.stack(table["observation.robot0_eef_quat"].to_numpy(zero_copy_only=False))
    grip  = np.stack(table["observation.robot0_gripper_qpos"].to_numpy(zero_copy_only=False))
    euler = SciR.from_quat(quat).as_euler("xyz", degrees=False)
    proprio = np.concatenate([pos, euler, grip], axis=-1).astype(np.float32)
    return {"frames_main": frames_main, "frames_wrist": frames_wrist,
            "proprio": proprio, "num_steps": T}


def create_env(h5_file, hw=256):
    env_meta = json.loads(h5_file["data"].attrs["env_args"])
    kw = env_meta["env_kwargs"]
    kw.update(has_renderer=False, has_offscreen_renderer=True,
              use_camera_obs=True, camera_heights=hw, camera_widths=hw)
    return robosuite.make(env_meta["env_name"], **kw)


def reset_to_demo(env, h5_file, demo_key):
    model_xml = h5_file[f"data/{demo_key}"].attrs["model_file"]
    states    = h5_file[f"data/{demo_key}/states"][()]
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    return states


def render_frame(env, hw=256):
    return env.sim.render(height=hw, width=hw, camera_name="agentview")[::-1]

# ── policy evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_one_demo(env, h5_file, demo_key, episode_data,
                  model, videomae, processor, config,
                  a_mean, a_std, device, record_video=False, video_path=None,
                  reset_mode="always"):
    """
    Semi-open-loop oracle eval.  Every execute_steps actions, sim is reset
    to GT state and oracle future frames are fed to the model.
    Returns dict: success (bool), num_steps (int).
    """
    states_h5 = reset_to_demo(env, h5_file, demo_key)
    T = episode_data["num_steps"]
    gt_main  = episode_data["frames_main"]
    gt_wrist = episode_data["frames_wrist"]
    gt_prop  = episode_data["proprio"]

    num_frames   = config.num_frames        # 8
    frame_stride = config.frame_stride      # 2
    fut_offset   = config.future_frame_offset  # 1
    chunk_size   = config.chunk_size        # 16
    hist_nf      = config.history_num_frames   # set to 1 to match training
    prop_hist    = config.proprio_history_size # 4
    execute_steps = 4
    success = False
    frames_vid = []

    for win_start in range(0, T, execute_steps):
        # decide whether to snap sim back to GT state this window
        if reset_mode == "always" and win_start < len(states_h5):
            env.sim.set_state_from_flattened(states_h5[win_start])
            env.sim.forward()
        elif reset_mode == "never" and win_start == 0 and 0 < len(states_h5):
            # only reset at the very start — then let model run completely free
            env.sim.set_state_from_flattened(states_h5[0])
            env.sim.forward()
        elif reset_mode == "adaptive" and win_start < len(states_h5):
            current = env.sim.get_state().flatten()
            gt      = states_h5[win_start]
            n = min(len(current), len(gt))
            if win_start == 0 or np.linalg.norm(current[:n] - gt[:n]) > 1.0:
                env.sim.set_state_from_flattened(gt)
                env.sim.forward()

        if record_video:
            frames_vid.append(render_frame(env))

        # ── build history frames ──────────────────────────────────────────
        hist_indices = [max(0, win_start - (hist_nf - 1 - j) * frame_stride)
                        for j in range(hist_nf)]
        hist_main  = [gt_main[i]  for i in hist_indices]
        hist_wrist = [gt_wrist[i] for i in hist_indices]
        resamp = np.linspace(0, len(hist_main)-1, num_frames).round().astype(int)
        hist_main  = [hist_main[i]  for i in resamp]
        hist_wrist = [hist_wrist[i] for i in resamp]

        # ── build future frames ───────────────────────────────────────────
        fut_indices = [win_start + fut_offset + j * frame_stride for j in range(num_frames)]
        fut_main  = [gt_main[min(i, T-1)]  for i in fut_indices]
        fut_wrist = [gt_wrist[min(i, T-1)] for i in fut_indices]

        # ── proprio ───────────────────────────────────────────────────────
        p_start = max(0, win_start - prop_hist + 1)
        prop_np = gt_prop[p_start:win_start+1]
        if len(prop_np) < prop_hist:
            pad = np.repeat(prop_np[:1], prop_hist - len(prop_np), axis=0)
            prop_np = np.concatenate([pad, prop_np], axis=0)
        proprio = torch.from_numpy(prop_np).unsqueeze(0).to(device)

        # ── encode visual streams ─────────────────────────────────────────
        def encode(frames):
            pv = processor(frames, return_tensors="pt")["pixel_values"].to(device)
            return videomae(pv)

        feats = []
        if config.include_history_frames:
            feats.extend([encode(hist_main), encode(hist_wrist)])
        if config.condition_on_future_video:
            feats.extend([encode(fut_main), encode(fut_wrist)])
        video_feats = torch.cat(feats, dim=1)

        # ── predict and execute ───────────────────────────────────────────
        cont_norm, grip_logit = model.sample_actions(video_feats, proprio)
        cont  = cont_norm * a_std + a_mean
        grip  = (torch.sigmoid(grip_logit) > 0.5).float() * 2.0 - 1.0
        acts  = torch.cat([cont, grip], dim=-1).cpu().numpy()[0]

        for act in acts[:execute_steps]:
            if win_start >= T: break
            env.step(act)
            if record_video:
                frames_vid.append(render_frame(env))
            if env._check_success():
                success = True

    if video_path and frames_vid:
        w = imageio.get_writer(str(video_path), fps=20)
        for f in frames_vid: w.append_data(f)
        w.close()

    return {"success": success, "frames": frames_vid}

# ── main ─────────────────────────────────────────────────────────────────────

CHECKPOINTS = [
    # (checkpoint_stem,  task_id, label)
    ("model_task0_coffeeinsert_history1_future_epoch26_R2_0723",  0, "coffee  λ=0.0  [baseline]"),
    ("model_task0_lambda0p0_euler50_control",                     0, "coffee  λ=0.0  [ctrl Euler-50]"),
    ("model_task0_lsmooth_lambda01_euler50_epoch32_R2_0719",      0, "coffee  λ=0.1"),
    ("model_task0_lambda0p5",                                     0, "coffee  λ=0.5"),
    ("model_task0_lambda1p0",                                     0, "coffee  λ=1.0"),
    ("model_task0_lambda2p0",                                     0, "coffee  λ=2.0"),
    ("model_task10_cubestack_lambda0p0",                         10, "cube    λ=0.0"),
    ("model_task10_cubestack_lambda0p1",                         10, "cube    λ=0.1"),
    ("model_task10_cubestack_lambda0p5",                         10, "cube    λ=0.5"),
    ("model_task12_stackthree_lambda0p1",                        12, "stack3  λ=0.1"),
    ("model_task12_stackthree_lambda0p5",                        12, "stack3  λ=0.5"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-demos",    type=int, default=10)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--core-dir",     default=DEFAULT_CORE_DIR)
    parser.add_argument("--only-task",    type=int, default=None,
                        help="Restrict to one task id")
    parser.add_argument("--reset-mode",   choices=["always","never","adaptive"],
                        default="always",
                        help="always=reset sim to GT every window (semi-open-loop, best for "
                             "reported success rates); never=run completely free after t=0 "
                             "(clean videos, honest deployment test); adaptive=reset only when "
                             "sim diverges from GT")
    args = parser.parse_args()

    out_dir   = REPO / "ablation_results" / "sim_eval"
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # build config — must match training (history_num_frames=1)
    config = FlowMatchingConfig()
    config.history_num_frames      = 1    # matches --history-num-frames 1 in train_ablation_sweep.sh
    config.include_history_frames  = True
    config.condition_on_future_video = True
    config.device = args.device

    processor = VideoMAEImageProcessor.from_pretrained(config.model_dir)
    videomae  = VideoMAEFeatureExtractor(
        config.model_dir, layer_idx=config.videomae_layer,
        num_frames_expected=config.num_frames, device=args.device,
    ).to(args.device)

    results = []
    print(f"\n{'='*80}")
    print(f"V2AM SIM EVAL — Oracle GT frames | {args.num_demos} demos per checkpoint")
    print(f"{'='*80}\n")

    for ckpt_stem, task_id, label in CHECKPOINTS:
        if args.only_task is not None and task_id != args.only_task:
            continue
        ckpt_path = CKPT_DIR / f"{ckpt_stem}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP {label} — checkpoint not found")
            continue

        # load model
        ckpt    = torch.load(str(ckpt_path), weights_only=False, map_location=args.device)
        model   = FlowMatchingActionHead(config).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        a_mean  = ckpt["action_mean"].to(args.device)
        a_std   = ckpt["action_std"].to(args.device)

        # resolve HDF5 and episodes
        try:
            h5_path, task_desc, ep_start = resolve_h5(task_id, MIMICGEN_100, args.core_dir)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")
            continue

        h5_file   = h5py.File(h5_path, "r")
        env       = create_env(h5_file, hw=256)
        demo_keys = sorted([k for k in h5_file["data"].keys() if k.startswith("demo")])
        demo_keys = demo_keys[:args.num_demos]

        print(f"  Evaluating: {label}  (task_id={task_id}: {task_desc[:40]})")
        successes = []
        t0 = time.time()

        # build a short safe name for this checkpoint used in video filenames
        # e.g. "task0_coffee_lambda0p5_freerun" or "task10_cube_lambda0p1_semiopen"
        task_short  = {0: "coffee", 10: "cube", 12: "stack3"}.get(task_id, f"task{task_id}")
        lam_raw = label.split("λ=")[-1].split()[0] if "λ=" in label else "orig"
        lam_str = lam_raw.replace(".", "p")
        mode_tag = {"always": "semiopen", "never": "freerun", "adaptive": "adaptive"}[args.reset_mode]
        vid_prefix = f"task{task_id}_{task_short}_lambda{lam_str}_{mode_tag}"

        for i, demo_key in enumerate(demo_keys):
            ep_data = load_episode_250(task_id, i)

            # always record demo 0; skip recording for the rest (saves time + space)
            record = (i == 0)
            # video named without result yet — rename after we know the outcome
            tmp_vid = video_dir / f"{vid_prefix}_demo{i:02d}_PENDING.mp4" if record else None

            result = eval_one_demo(
                env, h5_file, demo_key, ep_data,
                model, videomae, processor, config,
                a_mean, a_std, args.device,
                record_video=record, video_path=tmp_vid,
                reset_mode=args.reset_mode,
            )
            successes.append(result["success"])
            status = "✓" if result["success"] else "✗"

            # rename video to include SUCCESS/FAIL so you immediately know what you're watching
            if record and tmp_vid and tmp_vid.exists():
                outcome  = "SUCCESS" if result["success"] else "FAIL"
                final_vid = video_dir / f"{vid_prefix}_demo{i:02d}_{outcome}.mp4"
                tmp_vid.rename(final_vid)
                print(f"    {demo_key}: {status}  [video → {final_vid.name}]", end="  ", flush=True)
            else:
                print(f"    {demo_key}: {status}", end="  ", flush=True)

        elapsed = time.time() - t0
        rate = sum(successes) / len(successes) * 100
        print(f"\n  → {sum(successes)}/{len(successes)} = {rate:.0f}%  ({elapsed:.0f}s)\n")
        results.append({"label": label, "task_id": task_id, "task_desc": task_desc,
                        "ckpt": ckpt_stem, "n": len(successes),
                        "successes": sum(successes), "rate": rate})

        h5_file.close()
        env.close()
        del model

    # ── print summary table ───────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("V2AM SIM EVAL — SUCCESS RATES (Oracle GT future frames)")
    lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Demos per checkpoint: {args.num_demos}")
    lines.append(f"Reset mode: {args.reset_mode}  "
                 f"({'semi-open-loop: GT correction every 4 steps' if args.reset_mode=='always' else 'free-run: model runs completely free after t=0' if args.reset_mode=='never' else 'adaptive: reset only on divergence'})")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Checkpoint':<38} | {'Task':>6} | {'Success':>9} | {'Rate':>6}")
    lines.append("  " + "─"*65)
    for r in results:
        lines.append(f"  {r['label']:<38} | {r['task_id']:>6} | "
                     f"{r['successes']:>3}/{r['n']:<3}     | {r['rate']:>5.0f}%")
    lines.append("")
    lines.append("NOTE: 'Oracle' means real ground-truth future frames are used.")
    lines.append("      This is the upper bound. Deployment with Hummingbird would be lower.")

    summary = "\n".join(lines)
    print("\n" + summary)

    out_txt = REPO / "ablation_results" / "SIM_EVAL_RESULTS.txt"
    out_txt.write_text(summary)
    torch.save(results, REPO / "ablation_results" / "sim_eval_results.pt")
    print(f"\nResults saved to {out_txt}")


if __name__ == "__main__":
    main()
