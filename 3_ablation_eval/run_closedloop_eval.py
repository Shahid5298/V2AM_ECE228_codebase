"""
Closed-loop evaluation: Hummingbird generates future frames from the ACTUAL
current camera view at every step. No ground-truth corrections, no teleporting.
The robot runs completely free from initial state.

This is the honest deployment evaluation. The video will be smooth because
there are no sim resets. Success rate reflects real task performance.

Domain gap note: our L_smooth checkpoints were trained on GT frames. Hummingbird
frames look different (slightly blurry, generated artifacts). Running them
zero-shot tests whether L_smooth regularization improves domain robustness.

Usage (inside tmux — takes ~30 min per checkpoint due to Hummingbird inference):
    cd <repo-root>
    conda run -n ml python run_closedloop_eval.py --num-demos 5

Results: ablation_results/CLOSEDLOOP_RESULTS.txt
Videos:  ablation_results/closedloop_videos/
"""

from __future__ import annotations

import argparse, collections, datetime, io, json, os, sys, time
from pathlib import Path

import h5py, imageio, numpy as np, torch
from PIL import Image
from scipy.spatial.transform import Rotation as SciR
from transformers import VideoMAEImageProcessor

# ── path setup ───────────────────────────────────────────────────────────────
# External dependencies (not part of this repo). Point these env vars at:
#   MIMICGEN_REPO  : directory containing the `mimicgen/` simulation package
#   HUMMINGBIRD_I2V: the Hummingbird i2v tree (its `lora/` holds the LoRA scripts)
REPO         = Path(__file__).resolve().parents[1]
MIMICGEN_REPO = Path(os.environ.get("MIMICGEN_REPO", str(Path.home() / "mimicgen_sim")))
HB_ROOT      = Path(os.environ.get("HUMMINGBIRD_I2V", str(Path.home() / "Hummingbird" / "i2v")))
LORA_DIR     = HB_ROOT / "lora"
MIMICGEN_PKG = MIMICGEN_REPO / "mimicgen"

for p in [str(REPO / "src"), str(MIMICGEN_REPO), str(MIMICGEN_PKG),
          str(HB_ROOT), str(LORA_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import mimicgen.envs.robosuite  # noqa — registers env classes
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
from flow_matching.config import FlowMatchingConfig
from flow_matching.model  import FlowMatchingActionHead
from videomae_encoder     import VideoMAEFeatureExtractor

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_CORE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--amandlek--mimicgen_datasets/"
    "snapshots/9bcacb0446b0f895dd352164bd39938710df4a1e/core"
)
MIMICGEN_100 = os.path.expanduser("~/mimicgen_training_data_100_with_actions")
CKPT_DIR     = REPO / "checkpoint"
LORA_CKPT    = str(LORA_DIR / "checkpoints_mimicgen" / "checkpoint-16000")

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

# ── Hummingbird ───────────────────────────────────────────────────────────────

def load_hummingbird(device: str, lora_path: str = LORA_CKPT):
    from inference_lora import load_i2v_model
    from lora_utils     import load_lora_weights

    cfg  = str(HB_ROOT / "configs/inference_i2v_512_v2.0_distil.yaml")
    ckpt = str(HB_ROOT / "hum_infer/checkpoints/stage_1.ckpt")
    unet = str(HB_ROOT / "hum_infer/checkpoints/unet.pt")
    iprj = str(HB_ROOT / "hum_infer/checkpoints/img_proj.pt")
    reneg_s = str(HB_ROOT / "hum_infer/checkpoints/reneg_checkpoint.bin")

    model = load_i2v_model(cfg, ckpt, unet, iprj, device)
    if hasattr(model, "model"):
        model.model = load_lora_weights(model.model, lora_path)
    else:
        model = load_lora_weights(model, lora_path)

    reneg = reneg_s if Path(reneg_s).exists() else None
    return model, reneg


@torch.no_grad()
def hummingbird_generate(hb_model, reneg, current_frame_np: np.ndarray,
                         prompt: str, device: str, num_frames: int = 8) -> np.ndarray:
    """Given the current actual camera frame (H,W,3 uint8), generate num_frames future frames."""
    from inference_lora import generate_future_frames

    pil_img = Image.fromarray(current_frame_np).resize((256, 256))
    video = generate_future_frames(
        model=hb_model,
        image=pil_img,
        prompt=prompt,
        height=256, width=256,
        video_length=num_frames,
        ddim_steps=16,
        unconditional_guidance_scale=7.5,
        device=device,
        reneg_path=reneg,
    )
    # video: (T, C, H, W) float32 [0,1] → (T, H, W, 3) uint8
    return (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

# ── sim helpers ───────────────────────────────────────────────────────────────

def resolve_h5(task_id, mimicgen_root, core_dir):
    tasks_jsonl = Path(mimicgen_root) / "meta" / "tasks.jsonl"
    task_map = {}
    with open(tasks_jsonl) as f:
        for line in f:
            d = json.loads(line)
            task_map[d["task_index"]] = d
    ep_offsets = {}; cur = 0
    for fname in FILES_ORDER:
        ep_offsets[fname] = cur; cur += TAKE_PER_FILE[fname]
    entry = task_map[task_id]
    h5_path = Path(core_dir) / entry["source_file"]
    base = entry["source_file"].replace(".hdf5","")
    return str(h5_path), entry["task"], ep_offsets.get(base, 0)


def create_env(h5_file, hw=256):
    meta = json.loads(h5_file["data"].attrs["env_args"])
    kw = meta["env_kwargs"]
    kw.update(has_renderer=False, has_offscreen_renderer=True,
              use_camera_obs=True, camera_heights=hw, camera_widths=hw)
    return robosuite.make(meta["env_name"], **kw)


def reset_to_demo(env, h5_file, demo_key):
    xml    = h5_file[f"data/{demo_key}"].attrs["model_file"]
    states = h5_file[f"data/{demo_key}/states"][()]
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    return states


def render_both(env, hw=256):
    """Render agentview and wrist cameras from current sim state."""
    main  = env.sim.render(height=hw, width=hw, camera_name="agentview")[::-1]
    wrist = env.sim.render(height=hw, width=hw, camera_name="robot0_eye_in_hand")[::-1]
    return main, wrist


def get_proprio(obs):
    pos   = np.asarray(obs["robot0_eef_pos"],       dtype=np.float32)
    quat  = np.asarray(obs["robot0_eef_quat"],      dtype=np.float32)
    grip  = np.asarray(obs["robot0_gripper_qpos"],  dtype=np.float32)
    euler = SciR.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)
    return np.concatenate([pos, euler, grip])

# ── closed-loop eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_closed_loop(env, h5_file, demo_key, prompt,
                     model, videomae, hb_model, hb_reneg, processor,
                     config, a_mean, a_std, device,
                     execute_steps=4, record_video=False, video_path=None):
    """
    TRUE closed loop:
      1. Render actual current frame from sim
      2. Hummingbird generates future frames from that real frame
      3. VideoMAE encodes them → action head predicts 16 actions
      4. Execute first execute_steps actions
      5. No GT corrections anywhere. Robot runs free.

    Returns: {"success": bool, "steps": int}
    """
    states_h5 = reset_to_demo(env, h5_file, demo_key)
    T = len(states_h5) - 1
    obs = env._get_observations(force_update=True)

    prop_hist_size = config.proprio_history_size  # 4
    hist_size      = config.history_num_frames      # 1

    # rolling buffers filled with real rendered frames and proprio
    hist_main  = collections.deque(maxlen=hist_size)
    hist_wrist = collections.deque(maxlen=hist_size)
    prop_buf   = collections.deque(maxlen=prop_hist_size)

    # seed buffers with the initial rendered frame
    init_main, init_wrist = render_both(env)
    init_prop = get_proprio(obs)
    for _ in range(hist_size):
        hist_main.append(init_main.copy())
        hist_wrist.append(init_wrist.copy())
    for _ in range(prop_hist_size):
        prop_buf.append(init_prop.copy())

    num_frames   = config.num_frames   # 8
    success      = False
    frames_vid   = []
    step_count   = 0

    while step_count < T:
        # ── render current actual frame ──────────────────────────────────
        cur_main, cur_wrist = render_both(env)
        if record_video:
            frames_vid.append(cur_main)

        # ── Hummingbird: generate 8 future frames from current real frame ──
        fut_frames = hummingbird_generate(
            hb_model, hb_reneg, cur_main, prompt, device, num_frames=num_frames
        )  # (8, H, W, 3) uint8 — hallucinated future

        # ── history: resample rolling buffer to num_frames ───────────────
        hist_list = list(hist_main)
        resamp = np.linspace(0, len(hist_list)-1, num_frames).round().astype(int)
        hist_main_8  = [hist_list[j] for j in resamp]
        hist_wrist_8 = [list(hist_wrist)[j] for j in resamp]

        # ── proprio: most recent history ────────────────────────────────
        prop_np = np.stack(list(prop_buf))   # (prop_hist_size, 8)
        proprio = torch.from_numpy(prop_np).unsqueeze(0).to(device)

        # ── encode: history streams + hallucinated future streams ────────
        def encode(frames_list):
            pv = processor(frames_list, return_tensors="pt")["pixel_values"].to(device)
            return videomae(pv)

        feats = []
        if config.include_history_frames:
            feats.extend([encode(hist_main_8), encode(hist_wrist_8)])
        if config.condition_on_future_video:
            # future main = Hummingbird hallucination (real, current-conditioned)
            # future wrist = we don't have a wrist hallucinator; use current wrist repeated
            wrist_repeated = [cur_wrist] * num_frames
            feats.extend([encode(list(fut_frames)), encode(wrist_repeated)])

        video_feats = torch.cat(feats, dim=1)

        # ── predict actions ───────────────────────────────────────────────
        cont_norm, grip_logit = model.sample_actions(video_feats, proprio)
        cont = cont_norm * a_std + a_mean
        grip = (torch.sigmoid(grip_logit) > 0.5).float() * 2.0 - 1.0
        acts = torch.cat([cont, grip], dim=-1).cpu().numpy()[0]  # (16, 7)

        # ── execute execute_steps actions ─────────────────────────────────
        for act in acts[:execute_steps]:
            if step_count >= T:
                break
            obs, _, _, _ = env.step(act)
            step_count += 1

            # update rolling buffers from actual new sim state
            new_main, new_wrist = render_both(env)
            hist_main.append(new_main)
            hist_wrist.append(new_wrist)
            prop_buf.append(get_proprio(obs))

            if record_video:
                frames_vid.append(new_main)
            if env._check_success():
                success = True

    if record_video and video_path and frames_vid:
        w = imageio.get_writer(str(video_path), fps=20)
        for f in frames_vid: w.append_data(f)
        w.close()

    return {"success": success, "steps": step_count}

# ── checkpoints to evaluate ───────────────────────────────────────────────────
# Fewer checkpoints for closed-loop since Hummingbird is slow (~16s per window)
CHECKPOINTS = [
    # (ckpt_stem, task_id, label)
    ("model_task0_coffeeinsert_history1_future_epoch26_R2_0723",  0, "coffee λ=0.0 [orig]"),
    ("model_task0_lambda0p1",                                      0, "coffee λ=0.1"),
    ("model_task0_lambda0p5",                                      0, "coffee λ=0.5"),
    ("model_task0_lambda1p0",                                      0, "coffee λ=1.0"),
    ("model_task10_cubestack_lambda0p0",                          10, "cube   λ=0.0"),
    ("model_task10_cubestack_lambda0p5",                          10, "cube   λ=0.5"),
]

# ── lambda0p1 was named differently in the original training ──────────────────
# check if it exists else fall back
def resolve_ckpt(stem):
    p = CKPT_DIR / f"{stem}.pt"
    # handle the renamed lsmooth checkpoint for task0 λ=0.1
    if not p.exists() and "lambda0p1" in stem and "task0" in stem:
        alt = CKPT_DIR / "model_task0_lsmooth_lambda01_euler50_epoch32_R2_0719.pt"
        if alt.exists():
            return alt
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-demos",     type=int, default=5)
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--core-dir",      default=DEFAULT_CORE_DIR)
    parser.add_argument("--execute-steps", type=int, default=4,
                        help="Actions to execute before replanning (4 = replan every 4 steps)")
    parser.add_argument("--only-task",     type=int, default=None)
    parser.add_argument("--only-first",    action="store_true",
                        help="Run only the first checkpoint (use for pipeline testing)")
    args = parser.parse_args()

    video_dir = REPO / "ablation_results" / "closedloop_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── config: match training ────────────────────────────────────────────
    config = FlowMatchingConfig()
    config.history_num_frames       = 1
    config.include_history_frames   = True
    config.condition_on_future_video = True
    config.device = args.device

    processor = VideoMAEImageProcessor.from_pretrained(config.model_dir)
    videomae  = VideoMAEFeatureExtractor(
        config.model_dir, layer_idx=config.videomae_layer,
        num_frames_expected=config.num_frames, device=args.device,
    ).to(args.device)

    # ── load Hummingbird once (expensive) ────────────────────────────────
    print("[Hummingbird] Loading LoRA model (this takes ~30s)...")
    hb_model, hb_reneg = load_hummingbird(args.device)
    print("[Hummingbird] Ready.\n")

    # ── load task prompts ─────────────────────────────────────────────────
    tasks_jsonl = Path(MIMICGEN_100) / "meta" / "tasks.jsonl"
    task_prompts = {}
    with open(tasks_jsonl) as f:
        for line in f:
            d = json.loads(line)
            task_prompts[d["task_index"]] = d["task"]

    results = []
    print(f"{'='*80}")
    print(f"V2AM CLOSED-LOOP EVAL — Hummingbird future frames, no GT corrections")
    print(f"Demos: {args.num_demos}  |  Execute steps: {args.execute_steps}")
    print(f"{'='*80}\n")

    checkpoints_to_run = CHECKPOINTS[:1] if args.only_first else CHECKPOINTS
    for ckpt_stem, task_id, label in checkpoints_to_run:
        if args.only_task is not None and task_id != args.only_task:
            continue

        ckpt_path = resolve_ckpt(ckpt_stem)
        if not ckpt_path.exists():
            print(f"  SKIP {label} — {ckpt_path} not found")
            continue

        # load action head
        ckpt  = torch.load(str(ckpt_path), weights_only=False, map_location=args.device)
        model = FlowMatchingActionHead(config).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        a_mean = ckpt["action_mean"].to(args.device)
        a_std  = ckpt["action_std"].to(args.device)

        try:
            h5_path, task_desc, _ = resolve_h5(task_id, MIMICGEN_100, args.core_dir)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}"); continue

        prompt    = task_prompts.get(task_id, "robot arm manipulating objects")
        h5_file   = h5py.File(h5_path, "r")
        env       = create_env(h5_file, hw=256)
        demo_keys = sorted([k for k in h5_file["data"].keys() if k.startswith("demo")])
        demo_keys = demo_keys[:args.num_demos]

        # video name: task_label_lambda_closedloop_demo00_SUCCESS/FAIL.mp4
        task_short = {0:"coffee", 10:"cube", 12:"stack3"}.get(task_id, f"task{task_id}")
        lam_str    = label.split("λ=")[-1].split()[0].replace(".","p") if "λ=" in label else "orig"
        vid_prefix = f"task{task_id}_{task_short}_lambda{lam_str}_closedloop"

        print(f"  {label}  [{task_desc[:45]}]")
        successes = []
        t0 = time.time()

        for i, demo_key in enumerate(demo_keys):
            tmp_path = video_dir / f"{vid_prefix}_demo{i:02d}_PENDING.mp4"
            result = eval_closed_loop(
                env, h5_file, demo_key, prompt,
                model, videomae, hb_model, hb_reneg, processor,
                config, a_mean, a_std, args.device,
                execute_steps=args.execute_steps,
                record_video=True, video_path=tmp_path,
            )
            successes.append(result["success"])
            outcome   = "SUCCESS" if result["success"] else "FAIL"
            final_vid = video_dir / f"{vid_prefix}_demo{i:02d}_{outcome}.mp4"
            if tmp_path.exists():
                tmp_path.rename(final_vid)
            print(f"    {demo_key}: {'✓' if result['success'] else '✗'}  "
                  f"[{final_vid.name}]")

        rate = sum(successes) / len(successes) * 100
        elapsed = time.time() - t0
        print(f"  → {sum(successes)}/{len(successes)} = {rate:.0f}%  ({elapsed:.0f}s)\n")
        results.append({"label": label, "task_id": task_id, "n": len(successes),
                        "successes": sum(successes), "rate": rate})

        h5_file.close(); env.close(); del model

    # ── summary ────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("V2AM CLOSED-LOOP EVAL — Hummingbird future frames, NO GT corrections")
    lines.append(f"Date:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Demos: {args.num_demos}  |  Execute steps per replan: {args.execute_steps}")
    lines.append(f"Hummingbird LoRA: {LORA_CKPT}")
    lines.append("")
    lines.append("What this means: at every replanning step, Hummingbird takes the ACTUAL")
    lines.append("current camera frame and generates 8 future frames. The action head then")
    lines.append("predicts actions from those hallucinated frames. No sim state corrections.")
    lines.append("Success = env._check_success() (physics check, not visual judgment).")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Checkpoint':<32} | {'Task':>4} | {'Success':>8} | {'Rate':>6}")
    lines.append("  " + "─"*55)
    for r in results:
        lines.append(f"  {r['label']:<32} | {r['task_id']:>4} | "
                     f"  {r['successes']}/{r['n']}     | {r['rate']:>5.0f}%")
    lines.append("")
    lines.append("NOTE: These models were trained on GT frames (domain gap present).")
    lines.append("A higher success rate with L_smooth vs λ=0.0 would indicate that")
    lines.append("physics regularization improves robustness to visual artifacts.")

    summary = "\n".join(lines)
    print("\n" + summary)

    out = REPO / "ablation_results" / "CLOSEDLOOP_RESULTS.txt"
    out.write_text(summary)
    torch.save(results, REPO / "ablation_results" / "closedloop_results.pt")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
