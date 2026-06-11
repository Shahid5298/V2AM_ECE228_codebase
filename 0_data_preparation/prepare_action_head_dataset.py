"""
Prepare a fresh 100-episode subset of MimicGen directly from the core HDF5 files.
This script embeds the rendered images, proprioceptive state, and actions into the parquets.

Usage:
  conda activate robosuite
  python prepare_action_head_dataset.py \
    --core_dir ~/.cache/huggingface/hub/datasets--amandlek--mimicgen_datasets/snapshots/.../core \
    --output_dir ~/mimicgen_training_data_100_unified \
    --height 256 --width 256 \
    --num_workers 4
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import mimicgen
except ImportError:
    pass

# HDF5 files in sorted order
FILES = [
    "coffee_d0", "coffee_d1", "coffee_d2",             # 0,1,2
    "coffee_preparation_d0", "coffee_preparation_d1",   # 3,4
    "mug_cleanup_d0", "mug_cleanup_d1",                 # 5,6
    "square_d0", "square_d1", "square_d2",              # 7,8,9
    "stack_d0", "stack_d1",                             # 10,11
    "stack_three_d0", "stack_three_d1",                 # 12,13
    "threading_d0", "threading_d1", "threading_d2",     # 14,15,16
    "three_piece_assembly_d0", "three_piece_assembly_d1", "three_piece_assembly_d2",  # 17,18,19
]

# How many episodes to take from each file to make exactly 800
TAKE = {
    # coffee: 33+33+34=100
    "coffee_d0": 33, "coffee_d1": 33, "coffee_d2": 34,
    # coffee_preparation: 50+50=100
    "coffee_preparation_d0": 50, "coffee_preparation_d1": 50,
    # mug_cleanup: 50+50=100
    "mug_cleanup_d0": 50, "mug_cleanup_d1": 50,
    # square: 33+33+34=100
    "square_d0": 33, "square_d1": 33, "square_d2": 34,
    # stack: 50+50=100
    "stack_d0": 50, "stack_d1": 50,
    # stack_three: 50+50=100
    "stack_three_d0": 50, "stack_three_d1": 50,
    # threading: 33+33+34=100
    "threading_d0": 33, "threading_d1": 33, "threading_d2": 34,
    # three_piece_assembly: 33+33+34=100
    "three_piece_assembly_d0": 33, "three_piece_assembly_d1": 33, "three_piece_assembly_d2": 34,
}

TASK_DESCRIPTIONS = {
    "coffee":              "a robot arm picking up a coffee pod and placing it into a coffee machine",
    "coffee_preparation":  "a robot arm preparing coffee by placing a mug and inserting a pod into the machine",
    "mug_cleanup":         "a robot arm cleaning up a mug from a table",
    "square":              "a robot arm picking up a square nut and placing it on a peg",
    "stack":               "a robot arm stacking a red cube on top of a green cube",
    "stack_three":         "a robot arm stacking three cubes on top of each other",
    "threading":           "a robot arm threading a needle through a hole",
    "three_piece_assembly":"a robot arm assembling three pieces together on pegs",
}

def get_task_description(filename: str) -> str:
    name = filename.lower().replace(".hdf5", "")
    for suffix in ["_d0", "_d1", "_d2", "_d3", "_d4"]:
        name = name.replace(suffix, "")
    return TASK_DESCRIPTIONS.get(name, f"a robot arm performing the {name} task")


def encode_image_to_bytes(img_array: np.ndarray) -> bytes:
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_single_hdf5(
    h5_path: str,
    task_index: int,
    output_data_dir: str,
    ep_start: int,
    take_n: int,
    height: int,
    width: int,
    cameras: list,
    result_queue: Queue = None,
):
    h5_path = Path(h5_path)
    output_data_dir = Path(output_data_dir)

    import robosuite
    try:
        import mimicgen
    except ImportError:
        pass

    f = h5py.File(h5_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    env_name = env_meta["env_name"]
    env_kwargs = env_meta["env_kwargs"]

    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_heights"] = height
    env_kwargs["camera_widths"] = width

    env = robosuite.make(env_name, **env_kwargs)

    # Mimicgen original renderer used alphabetical sorting for demos
    demos = sorted(list(f["data"].keys()))[:take_n]

    ep_counter = ep_start
    for ep in tqdm(demos, desc=f"  {h5_path.stem}", leave=False):
        states = f[f"data/{ep}/states"][()]
        actions = f[f"data/{ep}/actions"][()]
        
        obs_group = f[f"data/{ep}/obs"]
        eef_pos = obs_group["robot0_eef_pos"][()]
        eef_quat = obs_group["robot0_eef_quat"][()]
        gripper_qpos = obs_group["robot0_gripper_qpos"][()]
        joint_pos = obs_group["robot0_joint_pos"][()]
        proprio_state = np.concatenate([eef_pos, eef_quat, gripper_qpos, joint_pos], axis=1)

        model_xml = f[f"data/{ep}"].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        rows = []
        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            agentview = env.sim.render(height=height, width=width, camera_name=cameras[0])
            agentview = agentview[::-1]
            wrist = env.sim.render(height=height, width=width, camera_name=cameras[1])
            wrist = wrist[::-1]
            
            # Action and proprio data matching sequence lengths. Parquet lengths will match states lengths.
            action = actions[i] if i < len(actions) else np.zeros(actions.shape[1])
            p_state = proprio_state[i] if i < len(proprio_state) else np.zeros(proprio_state.shape[1])
            eef_p = eef_pos[i] if i < len(eef_pos) else np.zeros(eef_pos.shape[1])
            eef_q = eef_quat[i] if i < len(eef_quat) else np.zeros(eef_quat.shape[1])
            grip_q = gripper_qpos[i] if i < len(gripper_qpos) else np.zeros(gripper_qpos.shape[1])
            joint_p = joint_pos[i] if i < len(joint_pos) else np.zeros(joint_pos.shape[1])

            rows.append({
                "observation.image": {"bytes": encode_image_to_bytes(agentview)},
                "observation.image_wrist": {"bytes": encode_image_to_bytes(wrist)},
                "action": action,
                "observation.state": p_state,
                "observation.robot0_eef_pos": eef_p,
                "observation.robot0_eef_quat": eef_q,
                "observation.robot0_gripper_qpos": grip_q,
                "observation.robot0_joint_pos": joint_p,
                "task_index": task_index,
            })

        df = pd.DataFrame(rows)
        out_path = output_data_dir / f"episode_{ep_counter:06d}.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        ep_counter += 1

    f.close()
    env.close()

    num_done = ep_counter - ep_start
    if result_queue is not None:
        result_queue.put((h5_path.name, num_done))
    return ep_counter


def main(args):
    core_dir = Path(os.path.expanduser(args.core_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))

    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Write tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        # We preserve the original task indexing (there are 20 tasks, but 8 task groups)
        # Because we're reading sequentially through FILES, `task_index` maps to each HDF5 file
        for task_index, h5_name in enumerate(FILES):
            task_desc = get_task_description(h5_name)
            f.write(json.dumps({"task_index": task_index, "task": task_desc, "source_file": h5_name + ".hdf5"}) + "\n")

    # Pre-compute continuous episode indices (0 to 799) across tasks
    ep_starts = []
    current_ep = 0
    for h5_name in FILES:
        ep_starts.append(current_ep)
        current_ep += TAKE[h5_name]

    num_workers = min(args.num_workers, len(FILES))

    if num_workers <= 1:
        print(f"\nRunning sequentially...")
        for task_index, h5_name in enumerate(FILES):
            print(f"\n[{task_index+1}/{len(FILES)}] {h5_name}")
            h5_path = core_dir / (h5_name + ".hdf5")
            take_n = TAKE[h5_name]
            process_single_hdf5(
                str(h5_path), task_index, str(data_dir),
                ep_starts[task_index], take_n,
                args.height, args.width, args.cameras
            )
    else:
        print(f"\nRunning with {num_workers} parallel workers...")
        result_queue = Queue()

        for batch_start in range(0, len(FILES), num_workers):
            batch = list(enumerate(FILES))[batch_start:batch_start + num_workers]
            procs = []

            for task_index, h5_name in batch:
                h5_path = core_dir / (h5_name + ".hdf5")
                take_n = TAKE[h5_name]
                p = Process(
                    target=process_single_hdf5,
                    args=(
                        str(h5_path), task_index, str(data_dir),
                        ep_starts[task_index], take_n,
                        args.height, args.width, args.cameras, result_queue,
                    ),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            while not result_queue.empty():
                name, count = result_queue.get()
                print(f"  Completed: {name} ({count} episodes)")

    print(f"\n{'='*60}")
    print(f"Done! Generated unified 100 subset dataset at: {output_dir}")
    print(f"Total Episodes Processed: {current_ep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core_dir", type=str, required=True,
                        help="Path to MimicGen core/ directory with HDF5 files")
    parser.add_argument("--output_dir", type=str, default="~/mimicgen_training_data_100_new",
                        help="Output directory for Parquet data")
    parser.add_argument("--height", type=int, default=256, help="Render height")
    parser.add_argument("--width", type=int, default=256, help="Render width")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (each gets its own MuJoCo context)")
    parser.add_argument("--cameras", nargs="+",
                        default=["agentview", "robot0_eye_in_hand"],
                        help="Camera views to render")
    args = parser.parse_args()
    main(args)
