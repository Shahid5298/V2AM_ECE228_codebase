"""
Prepare MimicGen data for Hummingbird I2V LoRA training.

Renders high-res frames from all core MimicGen HDF5 files (excluding unused/)
and saves them in the same Parquet format used by LiberoI2VDataset.

Supports parallel processing across HDF5 files with --num_workers.

Output structure:
  output_dir/
    data/
      chunk-000/
        episode_000000.parquet   # each parquet = 1 episode
        episode_000001.parquet
        ...
    meta/
      tasks.jsonl                # task_index -> task description

Each Parquet file has columns:
  - observation.image: dict with 'bytes' key (PNG-encoded agentview image)
  - observation.image_wrist: dict with 'bytes' key (PNG-encoded robot0_eye_in_hand)
  - task_index: int

Usage:
  conda activate robosuite
  python prepare_video_model_dataset.py \\
    --core_dir ~/.cache/huggingface/hub/datasets--amandlek--mimicgen_datasets/snapshots/.../core \\
    --output_dir ~/mimicgen_training_data \\
    --height 256 --width 256 \\
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
import robosuite
from PIL import Image
from tqdm import tqdm

# Register mimicgen environments
try:
    import mimicgen
except ImportError:
    print("Warning: mimicgen not installed, some environments may not register.")


# Human-readable task descriptions for each MimicGen task
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
    """Infer task description from HDF5 filename."""
    name = filename.lower().replace(".hdf5", "")
    for suffix in ["_d0", "_d1", "_d2", "_d3", "_d4"]:
        name = name.replace(suffix, "")
    return TASK_DESCRIPTIONS.get(name, f"a robot arm performing the {name} task")


def encode_image_to_bytes(img_array: np.ndarray) -> bytes:
    """Encode a numpy RGB image to PNG bytes."""
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_single_hdf5(
    h5_path: str,
    task_index: int,
    output_data_dir: str,
    ep_start: int,
    height: int,
    width: int,
    max_episodes: int,
    cameras: list,
    result_queue: Queue = None,
):
    """
    Worker function: process one HDF5 file.
    Can run in its own process with its own MuJoCo context.
    """
    h5_path = Path(h5_path)
    output_data_dir = Path(output_data_dir)

    # Each process needs its own imports for MuJoCo
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

    print(f"  [Worker] Creating environment: {env_name} for {h5_path.name}")
    env = robosuite.make(env_name, **env_kwargs)

    demos = sorted(list(f["data"].keys()))
    if max_episodes:
        demos = demos[:max_episodes]

    ep_counter = ep_start
    for ep in tqdm(demos, desc=f"  {h5_path.stem}", leave=False):
        states = f[f"data/{ep}/states"][()]
        model_xml = f[f"data/{ep}"].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        rows = []
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            agentview = env.sim.render(height=height, width=width, camera_name=cameras[0])
            agentview = agentview[::-1]
            wrist = env.sim.render(height=height, width=width, camera_name=cameras[1])
            wrist = wrist[::-1]

            rows.append({
                "observation.image": {"bytes": encode_image_to_bytes(agentview)},
                "observation.image_wrist": {"bytes": encode_image_to_bytes(wrist)},
                "task_index": task_index,
            })

        df = pd.DataFrame(rows)
        out_path = output_data_dir / f"episode_{ep_counter:06d}.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        ep_counter += 1

    f.close()
    env.close()

    num_done = ep_counter - ep_start
    print(f"  [Worker] Done {h5_path.name}: {num_done} episodes (ep_{ep_start:06d} - ep_{ep_counter-1:06d})")
    if result_queue is not None:
        result_queue.put((h5_path.name, num_done))
    return ep_counter


def main(args):
    core_dir = Path(os.path.expanduser(args.core_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))

    hdf5_files = sorted([
        f for f in core_dir.glob("*.hdf5")
        if "unused" not in str(f)
    ])

    if not hdf5_files:
        print(f"No HDF5 files found in {core_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files to process:")
    for f in hdf5_files:
        print(f"  - {f.name}")

    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Write tasks.jsonl upfront
    tasks = []
    for task_index, h5_path in enumerate(hdf5_files):
        task_desc = get_task_description(h5_path.name)
        tasks.append({"task_index": task_index, "task": task_desc, "source_file": h5_path.name})

    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    # Pre-compute episode start offsets so parallel workers don't collide
    max_ep = args.max_episodes or 1000
    ep_starts = [i * max_ep for i in range(len(hdf5_files))]

    num_workers = min(args.num_workers, len(hdf5_files))

    if num_workers <= 1:
        # Sequential
        print(f"\nRunning sequentially...")
        total = 0
        for task_index, h5_path in enumerate(hdf5_files):
            print(f"\n[{task_index+1}/{len(hdf5_files)}] {h5_path.name}")
            process_single_hdf5(
                h5_path=str(h5_path),
                task_index=task_index,
                output_data_dir=str(data_dir),
                ep_start=ep_starts[task_index],
                height=args.height,
                width=args.width,
                max_episodes=args.max_episodes,
                cameras=args.cameras,
            )
            total += min(max_ep, 1000)
    else:
        # Parallel: launch workers in batches of num_workers
        print(f"\nRunning with {num_workers} parallel workers...")
        result_queue = Queue()

        for batch_start in range(0, len(hdf5_files), num_workers):
            batch = list(enumerate(hdf5_files))[batch_start:batch_start + num_workers]
            procs = []

            for task_index, h5_path in batch:
                p = Process(
                    target=process_single_hdf5,
                    args=(
                        str(h5_path), task_index, str(data_dir),
                        ep_starts[task_index],
                        args.height, args.width, args.max_episodes,
                        args.cameras, result_queue,
                    ),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            # Collect results
            while not result_queue.empty():
                name, count = result_queue.get()
                print(f"  Completed: {name} ({count} episodes)")

    total_eps = len(hdf5_files) * min(max_ep, 1000)
    print(f"\n{'='*60}")
    print(f"Done! Processed {len(hdf5_files)} tasks, ~{total_eps} total episodes.")
    print(f"Output: {data_dir}")
    print(f"Meta:   {tasks_path}")
    print(f"\nTo train: update config.yaml cache_dir to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MimicGen data for Hummingbird I2V LoRA")
    parser.add_argument("--core_dir", type=str, required=True,
                        help="Path to MimicGen core/ directory with HDF5 files")
    parser.add_argument("--output_dir", type=str, default="~/mimicgen_training_data",
                        help="Output directory for Parquet data")
    parser.add_argument("--height", type=int, default=256, help="Render height")
    parser.add_argument("--width", type=int, default=256, help="Render width")
    parser.add_argument("--max_episodes", type=int, default=250,
                        help="Max episodes per HDF5 file (default: 250)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (each gets its own MuJoCo context)")
    parser.add_argument("--cameras", nargs="+",
                        default=["agentview", "robot0_eye_in_hand"],
                        help="Camera views to render")
    args = parser.parse_args()
    main(args)
