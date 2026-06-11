"""
Extract a random first frame from MimicGen Parquet data and run LoRA inference.

Usage:
  python run_mimicgen_inference.py --data_dir ~/mimicgen_training_data --lora_path ./checkpoints/epoch-1
"""

import argparse
import glob
import io
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


def extract_random_frame_and_prompt(data_dir: str, meta_dir: str):
    """Pick a random episode, extract frame 0 and the task prompt."""
    data_path = Path(data_dir)
    parquet_files = sorted(glob.glob(str(data_path / "chunk-*" / "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(str(data_path / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Pick random episode
    chosen = random.choice(parquet_files)
    print(f"Selected episode: {Path(chosen).name}")

    df = pd.read_parquet(chosen)

    # Get first frame image
    img_data = df.iloc[0]["observation.image"]
    if isinstance(img_data, dict) and "bytes" in img_data:
        image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    elif isinstance(img_data, bytes):
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
    else:
        raise ValueError(f"Unknown image format: {type(img_data)}")

    # Get task prompt
    task_index = int(df.iloc[0].get("task_index", 0))
    tasks_file = os.path.join(meta_dir, "tasks.jsonl")
    prompt = "a robot arm performing a manipulation task"
    if os.path.exists(tasks_file):
        with open(tasks_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry["task_index"] == task_index:
                    prompt = entry["task"]
                    break

    return image, prompt


def main():
    parser = argparse.ArgumentParser(description="Run LoRA inference on MimicGen data")
    parser.add_argument("--data_dir", type=str, default="~/mimicgen_training_data/data")
    parser.add_argument("--meta_dir", type=str, default="~/mimicgen_training_data/meta")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output", type=str, default="mimicgen_prediction.mp4")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    meta_dir = os.path.expanduser(args.meta_dir)

    random.seed(args.seed)

    # Extract random first frame and prompt
    image, prompt = extract_random_frame_and_prompt(data_dir, meta_dir)
    print(f"Prompt: {prompt}")
    print(f"Image size: {image.size}")

    # Save first frame as temp image for inference_lora.py
    tmp_img = "/tmp/mimicgen_first_frame.png"
    image.save(tmp_img)
    print(f"Saved conditioning image to {tmp_img}")

    # Run inference
    from inference_lora import load_i2v_model, generate_future_frames
    from lora_utils import load_lora_weights
    from pytorch_lightning import seed_everything
    import torch
    import torchvision

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = Path(__file__).parent
    config_path = str(script_dir / "../configs/inference_i2v_512_v2.0_distil.yaml")
    ckpt_path = str(script_dir / "../hum_infer/checkpoints/stage_1.ckpt")
    unet_path = str(script_dir / "../hum_infer/checkpoints/unet.pt")
    img_proj_path = str(script_dir / "../hum_infer/checkpoints/img_proj.pt")
    reneg_path = str(script_dir / "../hum_infer/checkpoints/reneg_checkpoint.bin")

    print("\nLoading I2V model...")
    model = load_i2v_model(config_path, ckpt_path, unet_path, img_proj_path, device)

    print(f"Loading LoRA weights from {args.lora_path}...")
    if hasattr(model, 'model'):
        model.model = load_lora_weights(model.model, args.lora_path)
    else:
        model = load_lora_weights(model, args.lora_path)

    print(f"Generating future frames for: '{prompt}'")
    video = generate_future_frames(
        model=model,
        image=image,
        prompt=prompt,
        height=args.height,
        width=args.width,
        video_length=args.video_length,
        ddim_steps=16,
        unconditional_guidance_scale=7.5,
        device=device,
        reneg_path=reneg_path if os.path.exists(reneg_path) else None,
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    video_uint8 = (video * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
    torchvision.io.write_video(args.output, video_uint8, fps=8, video_codec='h264')

    # Also save the input frame alongside for comparison
    input_path = args.output.replace(".mp4", "_input.png")
    image.save(input_path)

    print(f"\nDone!")
    print(f"  Input frame:    {input_path}")
    print(f"  Predicted video: {args.output}")


if __name__ == "__main__":
    main()
