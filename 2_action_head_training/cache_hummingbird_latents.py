"""Cache LoRA-conditioned Hummingbird future latents for MimicGen windows.

This script aligns cached latents with the existing flow-matching windowing:
each cached item corresponds to a MimicGen window start timestep `t`.
For each window, it uses the current image at `t` plus the task prompt to
generate a future latent video with Hummingbird, then saves those latents
grouped by source episode.

Output layout:
  <output_dir>/
    manifest.jsonl
    task_0/
      chunk-000/
        demo_000.pt

Each `.pt` file contains:
  {
    "latents": (N, C, T, H, W),
    "starts": (N,),
    "task_id": int,
    "prompt": str,
    "parquet_path": str,
    ...
  }
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torchvision.transforms as transforms
from PIL import Image
from einops import repeat
from omegaconf import OmegaConf
from tqdm import tqdm

# Make the repo root importable so that `src` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.flow_matching.config import FlowMatchingConfig
from src.mimicgen_dataset import MimicGenWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hummingbird-root", type=Path, required=True,
                        help="Path to /home/.../Hummingbird/i2v")
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="Optional LoRA adapter directory")
    parser.add_argument("--base-config", type=Path, default=None,
                        help="Optional Hummingbird config YAML")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional base stage_1 checkpoint")
    parser.add_argument("--unet-path", type=Path, default=None,
                        help="Optional UNet checkpoint")
    parser.add_argument("--img-proj-path", type=Path, default=None,
                        help="Optional image projection checkpoint")
    parser.add_argument("--reneg-path", type=Path, default=None,
                        help="Optional negative conditioning checkpoint")
    parser.add_argument("--mimicgen-root", type=Path,
                        default=Path(os.environ.get("MIMICGEN_DATA", str(Path.home() / "mimicgen_training_data"))))
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/hummingbird_latents"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--video-length", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--fps-condition", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--future-frame-offset", type=int, default=1)
    parser.add_argument("--image-column", type=str, default="observation.image",
                        choices=["observation.image", "observation.image_wrist"])
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-windows-per-episode", type=int, default=None)
    return parser.parse_args()


def resolve_path(base_root: Path, explicit: Path | None, relative_default: str) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (base_root / relative_default).resolve()


def add_hummingbird_paths(hummingbird_root: Path) -> None:
    sys.path.insert(0, str(hummingbird_root))
    sys.path.insert(0, str(hummingbird_root / "hum_infer"))


def load_module_attr(module_path: Path, module_name: str, attr_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def load_hummingbird_model(args: argparse.Namespace):
    add_hummingbird_paths(args.hummingbird_root)

    load_lora_weights = load_module_attr(
        args.hummingbird_root / "lora" / "lora_utils.py",
        "hummingbird_lora_utils",
        "load_lora_weights",
    )
    instantiate_from_config = load_module_attr(
        args.hummingbird_root / "hum_infer" / "utils" / "utils.py",
        "hummingbird_utils",
        "instantiate_from_config",
    )

    config_path = resolve_path(
        args.hummingbird_root, args.base_config, "configs/inference_i2v_512_v2.0_distil.yaml",
    )
    ckpt_path = resolve_path(
        args.hummingbird_root, args.checkpoint, "hum_infer/checkpoints/stage_1.ckpt",
    )
    unet_path = resolve_path(
        args.hummingbird_root, args.unet_path, "hum_infer/checkpoints/unet.pt",
    )
    img_proj_path = resolve_path(
        args.hummingbird_root, args.img_proj_path, "hum_infer/checkpoints/img_proj.pt",
    )
    reneg_path = resolve_path(
        args.hummingbird_root, args.reneg_path, "hum_infer/checkpoints/reneg_checkpoint.bin",
    )

    full_config = OmegaConf.load(str(config_path))
    model_config = full_config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False

    model = instantiate_from_config(model_config)

    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    if unet_path.exists():
        unet_sd = torch.load(str(unet_path), map_location="cpu")
        model.model.diffusion_model.load_state_dict(unet_sd, strict=False)

    if img_proj_path.exists():
        img_proj_sd = torch.load(str(img_proj_path), map_location="cpu")
        model.image_proj_model.load_state_dict(img_proj_sd, strict=False)

    if args.adapter_path is not None:
        adapter_path = args.adapter_path.expanduser().resolve()
        model.model = load_lora_weights(model.model, str(adapter_path))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, device, reneg_path


def decode_image(image_data) -> Image.Image:
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    raise ValueError(f"Unsupported image payload type: {type(image_data)}")


def build_window_index(args: argparse.Namespace) -> tuple[list[tuple[str, int, int]], dict[int, str]]:
    cfg = FlowMatchingConfig(
        mimicgen_data_root=args.mimicgen_root,
        task_filter_indices=tuple(sorted(set(args.task_ids))),
        num_frames=args.video_length,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
        chunk_size=args.chunk_size,
        future_frame_offset=args.future_frame_offset,
    )
    dataset = MimicGenWindowDataset(cfg)

    tasks_path = args.mimicgen_root / "meta" / "tasks.jsonl"
    task_prompts: dict[int, str] = {}
    with open(tasks_path, "r") as f:
        for line in f:
            row = json.loads(line)
            task_prompts[row["task_index"]] = row["task"]

    return dataset.index, task_prompts


def make_latent(
    model,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    reneg_path: Path,
) -> torch.Tensor:
    from lvdm.models.samplers.ddim import DDIMSampler

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.CenterCrop((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ddim_sampler = DDIMSampler(model)
    frame = transform(image).unsqueeze(0).to(device)
    videos = frame.unsqueeze(2).repeat(1, 1, args.video_length, 1, 1)

    img = videos[:, :, 0]
    img_emb = model.embedder(img)
    img_emb = model.image_proj_model(img_emb)
    cond_emb = model.get_learned_conditioning([prompt])
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    if model.model.conditioning_key == "hybrid":
        z = get_latent_z(model, videos)
        img_cat_cond = z[:, :, :1, :, :]
        img_cat_cond = repeat(img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]

    uc = None
    if args.guidance_scale != 1.0:
        if reneg_path.exists():
            uc_emb = torch.load(str(reneg_path), map_location=device).to(device)
        else:
            uc_emb = model.get_learned_conditioning([""])
        uc_img_emb = model.embedder(torch.zeros_like(img))
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uc["c_concat"] = [img_cat_cond]

    noise_shape = [1, 4, args.video_length, args.height // 8, args.width // 8]
    fs_tensor = torch.tensor([args.fps_condition], dtype=torch.long, device=device)

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
        samples, *_ = ddim_sampler.sample(
            S=args.ddim_steps,
            conditioning=cond,
            batch_size=1,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=args.guidance_scale,
            unconditional_conditioning=uc,
            eta=1.0,
            fs=fs_tensor,
            timestep_spacing="uniform_trailing",
            guidance_rescale=0.7,
        )

    return samples.detach().cpu()


def get_latent_z(model, videos: torch.Tensor) -> torch.Tensor:
    b, _, t, _, _ = videos.shape
    x = videos.permute(0, 2, 1, 3, 4).reshape(b * t, videos.shape[1], videos.shape[3], videos.shape[4])
    z = model.encode_first_stage(x)
    return z.reshape(b, t, z.shape[1], z.shape[2], z.shape[3]).permute(0, 2, 1, 3, 4)


def save_episode_cache(
    cache_path: Path,
    latents: list[torch.Tensor],
    starts: list[int],
    task_id: int,
    prompt: str,
    parquet_path: str,
    args: argparse.Namespace,
) -> None:
    dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    latent_tensor = torch.cat(latents, dim=0).to(dtype)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": latent_tensor,
        "starts": torch.tensor(starts, dtype=torch.long),
        "task_id": task_id,
        "prompt": prompt,
        "parquet_path": parquet_path,
        "image_column": args.image_column,
        "video_length": args.video_length,
        "ddim_steps": args.ddim_steps,
        "guidance_scale": args.guidance_scale,
        "latent_kind": "hummingbird_generated_diffusion_latent",
    }, cache_path)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    model, device, reneg_path = load_hummingbird_model(args)
    index, task_prompts = build_window_index(args)

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for parquet_path, start, task_id in index:
        grouped[parquet_path].append((start, task_id))

    manifest_path = args.output_dir / "manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episode_items = sorted(grouped.items())
    if args.limit_episodes is not None:
        episode_items = episode_items[:args.limit_episodes]

    manifest_mode = "w" if args.overwrite else "a"
    with open(manifest_path, manifest_mode, encoding="utf-8") as manifest:
        for episode_idx, (parquet_path, start_task_pairs) in enumerate(tqdm(episode_items, desc="Episodes", unit="ep")):
            task_id = start_task_pairs[0][1]
            prompt = task_prompts[task_id]
            rel = Path(parquet_path).relative_to(args.mimicgen_root / "data")
            cache_path = args.output_dir / f"task_{task_id}" / rel.with_suffix(".pt")

            if cache_path.exists() and not args.overwrite:
                continue

            starts = [start for start, _ in start_task_pairs]
            if args.limit_windows_per_episode is not None:
                starts = starts[:args.limit_windows_per_episode]

            table = pq.read_table(parquet_path, columns=[args.image_column])
            image_rows = table[args.image_column].to_pylist()

            latents: list[torch.Tensor] = []
            for local_idx, start in enumerate(tqdm(starts, desc=rel.name, leave=False, unit="win")):
                seed = args.seed + episode_idx * 100000 + local_idx
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                image = decode_image(image_rows[start])
                latents.append(make_latent(model, device, image, prompt, args, reneg_path))

            save_episode_cache(cache_path, latents, starts, task_id, prompt, parquet_path, args)
            manifest.write(json.dumps({
                "task_id": task_id,
                "prompt": prompt,
                "parquet_path": parquet_path,
                "cache_path": str(cache_path),
                "num_windows": len(starts),
                "latent_shape": list(torch.cat(latents, dim=0).shape),
            }) + "\n")
            manifest.flush()


if __name__ == "__main__":
    main()
