# Stage 0 — Data preparation

Downloads the public **MimicGen** demonstrations and converts them into the
per-episode Parquet format that the rest of the pipeline consumes. Running this
stage reproduces the dataset from scratch; alternatively use the pre-built
download in [`../docs/SETUP.md`](../docs/SETUP.md).

Source dataset: **`amandlek/mimicgen_datasets`** (the `core/` split) on the
HuggingFace Hub — <https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core>.

These scripts import the `mimicgen` package (to render frames from the env), so
install MimicGen + robosuite first and run them from an environment where
`import mimicgen` works.

## Pipeline

```
amandlek/mimicgen_datasets/core  (HDF5 on HuggingFace)
        │   download_datasets.py
        ▼
   core/*.hdf5  (local)
        │   prepare_action_head_dataset.py   (images + proprio + actions)
        │   prepare_video_model_dataset.py      (images only, for i2v LoRA)
        ▼
   $MIMICGEN_DATA/
   ├── data/chunk-000/episode_000000.parquet …
   └── meta/tasks.jsonl
```

## Files

| File                               | Purpose                                                        |
|------------------------------------|----------------------------------------------------------------|
| `download_datasets.py`             | download the MimicGen `core` HDF5 datasets from HuggingFace     |
| `prepare_action_head_dataset.py` | render HDF5 → Parquet with agentview/wrist images, proprio **and** actions (this is the action-head training data) |
| `prepare_video_model_dataset.py`| render HDF5 → Parquet (images only) — the i2v LoRA training data |
| `add_actions_to_parquets.py`       | add actions/proprio to already-rendered Parquet files (avoids re-rendering) |

## Usage

```bash
# 1. download the core MimicGen datasets (HDF5)
python 0_data_preparation/download_datasets.py --dataset_type core --download_dir ./core

# 2. convert to the Parquet format used by the action head (images + proprio + actions)
python 0_data_preparation/prepare_action_head_dataset.py \
    --core_dir ./core \
    --output_dir "$MIMICGEN_DATA" \
    --height 256 --width 256 --num_workers 4

# (optional) render the image-only Parquets used for i2v LoRA training
python 0_data_preparation/prepare_video_model_dataset.py \
    --core_dir ./core --output_dir ./mimicgen_i2v_frames --num_workers 4
```

Each `episode_XXXXXX.parquet` holds per-step `observation.image` (PNG bytes),
`observation.image_wrist`, proprioception, the 7-D action, and `task_index`.
Point `MIMICGEN_DATA` at the `--output_dir` you chose.
