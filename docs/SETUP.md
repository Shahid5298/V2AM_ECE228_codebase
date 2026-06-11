# Setup — data & external models

This repository contains **code only**. Four external resources are needed to
run it — all are public or regenerable from the scripts in this repo. Set the
listed environment variables so the code can find each one.

| Resource                       | Env var           | How to obtain                              |
|--------------------------------|-------------------|--------------------------------------------|
| VideoMAE encoder               | *(none)*          | downloads automatically (HuggingFace)      |
| MimicGen simulation package    | `MIMICGEN_REPO`   | public — clone / install                   |
| MimicGen demonstration data    | `MIMICGEN_DATA`   | regenerate (stage 0)                       |
| Image-to-video (i2v) model     | `HUMMINGBIRD_I2V` | base public (AMD); fine-tuned LoRA via stage 1 |

```bash
export MIMICGEN_DATA=/path/where/you/extracted/mimicgen_data
export MIMICGEN_REPO=/path/to/mimicgen
export HUMMINGBIRD_I2V=/path/to/i2v
```

---

## 1. VideoMAE encoder — automatic

The frozen encoder is `MCG-NJU/videomae-base-finetuned-ssv2`. It is pulled from
the HuggingFace Hub the first time you run training/eval — no manual download and
no environment variable required.

## 2. MimicGen simulation package (`MIMICGEN_REPO`) — public

Only needed for the **simulator / closed-loop** evaluation scripts in
`3_ablation_eval/` (`run_closedloop_eval.py`, `run_sim_eval_batch.py`,
`evaluate_in_sim.py`). Install the MimicGen environments and robosuite (this repo
pins `robosuite==1.4.1`), then point `MIMICGEN_REPO` at the directory that
contains the importable `mimicgen/` package:

```
$MIMICGEN_REPO/
└── mimicgen/
    └── envs/robosuite/...        # Stack, Coffee, Threading, … env classes
```

## 3. MimicGen demonstration data (`MIMICGEN_DATA`)

Regenerate it from the public source using stage 0 (`0_data_preparation/`). The
data is built from `amandlek/mimicgen_datasets` (the `core` split) on HuggingFace —
<https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core>.
```bash
python 0_data_preparation/download_datasets.py --dataset_type core --download_dir ./core
python 0_data_preparation/prepare_action_head_dataset.py \
       --core_dir ./core --output_dir "$MIMICGEN_DATA" --height 256 --width 256 --num_workers 4
```

The directory must end up looking **exactly** like this (the loader globs
`data/chunk-*/*.parquet` and reads task names from `meta/tasks.jsonl`):

```
$MIMICGEN_DATA/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    └── tasks.jsonl
```

Each `episode_XXXXXX.parquet` holds per-step `observation.image` (encoded image
bytes), wrist image, proprioception, the 7-D action, and a `task_index` column.

## 4. Image-to-video model (`HUMMINGBIRD_I2V`)

This has a **public base model** plus the **fine-tuned LoRA adapters** from stage 1.

**Base i2v checkpoints — public (AMD Hummingbird):**
download `stage_1.ckpt`, `unet.pt`, `img_proj.pt` (and optional `SR.pth`) from
<https://huggingface.co/amd/AMD-Hummingbird-I2V>.

**Fine-tuned LoRA adapters — produced by stage 1:**
regenerate them by running the fine-tuning in `1_video_finetuning/` (see that
folder's README). They are written to `lora/checkpoints_mimicgen*/`.

Arrange everything so the tree looks like this:

```
$HUMMINGBIRD_I2V/
├── configs/
│   └── inference_i2v_512_v2.0_distil.yaml
├── hum_infer/
│   └── checkpoints/                            # base model (amd/AMD-Hummingbird-I2V)
│       ├── stage_1.ckpt
│       ├── unet.pt
│       ├── img_proj.pt
│       └── reneg_checkpoint.bin                # optional
└── lora/                                       # fine-tuned adapters (stage 1 output)
    ├── checkpoints_mimicgen/checkpoint-16000/    # general / coffee i2v LoRA
    ├── checkpoints_mimicgen_stack/epoch-9/       # cube-stacking i2v LoRA
    └── checkpoints_mimicgen_t10/checkpoint-3500/ # task-10 i2v LoRA
```

---

## Quick check

```bash
python - <<'PY'
import os
from pathlib import Path
for var, sub in [("MIMICGEN_DATA","data"), ("MIMICGEN_REPO","mimicgen"),
                 ("HUMMINGBIRD_I2V","hum_infer/checkpoints")]:
    p = Path(os.environ.get(var, "")) / sub
    print(f"{var:16s} -> {'OK' if p.exists() else 'MISSING'}  ({p})")
PY
```

Once all three report `OK` (VideoMAE downloads on demand), follow the commands in
the top-level `README.md` to train and evaluate.
