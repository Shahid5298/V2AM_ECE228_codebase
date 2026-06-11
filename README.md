# V2AM — Video-to-Action Model with Physics-Informed Flow Matching

V2AM is a robot imitation-learning system for manipulation. A frozen video
encoder turns camera observations into features; a **flow-matching action head**
predicts 16-step action chunks (6-D end-effector pose + 1-D gripper); and a
fine-tuned **image-to-video (i2v) model** supplies *imagined future frames* as an
extra conditioning stream. On top of this, V2AM adds three physics-informed
improvements and a full ablation study of them.

The model is trained and evaluated on [MimicGen](https://mimicgen.github.io/)
manipulation tasks (coffee-pod insertion, cube stacking, stack-three, …).

---

## The pipeline at a glance

```
                 ┌──────────────────────────────────────────────────────────┐
   MimicGen   →  │ STAGE 0  Data preparation (download + convert to Parquet) │ → demo data
   (HuggingFace) │          0_data_preparation/                              │
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
   raw camera →  │ STAGE 1  Video model (i2v) LoRA fine-tuning               │ → imagined
   frame +       │          1_video_finetuning/                              │   future frames
   task prompt   └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
   frames     →  │ STAGE 2  VideoMAE features  +  flow-matching action head  │ → 16-step
   + proprio     │          src/  +  2_action_head_training/                 │   action chunk
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │ STAGE 3  Ablation & evaluation                            │ → MSE / R² /
                 │          3_ablation_eval/                                  │   jerk / success
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │ STAGE 4  Visualization of imagined rollouts               │ → side-by-side
                 │          4_visualization/                                 │   videos
                 └──────────────────────────────────────────────────────────┘
```

### What each stage does

0. **Data preparation** (`0_data_preparation/`) — downloads the public MimicGen
   demonstrations (`amandlek/mimicgen_datasets` core, on HuggingFace) and converts
   the HDF5 files into the per-episode Parquet format the rest of the pipeline
   consumes.

1. **Video-model fine-tuning** (`1_video_finetuning/`) — LoRA-fine-tunes an
   image-to-video diffusion model on MimicGen demonstrations so that, given the
   current camera frame and a task prompt, it generates the next few frames of
   the task. These "imagined" future frames become a conditioning stream for the
   action head.

2. **Action-head training** (`src/` + `2_action_head_training/`) — a **frozen
   VideoMAE** encoder (`MCG-NJU/videomae-base-finetuned-ssv2`, layer-6 features)
   turns the conditioning frames into tokens, and a **conditional flow-matching
   transformer** decodes them into a 16-step action chunk. Continuous dimensions
   use flow matching; the gripper uses a BCE head. An optional **L_smooth jerk
   penalty** (weight λ) regularizes the predicted trajectory.

3. **Ablation & evaluation** (`3_ablation_eval/`) — evaluates the trained
   checkpoints under four inference modes (Euler, flow ensembling k=5, adaptive
   Neural-ODE `dopri5`, and `dopri5`+ensemble), sweeps the L_smooth strength λ,
   and runs closed-loop / simulator success evaluations.

4. **Visualization** (`4_visualization/`) — renders the i2v model's imagined
   rollouts side-by-side with ground truth.

The three V2AM contributions evaluated in stage 3 are: **flow ensembling**
(average several noise seeds for lower variance), the **adaptive Neural-ODE
solver**, and the **L_smooth jerk penalty** (smoother trajectories, with the
largest benefit on contact-rich tasks).

---

## Repository layout

```
.
├── README.md                  ← you are here
├── pyproject.toml             ← Python dependencies
├── src/                       ← shared library (imported by every stage)
│   ├── config.py              ← dataset/run configuration
│   ├── videomae_encoder.py    ← frozen VideoMAE feature extractor (layer 6)
│   ├── mimicgen_dataset.py    ← MimicGen sliding-window dataset
│   ├── utils.py               ← seeding, checkpoints, LR schedule
│   └── flow_matching/         ← flow-matching action head + train/eval
│
├── 0_data_preparation/        ← STAGE 0: download MimicGen + convert to Parquet
├── 1_video_finetuning/        ← STAGE 1: i2v LoRA fine-tuning on MimicGen
├── 2_action_head_training/    ← STAGE 2: train / fine-tune the action head
├── 3_ablation_eval/           ← STAGE 3: ablation & evaluation
├── 4_visualization/           ← STAGE 4: imagined-rollout videos
└── docs/                      ← model + experiment write-ups
```

Each stage folder has its own short `README.md` describing its scripts.

---

## Setup

```bash
# Python 3.10+; install dependencies (uv or pip)
pip install -e .          # uses pyproject.toml
```

This repo contains **code only**. The datasets and model weights live outside it
and are referenced via environment variables. **See [`docs/SETUP.md`](docs/SETUP.md)
for how to obtain each one and the exact folder layout each variable must point to.**

| Env var           | Points to                                               | Source                       |
|-------------------|---------------------------------------------------------|------------------------------|
| *(none)*          | VideoMAE encoder                                        | auto-downloads (HF)          |
| `MIMICGEN_REPO`   | directory containing the `mimicgen/` simulation package | public                       |
| `MIMICGEN_DATA`   | MimicGen demonstration parquet files                    | regenerate via stage 0       |
| `HUMMINGBIRD_I2V` | image-to-video model tree (base + fine-tuned LoRA)      | base public (AMD) + stage 1  |

```bash
export MIMICGEN_DATA=/path/to/mimicgen_data
export MIMICGEN_REPO=/path/to/mimicgen
export HUMMINGBIRD_I2V=/path/to/i2v
```

---

## How the model was trained & evaluated

All commands are run **from the repository root**.

### Stage 0 — prepare the data
```bash
python 0_data_preparation/download_datasets.py --dataset_type core --download_dir ./core
python 0_data_preparation/prepare_action_head_dataset.py \
       --core_dir ./core --output_dir "$MIMICGEN_DATA" --height 256 --width 256 --num_workers 4
```

### Stage 1 — fine-tune the video model
```bash
bash 1_video_finetuning/run_train.sh          # LoRA fine-tune i2v on MimicGen
python 1_video_finetuning/run_mimicgen_inference.py   # generate future frames
```

### Stage 2 — train the flow-matching action head
```bash
# Baseline (current frame + imagined future video), single task, 40 epochs:
python 2_action_head_training/train_flow_matching_head.py \
       --task-id 0 --epochs 40 --history-num-frames 1

# Ablation — current frame only (no future video):
python 2_action_head_training/train_flow_matching_head.py \
       --task-id 0 --epochs 40 --history-num-frames 1 --no-future-video

# Physics-informed L_smooth jerk penalty (λ = 0.5):
python 2_action_head_training/train_flow_matching_head.py \
       --task-id 0 --epochs 40 --history-num-frames 1 --smooth-loss-weight 0.5

# Reproduce the full ablation training sweep used in the report:
bash 2_action_head_training/train_ablation_sweep.sh
```

Optionally fine-tune the trained head on i2v-generated frames:
```bash
python 2_action_head_training/finetune_on_i2v.py --task-id 10 --epochs 20
```

### Stage 3 — ablation & evaluation
```bash
# Four inference modes on one checkpoint:
python 3_ablation_eval/run_ablation_eval.py

# Full batch ablation over all trained checkpoints:
bash 3_ablation_eval/batch_ablation_eval.sh

# Closed-loop / simulator success evaluation:
python 3_ablation_eval/run_closedloop_eval.py
python 3_ablation_eval/run_sim_eval_batch.py
```

### Stage 4 — visualize imagined rollouts
```bash
python 4_visualization/visualize_video_rollout.py
python 4_visualization/visualize_stack_rollout.py
```

---

## Model summary

| Component        | Detail                                                            |
|------------------|-------------------------------------------------------------------|
| Visual encoder   | VideoMAE ViT-Base (`videomae-base-finetuned-ssv2`), **frozen**, layer-6 features |
| Action head      | conditional flow-matching transformer (hidden 384, 4 layers, 6 heads, ~10.4M params) |
| Conditioning     | current frame + history frames + imagined future video + proprioception |
| Action space     | 16-step chunk: 6-D pose (flow matching) + 1-D gripper (BCE)        |
| Regularizer      | optional L_smooth jerk penalty, weight λ                           |
| Inference modes  | Euler · ensemble (k=5) · Neural-ODE `dopri5` · `dopri5`+ensemble   |

```