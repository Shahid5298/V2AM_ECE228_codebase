# Stage 2 — Action-head training

Trains the **flow-matching action head**. A frozen VideoMAE encoder
(`src/videomae_encoder.py`, layer-6 features) turns the conditioning frames
(current + history + imagined future video) into tokens, and a conditional
flow-matching transformer (`src/flow_matching/`) decodes them into a 16-step
action chunk — 6-D pose via flow matching, 1-D gripper via BCE. An optional
**L_smooth jerk penalty** (weight λ) regularizes the trajectory.

Run all commands from the repository root.

## Files

| File                          | Purpose                                                        |
|-------------------------------|----------------------------------------------------------------|
| `train_flow_matching_head.py` | main training entry point                                      |
| `train_ablation_sweep.sh`          | reproduces the ablation training sweep (λ sweep, tasks 0/10/12) |
| `train_ablation_sweep_extended.sh`         | round-2 sweep (best-λ task comparison + degradation point)     |
| `cache_hummingbird_latents.py`| pre-generate & cache imagined future frames for training       |
| `finetune_on_i2v.py`          | fine-tune the trained head on i2v-generated frames             |
| `evaluate_i2v.py`             | evaluate the i2v-conditioned head in the simulator             |
| `test_dataloader.py`          | sanity-check dataset shapes                                    |

## Usage

```bash
# Baseline: current frame + imagined future video
python 2_action_head_training/train_flow_matching_head.py \
       --task-id 0 --epochs 40 --history-num-frames 1

# Ablation: no future video
python 2_action_head_training/train_flow_matching_head.py \
       --task-id 0 --epochs 40 --history-num-frames 1 --no-future-video

# Physics-informed L_smooth jerk penalty (λ = 0.5)
python 2_action_head_training/train_flow_matching_head.py \
       --task-id 0 --epochs 40 --history-num-frames 1 --smooth-loss-weight 0.5
```

Key flags: `--task-id` / `--task-ids`, `--smooth-loss-weight` (λ),
`--no-future-video`, `--no-history-frames`, `--history-num-frames`, `--epochs`.

Checkpoints, action-normalization stats, and training history are written under
`outputs/` (and copied to `checkpoint/` by the overnight scripts).
