# Stage 4 — Visualization

Renders the image-to-video model's **imagined rollouts** so they can be compared
to ground truth. Useful for qualitatively checking that the fine-tuned video
model (stage 1) produces task-relevant future frames.

Run all commands from the repository root.

## Files

| File                  | Purpose                                                                |
|-----------------------|------------------------------------------------------------------------|
| `visualize_video_rollout.py` | autoregressive long rollout; saves ground-truth vs. imagined side-by-side videos |
| `visualize_stack_rollout.py` | stack-task visualization; one generation window, start-frame vs. imagined |

## Usage

```bash
python 4_visualization/visualize_video_rollout.py
python 4_visualization/visualize_stack_rollout.py
```

Both scripts need the i2v model and its LoRA checkpoints (`HUMMINGBIRD_I2V`) and
the MimicGen data (`MIMICGEN_DATA`). Videos are written to
`ablation_results/hb_videos/` and `ablation_results/hb_stack_videos/`.
