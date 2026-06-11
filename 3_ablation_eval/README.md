# Stage 3 — Ablation & evaluation

Evaluates the trained checkpoints and produces the numbers behind the report.
Each checkpoint is scored under four inference modes — **Euler**, **flow
ensembling (k=5)**, **adaptive Neural-ODE `dopri5`**, and **`dopri5`+ensemble** —
reporting MSE, R², gripper accuracy, per-dimension MSE, mean jerk, and
per-step variance. The L_smooth strength λ is swept, and closed-loop /
simulator success evaluations measure real task completion.

Run all commands from the repository root.

## Files

| File                      | Purpose                                                            |
|---------------------------|--------------------------------------------------------------------|
| `run_ablation_eval.py`    | four inference modes on a single checkpoint (offline metrics)      |
| `batch_ablation_eval.sh`  | run the ablation over every trained checkpoint                     |
| `batch_ablation_eval_extended.sh` | round-2 batch (λ sweep extension, task comparison)                 |
| `run_ensemble_sweep_eval.py`     | ensemble-size sweep (k = 1, 2, 5, 10, 20)                          |
| `run_closedloop_eval.py`  | closed-loop eval — i2v imagines future frames at every step        |
| `run_sim_eval_batch.py`   | simulator success eval with hard physics-based success detection   |
| `evaluate_in_sim.py`      | playback / policy rollout in the robosuite simulator              |
| `generate_full_report.py` | aggregate all `.pt` result files into the summary report          |

## Usage

```bash
python 3_ablation_eval/run_ablation_eval.py        # one checkpoint, 4 modes
bash   3_ablation_eval/batch_ablation_eval.sh      # all checkpoints
python 3_ablation_eval/run_closedloop_eval.py      # closed-loop success
python 3_ablation_eval/run_sim_eval_batch.py       # simulator success
python 3_ablation_eval/generate_full_report.py     # build the summary report
```

The closed-loop and simulator scripts need the `mimicgen/` simulation package
(`MIMICGEN_REPO`) and the i2v model (`HUMMINGBIRD_I2V`). Results are written to
`ablation_results/`.
