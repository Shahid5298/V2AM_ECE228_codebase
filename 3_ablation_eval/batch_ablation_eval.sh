#!/bin/bash
# =============================================================================
# Batch Ablation Evaluation — Run inference on all overnight checkpoints
# Run this tomorrow morning after train_ablation_sweep.sh completes.
#
# Usage:
#   (run from anywhere; this script cd's to the repo root)
#   bash batch_ablation_eval.sh
#
# Results saved to ablation_results/results_<name>.pt
# Summary printed to ablation_results/OVERNIGHT_SUMMARY.txt
# =============================================================================

cd "$(dirname "$0")/.."          # run from the repo root
PYTHON=${PYTHON:-python}
RESULTS_DIR=ablation_results

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "BATCH ABLATION EVAL STARTED: $(date)"
echo "============================================================"

# We already have results for these two — skip unless you want to re-run
# original model  : ablation_results_original_model.pt
# l_smooth model  : ablation_results_lsmooth_model.pt

# All new checkpoints from overnight training
declare -A CHECKPOINTS=(
    ["task0_lambda0p0_euler50_control"]="task0_lambda0p0_euler50_control"
    ["task0_lambda0p05"]="task0_lambda0p05"
    ["task0_lambda0p5"]="task0_lambda0p5"
    ["task10_cubestack_lambda0p0"]="task10_cubestack_lambda0p0"
    ["task10_cubestack_lambda0p1"]="task10_cubestack_lambda0p1"
    ["task12_stackthree_lambda0p1"]="task12_stackthree_lambda0p1"
)

for NAME in "${!CHECKPOINTS[@]}"; do
    CKPT="checkpoint/model_${NAME}.pt"
    STATS="checkpoint/action_stats_${NAME}.pt"
    OUT="$RESULTS_DIR/results_${NAME}.pt"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME — checkpoint not found: $CKPT"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Evaluating: $NAME"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 $PYTHON 3_ablation_eval/run_ablation_eval.py \
        --checkpoint "$CKPT" \
        --action-stats "$STATS" \
        --mode all \
        2>&1 | tee "$RESULTS_DIR/eval_log_${NAME}.txt"

    # The script always saves to ablation_results.pt — rename it
    if [ -f "ablation_results.pt" ]; then
        mv ablation_results.pt "$OUT"
        echo "Saved results -> $OUT"
    fi
done

# ------------------------------------------------------------
# Print summary table across all runs
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "GENERATING SUMMARY..."
echo "============================================================"

$PYTHON - << 'PYEOF'
import torch
from pathlib import Path

results_dir = Path("ablation_results")

# Map name -> (task, lambda, description)
RUN_META = {
    "ablation_results_original_model":      ("task_0",  0.0,  "coffee  | λ=0.0 | Euler-10 [EXISTING]"),
    "ablation_results_lsmooth_model":        ("task_0",  0.1,  "coffee  | λ=0.1 | Euler-50 [EXISTING]"),
    "results_task0_lambda0p0_euler50_control": ("task_0",  0.0,  "coffee  | λ=0.0 | Euler-50 [CONTROL]"),
    "results_task0_lambda0p05":              ("task_0",  0.05, "coffee  | λ=0.05"),
    "results_task0_lambda0p5":               ("task_0",  0.5,  "coffee  | λ=0.5"),
    "results_task10_cubestack_lambda0p0":    ("task_10", 0.0,  "cube    | λ=0.0"),
    "results_task10_cubestack_lambda0p1":    ("task_10", 0.1,  "cube    | λ=0.1"),
    "results_task12_stackthree_lambda0p1":   ("task_12", 0.1,  "stack3  | λ=0.1"),
}

lines = []
lines.append("=" * 100)
lines.append("OVERNIGHT ABLATION SUMMARY")
lines.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("=" * 100)

# ---- Euler single-sample results (main comparison) ----
lines.append("")
lines.append("TABLE A — EULER SINGLE SAMPLE (comparing across models/tasks/lambdas)")
lines.append(f"  {'Run':<42} | {'MSE':>8} | {'R²':>6} | {'Grip':>7} | {'NFE':>5} | {'Jerk':>10}")
lines.append("  " + "-" * 85)

for stem, (task, lam, desc) in RUN_META.items():
    pt = results_dir / f"{stem}.pt"
    if not pt.exists():
        lines.append(f"  {desc:<42} | MISSING")
        continue
    r = torch.load(pt, weights_only=False)
    m = r["euler"]
    lines.append(
        f"  {desc:<42} | {m['mse']:.5f} | {m['r2']:.4f} | {m['gripper_accuracy']*100:.2f}% "
        f"| {int(m['nfe']):>5} | {m['mean_jerk']:.6f}"
    )

# ---- Ensemble results ----
lines.append("")
lines.append("TABLE B — ENSEMBLE k=5 (comparing across models/tasks/lambdas)")
lines.append(f"  {'Run':<42} | {'MSE':>8} | {'R²':>6} | {'Grip':>7} | {'NFE':>5} | {'Jerk':>10}")
lines.append("  " + "-" * 85)

for stem, (task, lam, desc) in RUN_META.items():
    pt = results_dir / f"{stem}.pt"
    if not pt.exists():
        lines.append(f"  {desc:<42} | MISSING")
        continue
    r = torch.load(pt, weights_only=False)
    m = r["ensemble"]
    lines.append(
        f"  {desc:<42} | {m['mse']:.5f} | {m['r2']:.4f} | {m['gripper_accuracy']*100:.2f}% "
        f"| {int(m['nfe']):>5} | {m['mean_jerk']:.6f}"
    )

# ---- dopri5 results ----
lines.append("")
lines.append("TABLE C — dopri5 NEURAL ODE (comparing across models/tasks/lambdas)")
lines.append(f"  {'Run':<42} | {'MSE':>8} | {'R²':>6} | {'Grip':>7} | {'NFE (mean)':>12} | {'Jerk':>10}")
lines.append("  " + "-" * 90)

for stem, (task, lam, desc) in RUN_META.items():
    pt = results_dir / f"{stem}.pt"
    if not pt.exists():
        lines.append(f"  {desc:<42} | MISSING")
        continue
    r = torch.load(pt, weights_only=False)
    m = r["dopri5"]
    nfe_str = f"{m['nfe']:.1f}±{m['nfe_std']:.1f}"
    lines.append(
        f"  {desc:<42} | {m['mse']:.5f} | {m['r2']:.4f} | {m['gripper_accuracy']*100:.2f}% "
        f"| {nfe_str:>12} | {m['mean_jerk']:.6f}"
    )

# ---- Jerk reduction table: all lambda runs on task_0 ----
lines.append("")
lines.append("TABLE D — LAMBDA SWEEP: JERK ACROSS λ VALUES (task_0 coffee, Euler)")
lines.append(f"  {'λ':<8} | {'Euler MSE':>10} | {'Euler R²':>9} | {'Euler Jerk':>12} | {'Ens Jerk':>10} | {'Jerk vs λ=0'}")
lines.append("  " + "-" * 80)

lambda_runs = [
    ("ablation_results_original_model",        "0.0 (E10)"),
    ("results_task0_lambda0p0_euler50_control", "0.0 (E50)"),
    ("ablation_results_lsmooth_model",          "0.1 (E50)"),
    ("results_task0_lambda0p05",               "0.05"),
    ("results_task0_lambda0p5",                "0.5"),
]

baseline_jerk = None
for stem, lam_label in lambda_runs:
    pt = results_dir / f"{stem}.pt"
    if not pt.exists():
        lines.append(f"  {lam_label:<8} | MISSING")
        continue
    r = torch.load(pt, weights_only=False)
    ej = r["euler"]["mean_jerk"]
    enj = r["ensemble"]["mean_jerk"]
    if baseline_jerk is None and "0.0" in lam_label:
        baseline_jerk = ej
    diff = f"{(ej - baseline_jerk)/baseline_jerk*100:+.1f}%" if baseline_jerk else "—"
    lines.append(
        f"  {lam_label:<8} | {r['euler']['mse']:.6f} | {r['euler']['r2']:.5f} "
        f"| {ej:.6f}   | {enj:.6f} | {diff}"
    )

# ---- Task comparison ----
lines.append("")
lines.append("TABLE E — TASK COMPARISON: coffee vs cube_stack vs stack_three (λ=0.1, Euler)")
lines.append(f"  {'Task':<42} | {'MSE':>8} | {'R²':>6} | {'Jerk (Euler)':>13} | {'Jerk (Ens)':>11}")
lines.append("  " + "-" * 85)

task_runs = [
    ("ablation_results_lsmooth_model",       "coffee (task_0)  λ=0.1"),
    ("results_task10_cubestack_lambda0p1",   "cube stack(10)   λ=0.1"),
    ("results_task12_stackthree_lambda0p1",  "stack three(12)  λ=0.1"),
]
for stem, label in task_runs:
    pt = results_dir / f"{stem}.pt"
    if not pt.exists():
        lines.append(f"  {label:<42} | MISSING")
        continue
    r = torch.load(pt, weights_only=False)
    lines.append(
        f"  {label:<42} | {r['euler']['mse']:.5f} | {r['euler']['r2']:.4f} "
        f"| {r['euler']['mean_jerk']:.6f}      | {r['ensemble']['mean_jerk']:.6f}"
    )

lines.append("")
lines.append("=" * 100)

summary = "\n".join(lines)
print(summary)

out_path = Path("ablation_results/OVERNIGHT_SUMMARY.txt")
out_path.write_text(summary)
print(f"\nSummary saved to {out_path}")
PYEOF

echo ""
echo "============================================================"
echo "BATCH EVAL COMPLETE: $(date)"
echo "============================================================"
