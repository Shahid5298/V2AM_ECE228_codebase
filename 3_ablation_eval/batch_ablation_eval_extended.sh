#!/bin/bash
# =============================================================================
# Batch Ablation Eval 2 — Evaluates all Round 2 overnight checkpoints.
# dopri5 removed. Only Euler + Ensemble k=5.
#
# Run this after train_ablation_sweep_extended.sh completes:
#   (run from anywhere; this script cd's to the repo root)
#   bash batch_ablation_eval_extended.sh
#
# Results saved to ablation_results/results_<name>.pt
# Final report:  ablation_results/FINAL_FINDINGS_REPORT.txt
# =============================================================================

cd "$(dirname "$0")/.."          # run from the repo root
PYTHON=${PYTHON:-python}
RESULTS_DIR=ablation_results

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "BATCH ABLATION EVAL 2 STARTED: $(date)"
echo "============================================================"

# Helper: run euler + ensemble only, save result
eval_checkpoint() {
    local NAME=$1
    local TASK_ID=$2
    local OUT="$RESULTS_DIR/results_${NAME}.pt"
    local CKPT="checkpoint/model_${NAME}.pt"
    local STATS="checkpoint/action_stats_${NAME}.pt"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME — checkpoint not found: $CKPT"
        return
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Evaluating: $NAME  (task_id=$TASK_ID)"
    echo "------------------------------------------------------------"

    # Run euler
    CUDA_VISIBLE_DEVICES=0 $PYTHON 3_ablation_eval/run_ablation_eval.py \
        --checkpoint "$CKPT" \
        --action-stats "$STATS" \
        --task-id "$TASK_ID" \
        --mode euler \
        2>&1 | tee "$RESULTS_DIR/eval_log_${NAME}_euler.txt"

    # Stash euler result before ensemble overwrites ablation_results.pt
    if [ -f "ablation_results.pt" ]; then
        cp ablation_results.pt "${RESULTS_DIR}/tmp_euler_${NAME}.pt"
    fi

    # Run ensemble
    CUDA_VISIBLE_DEVICES=0 $PYTHON 3_ablation_eval/run_ablation_eval.py \
        --checkpoint "$CKPT" \
        --action-stats "$STATS" \
        --task-id "$TASK_ID" \
        --mode ensemble \
        2>&1 | tee "$RESULTS_DIR/eval_log_${NAME}_ensemble.txt"

    if [ -f "ablation_results.pt" ]; then
        cp ablation_results.pt "${RESULTS_DIR}/tmp_ensemble_${NAME}.pt"
    fi

    # Merge euler + ensemble into one .pt file
    $PYTHON - << PYEOF
import torch
from pathlib import Path

euler_path    = Path("$RESULTS_DIR/tmp_euler_${NAME}.pt")
ensemble_path = Path("$RESULTS_DIR/tmp_ensemble_${NAME}.pt")
out_path      = Path("$OUT")

r = {}
if euler_path.exists():
    r_e = torch.load(euler_path, weights_only=False)
    r["euler"] = r_e["euler"]
if ensemble_path.exists():
    r_en = torch.load(ensemble_path, weights_only=False)
    r["ensemble"] = r_en["ensemble"]

torch.save(r, out_path)
print(f"Merged -> {out_path}")

# Clean up tmp files
euler_path.unlink(missing_ok=True)
ensemble_path.unlink(missing_ok=True)
PYEOF

    echo "Saved: $OUT"
}

# ---- Round 2 checkpoints ----
eval_checkpoint "task10_cubestack_lambda0p5"  10
eval_checkpoint "task12_stackthree_lambda0p5" 12
eval_checkpoint "task0_lambda1p0"              0
eval_checkpoint "task0_lambda2p0"              0

# ---- Generate the final combined report ----
echo ""
echo "============================================================"
echo "GENERATING FINAL REPORT..."
echo "============================================================"

$PYTHON - << 'PYEOF'
import torch
from pathlib import Path
import datetime

results_dir = Path("ablation_results")

# ── All available results keyed by display name ──────────────────────────────
RUNS = {
    # ── λ sweep (task_0) ──────────────────────────────────────────────────────
    "task0 λ=0.0  (Euler-10 baseline)": ("ablation_results_original_model",   0),
    "task0 λ=0.0  (Euler-50 control)" : ("results_task0_lambda0p0_euler50_control", 0),
    "task0 λ=0.05"                    : ("results_task0_lambda0p05",           0),
    "task0 λ=0.1" : ("results_task0_lambda0p0_euler50_control", 0),   # placeholder
    "task0 λ=0.1 [lsmooth]"           : ("ablation_results_lsmooth_model",     0),
    "task0 λ=0.5"                     : ("results_task0_lambda0p5",            0),
    "task0 λ=1.0"                     : ("results_task0_lambda1p0",            0),
    "task0 λ=2.0"                     : ("results_task0_lambda2p0",            0),
    # ── Task comparison @ λ=0.1 (round 1) ────────────────────────────────────
    "task10 (cube stack)  λ=0.0"      : ("results_task10_cubestack_lambda0p0", 10),
    "task10 (cube stack)  λ=0.1"      : ("results_task10_cubestack_lambda0p1", 10),
    "task12 (stack three) λ=0.1"      : ("results_task12_stackthree_lambda0p1",12),
    # ── Task comparison @ λ=0.5 (round 2) ────────────────────────────────────
    "task10 (cube stack)  λ=0.5"      : ("results_task10_cubestack_lambda0p5", 10),
    "task12 (stack three) λ=0.5"      : ("results_task12_stackthree_lambda0p5",12),
}

def load(stem):
    p = results_dir / f"{stem}.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)

def row(label, r, mode="euler", baseline_jerk=None):
    if r is None or mode not in r:
        return f"  {label:<42} | MISSING"
    m = r[mode]
    jd = ""
    if baseline_jerk is not None:
        jd = f"  ({(m['mean_jerk']-baseline_jerk)/baseline_jerk*100:+.1f}% jerk)"
    return (f"  {label:<42} | {m['mse']:.5f} | {m['r2']:.4f} | "
            f"{m['gripper_accuracy']*100:.2f}% | {m['mean_jerk']:.6f}{jd}")

HDR = f"  {'Config':<42} | {'MSE':>7} | {'R²':>6} | {'Grip':>7} | {'Jerk':>10}"
SEP = "  " + "-" * 85

lines = []
lines.append("=" * 90)
lines.append("V2AM FINAL ABLATION REPORT")
lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("=" * 90)

# ── TABLE 1: Inference ablation ───────────────────────────────────────────────
lines += ["", "TABLE 1 — INFERENCE ABLATION (original model, task_0)",
          "  Same weights, only inference changes.", HDR, SEP]
orig = load("ablation_results_original_model")
if orig:
    lines.append(f"  {'V2AM baseline Euler-10':<42} | {orig['euler']['mse']:.5f} | {orig['euler']['r2']:.4f} | {orig['euler']['gripper_accuracy']*100:.2f}% | {orig['euler']['mean_jerk']:.6f}")
    m = orig['ensemble']
    bm = orig['euler']['mse']
    lines.append(f"  {'+ Ensemble k=5':<42} | {m['mse']:.5f} | {m['r2']:.4f} | {m['gripper_accuracy']*100:.2f}% | {m['mean_jerk']:.6f}  ({(m['mse']-bm)/bm*100:+.1f}% MSE)")
lines.append("")

# ── TABLE 2: λ sweep ─────────────────────────────────────────────────────────
lines += ["TABLE 2 — LAMBDA SWEEP (task_0, coffee pod, Euler inference)", HDR, SEP]
sweep = [
    ("task0 λ=0.0  (Euler-50 control)", "task0 λ=0.0 [ctrl]"),
    ("task0 λ=0.05",                    "task0 λ=0.05"),
    ("task0 λ=0.1 [lsmooth]",           "task0 λ=0.1"),
    ("task0 λ=0.5",                     "task0 λ=0.5"),
    ("task0 λ=1.0",                     "task0 λ=1.0"),
    ("task0 λ=2.0",                     "task0 λ=2.0"),
]
ctrl = load("results_task0_lambda0p0_euler50_control")
baseline_jerk = ctrl["euler"]["mean_jerk"] if ctrl else None
for key, label in sweep:
    stem = RUNS[key][0]
    r = load(stem)
    lines.append(row(label, r, "euler", baseline_jerk))
lines.append("")

# ── TABLE 3: λ sweep ensemble ────────────────────────────────────────────────
lines += ["TABLE 3 — LAMBDA SWEEP (task_0, coffee pod, Ensemble k=5)", HDR, SEP]
for key, label in sweep:
    stem = RUNS[key][0]
    r = load(stem)
    lines.append(row(label, r, "ensemble", None))
lines.append("")

# ── TABLE 4: Task comparison @ λ=0.5 ────────────────────────────────────────
lines += ["TABLE 4 — TASK COMPARISON AT λ=0.5 (Euler inference)",
          "  KEY RESULT: does contact-richness amplify L_smooth benefit at best λ?",
          HDR, SEP]
task_comp = [
    ("task0 λ=0.0  (Euler-50 control)", "coffee pod  λ=0.0"),
    ("task0 λ=0.5",                     "coffee pod  λ=0.5"),
    ("task10 (cube stack)  λ=0.0",      "cube stack  λ=0.0"),
    ("task10 (cube stack)  λ=0.5",      "cube stack  λ=0.5"),
    ("task12 (stack three) λ=0.5",      "stack three λ=0.5"),
]
for key, label in task_comp:
    stem = RUNS[key][0]
    r = load(stem)
    lines.append(row(label, r, "euler", None))
lines.append("")

# ── TABLE 5: Task comparison jerk delta ─────────────────────────────────────
lines += ["TABLE 5 — JERK REDUCTION BY TASK (λ=0.0 vs λ=0.5, Euler)",
          f"  {'Task':<30} | {'λ=0.0 Jerk':>12} | {'λ=0.5 Jerk':>12} | {'Jerk Δ':>10}",
          "  " + "-" * 70]
task_pairs = [
    ("results_task0_lambda0p0_euler50_control", "results_task0_lambda0p5",            "coffee pod (smooth)"),
    ("results_task10_cubestack_lambda0p0",       "results_task10_cubestack_lambda0p5", "cube stacking (contact)"),
    ("results_task12_stackthree_lambda0p1",      "results_task12_stackthree_lambda0p5","stack three (multi-step)"),
]
for stem0, stem5, label in task_pairs:
    r0 = load(stem0)
    r5 = load(stem5)
    if r0 and r5 and "euler" in r0 and "euler" in r5:
        j0 = r0["euler"]["mean_jerk"]
        j5 = r5["euler"]["mean_jerk"]
        delta = (j5 - j0) / j0 * 100
        lines.append(f"  {label:<30} | {j0:>12.6f} | {j5:>12.6f} | {delta:>+9.1f}%")
    else:
        lines.append(f"  {label:<30} | {'MISSING':>12} | {'MISSING':>12} |")
lines.append("")

# ── TABLE 6: Ensemble UQ / variance across λ ────────────────────────────────
lines += ["TABLE 6 — ENSEMBLE VARIANCE (UQ signal, task_0, all λ)",
          f"  {'Config':<30} | {'Mean Var':>10} | {'Range':>8}",
          "  " + "-" * 55]
var_runs = [
    ("ablation_results_original_model",          "λ=0.0 (Euler-10)"),
    ("results_task0_lambda0p0_euler50_control",  "λ=0.0 (Euler-50)"),
    ("ablation_results_lsmooth_model",           "λ=0.1"),
    ("results_task0_lambda0p5",                  "λ=0.5"),
    ("results_task0_lambda1p0",                  "λ=1.0"),
    ("results_task0_lambda2p0",                  "λ=2.0"),
]
for stem, label in var_runs:
    r = load(stem)
    if r and "ensemble" in r and r["ensemble"].get("per_step_variance"):
        v = r["ensemble"]["per_step_variance"]
        mean_v = sum(v) / len(v)
        rng    = max(v) - min(v)
        lines.append(f"  {label:<30} | {mean_v:>10.6f} | {rng:>8.6f}")
    else:
        lines.append(f"  {label:<30} | {'MISSING':>10} | {'MISSING':>8}")
lines.append("")

lines.append("=" * 90)
lines.append("END OF REPORT")
lines.append("=" * 90)

report_text = "\n".join(lines)
print(report_text)

out = results_dir / "FINAL_FINDINGS_REPORT.txt"
out.write_text(report_text)
print(f"\nReport saved to {out}")
PYEOF

echo ""
echo "============================================================"
echo "BATCH EVAL 2 COMPLETE: $(date)"
echo "============================================================"
