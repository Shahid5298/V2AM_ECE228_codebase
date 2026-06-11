#!/bin/bash
# =============================================================================
# Overnight Training Script 2 — V2AM Ablation (Round 2)
# Fixes the main gap from Round 1: all task comparisons now use λ=0.5 (the
# best λ), and extends the sweep to find where accuracy starts degrading.
#
# Jobs:
#   1. task10 (cube stacking) λ=0.5   — task comparison at best λ
#   2. task12 (stack three)  λ=0.5   — task comparison at best λ
#   3. task0  (coffee pod)   λ=1.0   — find degradation point
#   4. task0  (coffee pod)   λ=2.0   — confirm degradation point
#
# Usage (inside tmux), from anywhere:
#   bash 2_action_head_training/train_ablation_sweep_extended.sh
#
# Monitor:
#   tail -f logs/run_<name>.log
# =============================================================================

set -e
cd "$(dirname "$0")/.."          # run from the repo root
PYTHON=${PYTHON:-python}
SCRIPT=2_action_head_training/train_flow_matching_head.py
CKPT_DIR=checkpoint
LOG_DIR=logs

mkdir -p "$LOG_DIR"

TOTAL_START=$(date +%s)
echo "============================================================"
echo "OVERNIGHT TRAINING 2 STARTED: $(date)"
echo "============================================================"

# ------------------------------------------------------------
run_job() {
    local NAME=$1
    local TASK_ID=$2
    local LAMBDA=$3
    local EPOCHS=${4:-40}

    echo ""
    echo "------------------------------------------------------------"
    echo "START: $NAME  (task=$TASK_ID  lambda=$LAMBDA  epochs=$EPOCHS)"
    echo "Time:  $(date)"
    echo "------------------------------------------------------------"

    JOB_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
        --task-id "$TASK_ID" \
        --epochs "$EPOCHS" \
        --history-num-frames 1 \
        --smooth-loss-weight "$LAMBDA" \
        --wandb-project v2am-ablation \
        2>&1 | tee "$LOG_DIR/run_${NAME}.log"

    JOB_END=$(date +%s)
    ELAPSED=$(( (JOB_END - JOB_START) / 60 ))

    OUTPUT_DIR=$(ls -td outputs/flow_matching_future_video/*/ | head -1)
    echo "Output dir: $OUTPUT_DIR"

    if [ -f "${OUTPUT_DIR}best.pt" ]; then
        cp "${OUTPUT_DIR}best.pt"          "$CKPT_DIR/model_${NAME}.pt"
        echo "Saved checkpoint -> $CKPT_DIR/model_${NAME}.pt"
    fi
    if [ -f "${OUTPUT_DIR}action_stats_continuous_8tasks.pt" ]; then
        cp "${OUTPUT_DIR}action_stats_continuous_8tasks.pt" \
           "$CKPT_DIR/action_stats_${NAME}.pt"
        echo "Saved stats      -> $CKPT_DIR/action_stats_${NAME}.pt"
    fi
    if [ -f "${OUTPUT_DIR}history.pt" ]; then
        cp "${OUTPUT_DIR}history.pt"       "$CKPT_DIR/history_${NAME}.pt"
    fi

    echo "DONE: $NAME  (${ELAPSED} min)"
    echo "------------------------------------------------------------"
}
# ------------------------------------------------------------

# RUN 1 — Cube stacking at λ=0.5
# Purpose: Task comparison was done at λ=0.1 in round 1. λ=0.5 is the best λ.
#          If the 2.9x amplification (coffee=4.6%, cube=13.3%) holds at λ=0.5,
#          cube stacking jerk reduction could be ~50-60%. KEY result.
run_job "task10_cubestack_lambda0p5" 10 0.5 40

# RUN 2 — Stack three cubes at λ=0.5
# Purpose: Third data point for task comparison at the canonical best λ.
run_job "task12_stackthree_lambda0p5" 12 0.5 40

# RUN 3 — Coffee pod at λ=1.0
# Purpose: λ=0.5 gives -38.5% jerk with zero accuracy cost. Where does it
#          start to hurt? The λ sweep table is incomplete without a degradation
#          point. If MSE starts rising here, λ=0.5 is confirmed as sweet spot.
run_job "task0_lambda1p0" 0 1.0 40

# RUN 4 — Coffee pod at λ=2.0
# Purpose: Confirm degradation. If λ=1.0 still helps, λ=2.0 should show
#          clear accuracy trade-off, completing the sweet-spot narrative.
run_job "task0_lambda2p0" 0 2.0 40

# ------------------------------------------------------------
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE: $(date)"
echo "Total time: ${TOTAL_MIN} minutes"
echo ""
echo "Checkpoints saved:"
ls -lh "$CKPT_DIR"/model_task*.pt 2>/dev/null || echo "  (none found)"
echo ""
echo "Next steps:"
echo "  1. Run k-sweep (inference only, ~10 min):"
echo "     bash run_ensemble_sweep_eval.sh"
echo "  2. Run full eval on new checkpoints:"
echo "     bash batch_ablation_eval_extended.sh"
echo "  3. Generate final report:"
echo "     conda run -n ml python generate_full_report.py"
echo "============================================================"
