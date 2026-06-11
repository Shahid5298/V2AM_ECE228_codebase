#!/bin/bash
# =============================================================================
# Overnight Training Script — V2AM Ablation
# Runs 6 training jobs back-to-back. Each job logs to logs/, copies its best
# checkpoint to checkpoint/ with a descriptive name when done.
#
# Usage (inside tmux), from anywhere:
#   bash 2_action_head_training/train_ablation_sweep.sh
#
# To monitor progress while running:
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
echo "OVERNIGHT TRAINING STARTED: $(date)"
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

    # Find the most recently created output dir (the one just trained)
    OUTPUT_DIR=$(ls -td outputs/flow_matching_future_video/*/ | head -1)
    echo "Output dir: $OUTPUT_DIR"

    # Copy best checkpoint with descriptive name
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

# RUN 1 — Control: task_0, no L_smooth, Euler-50 checkpointing
# Purpose: isolate whether our L_smooth gain came from the penalty itself
#          or just from switching to 50-step eval for checkpoint selection.
run_job "task0_lambda0p0_euler50_control" 0 0.0 40

# RUN 2 — λ sweep low: task_0, λ=0.05
# Purpose: does even a small jerk penalty help?
run_job "task0_lambda0p05" 0 0.05 40

# RUN 3 — λ sweep high: task_0, λ=0.5
# Purpose: does aggressive smoothing hurt accuracy or help more?
run_job "task0_lambda0p5" 0 0.5 40

# RUN 4 — Cube stacking baseline: task_10, no L_smooth
# Purpose: baseline for a contact-rich task with a clear grasp moment.
run_job "task10_cubestack_lambda0p0" 10 0.0 40

# RUN 5 — Cube stacking + L_smooth: task_10, λ=0.1
# Purpose: KEY RUN. Sharper contact event should produce a clearer
#          jerk spike and more visible L_smooth effect.
run_job "task10_cubestack_lambda0p1" 10 0.1 40

# RUN 6 — Stack three cubes + L_smooth: task_12, λ=0.1
# Purpose: Hardest task (multiple stacking steps). Most contact-rich.
#          Tests whether L_smooth scales to harder, longer-horizon tasks.
run_job "task12_stackthree_lambda0p1" 12 0.1 40

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
echo "============================================================"
