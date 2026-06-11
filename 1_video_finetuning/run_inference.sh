#!/bin/bash
# I2V LoRA Inference — Generate future frames from a robot observation
# Usage: ./run_inference.sh <image_path> <prompt> [lora_path]

set -e

GPU=0
IMAGE=${1:?"Usage: ./run_inference.sh <image_path> <prompt> [lora_path]"}
PROMPT=${2:?"Usage: ./run_inference.sh <image_path> <prompt> [lora_path]"}
LORA_PATH=${3:-""}

EXTRA_ARGS=""
if [ -n "$LORA_PATH" ]; then
    EXTRA_ARGS="--lora_path $LORA_PATH"
fi

CUDA_VISIBLE_DEVICES=$GPU python3 inference_lora.py \
    --image "$IMAGE" \
    --prompt "$PROMPT" \
    --height 256 \
    --width 256 \
    --video_length 16 \
    --ddim_steps 16 \
    --guidance_scale 7.5 \
    --output "output/${PROMPT// /_}.mp4" \
    $EXTRA_ARGS
