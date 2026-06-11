#!/bin/bash
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# I2V LoRA Training on LIBERO Dataset

set -e

GPU=0

echo "============================================"
echo "  I2V LoRA Training on LIBERO Dataset"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU python3 train_lora.py \
    --config config.yaml

echo "Training complete!"
