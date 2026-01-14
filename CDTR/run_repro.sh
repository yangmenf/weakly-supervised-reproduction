#!/bin/bash
set -x

# Data paths
DATA_ROOT="/root/autodl-tmp/data/CUB/CUB_200_2011"
OUTPUT_DIR="/root/autodl-tmp/experiments/CDTR/outputs"
PYTHON_EXE="/root/miniconda3/envs/paper_repro/bin/python"

# Create output dir
mkdir -p $OUTPUT_DIR

# Run training
$PYTHON_EXE main.py \
  --model deit_small_patch16_224_CDTR_cub \
  --data-set CUB \
  --data-path "$DATA_ROOT" \
  --batch-size 32 \
  --lr 5e-5 \
  --epochs 50 \
  --output_dir "$OUTPUT_DIR"
