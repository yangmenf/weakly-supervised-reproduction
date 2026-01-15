#!/bin/bash
# 模型评估脚本
# 使用前请确保GPU已开启

set -e

echo "开始运行模型评估..."
echo "时间: $(date)"

# MoRe评估
echo "========================================="
echo "开始MoRe模型评估 (VOC数据集)"
echo "========================================="
cd /root/Result\ Reproduction/MoRe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paper_repro

python tools/infer_seg_voc.py \
    --model_path /root/autodl-tmp/experiments/MoRe/outputs/2026-01/voc_reproduce_voc_more_13-11-05-12/checkpoints/model_iter_20000.pth \
    --data_folder /root/autodl-tmp/data/VOC/VOCdevkit/VOC2012/ \
    --infer_set val \
    2>&1 | tee /root/autodl-tmp/experiments/MoRe/logs/eval_voc_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "MoRe评估完成"
echo ""

# CDTR评估
echo "========================================="
echo "开始CDTR模型评估 (CUB数据集)"
echo "========================================="
cd /root/Result\ Reproduction/CDTR

python main.py \
    --eval \
    --model deit_small_patch16_224_CDTR_cub \
    --data-path /root/autodl-tmp/data/CUB/CUB_200_2011 \
    --data-set CUB \
    --resume /root/autodl-tmp/experiments/CDTR/outputs/model_epoch50.pth \
    2>&1 | tee /root/autodl-tmp/experiments/CDTR/logs/eval_cub_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "CDTR评估完成"
echo ""
echo "所有评估完成！时间: $(date)"
