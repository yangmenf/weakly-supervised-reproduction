#!/bin/bash
# 启动论文复现实验的后台脚本

set -e

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paper_repro

# 创建日志目录（在数据盘）
mkdir -p /root/autodl-tmp/experiments/MoRe/logs
mkdir -p /root/autodl-tmp/experiments/CDTR/logs

# 启动MoRe训练（VOC数据集）
echo "启动MoRe训练..."
cd /root/Result\ Reproduction/MoRe
nohup bash run_train.sh scripts/train_voc.py 0 1 29500 "reproduce_voc_more" > /root/autodl-tmp/experiments/MoRe/logs/train_voc_$(date +%Y%m%d_%H%M%S).log 2>&1 &
MORE_PID=$!
echo "MoRe训练已启动，PID: $MORE_PID，日志: /root/autodl-tmp/experiments/MoRe/logs/train_voc_*.log"

# 等待几秒确保MoRe启动成功
sleep 5

# 启动CDTR训练（CUB数据集）
echo "启动CDTR训练..."
cd /root/Result\ Reproduction/CDTR
nohup bash run_repro.sh > /root/autodl-tmp/experiments/CDTR/logs/train_cub_$(date +%Y%m%d_%H%M%S).log 2>&1 &
CDTR_PID=$!
echo "CDTR训练已启动，PID: $CDTR_PID，日志: /root/autodl-tmp/experiments/CDTR/logs/train_cub_*.log"

# 保存PID到文件
echo "$MORE_PID" > /root/Result\ Reproduction/MoRe/.train_pid
echo "$CDTR_PID" > /root/Result\ Reproduction/CDTR/.train_pid

echo ""
echo "=========================================="
echo "两个实验已在后台启动"
echo "MoRe PID: $MORE_PID"
echo "CDTR PID: $CDTR_PID"
echo "=========================================="
echo ""
echo "监控命令："
echo "  tail -f /root/autodl-tmp/experiments/MoRe/logs/train_voc_*.log"
echo "  tail -f /root/autodl-tmp/experiments/CDTR/logs/train_cub_*.log"
echo "  nvidia-smi"
echo ""
echo "停止命令："
echo "  kill \$(cat /root/Result\\ Reproduction/MoRe/.train_pid)"
echo "  kill \$(cat /root/Result\\ Reproduction/CDTR/.train_pid)"
