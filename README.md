# Weakly Supervised Learning Experiments

本仓库包含 **MoRe**（弱监督语义分割）和 **CDTR**（弱监督目标定位）两个论文的复现实验代码。

## 📋 概述

本仓库实现了两个弱监督学习实验的复现：

### 🔷 MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation
- **论文**: [MoRe (AAAI 2025)](https://arxiv.org/pdf/2412.11076)
- **数据集**: PASCAL VOC 2012, MS COCO 2014
- **任务**: 弱监督语义分割 (WSSS)

### 🔷 CDTR: CLIP-Driven Transformer for Weakly Supervised Object Localization
- **论文**: [CDTR](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Category-aware_Allocation_Transformer_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf)
- **数据集**: CUB-200-2011, ILSVRC
- **任务**: 弱监督目标定位 (WSOL)

## 📁 仓库结构

```
.
├── MoRe/              # MoRe 实验代码
│   ├── scripts/       # 训练和评估脚本
│   ├── model/         # 模型实现
│   ├── datasets/      # 数据集加载器
│   ├── utils/         # 工具函数
│   └── README.md      # MoRe 相关文档
│
├── CDTR/              # CDTR 实验代码
│   ├── scripts/       # 训练和评估脚本
│   ├── models.py      # 模型实现
│   ├── datasets/      # 数据集加载器
│   └── README.md      # CDTR 相关文档
│
├── REPRODUCTION_GUIDE.md    # 详细复现指南
└── start_experiments.sh     # 启动两个实验的脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.8
- PyTorch 1.10+ (支持 CUDA)
- 详细依赖请参考各实验的 README 文件

### 运行实验

1. **MoRe 实验**:
   ```bash
   cd MoRe
   bash run_train.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port] train_voc
   ```

2. **CDTR 实验**:
   ```bash
   cd CDTR
   bash run_repro.sh
   ```

详细说明请参考：
- [MoRe README](MoRe/README.md)
- [CDTR README](CDTR/README.md)
- [复现指南](REPRODUCTION_GUIDE.md)

## 📊 实验状态

- ✅ **MoRe**: 已完成 PASCAL VOC 2012 训练（20,000 次迭代）
- ✅ **CDTR**: 已完成 CUB-200-2011 训练（50 个 epoch）

## 📝 说明

- 所有实验输出（检查点、日志）保存在 `/root/autodl-tmp/experiments/`
- 每个实验的输出和日志分别存放在独立目录中
- 详细配置说明请参考 `REPRODUCTION_GUIDE.md`

## 📚 参考文献

- MoRe: Yang, Z., et al. "MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation." AAAI 2025.
- CDTR: Chen, Z., et al. "Category-aware Allocation Transformer for Weakly Supervised Object Localization." ICCV 2023.

## 📄 许可

许可信息请参考原始论文和仓库。
