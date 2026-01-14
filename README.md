# Weakly Supervised Learning Experiments

Repository containing reproduction experiments for **MoRe** (Weakly Supervised Semantic Segmentation) and **CDTR** (Weakly Supervised Object Localization).

## 📋 Overview

This repository includes code implementations for reproducing two weakly supervised learning experiments:

### 🔷 MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation
- **Paper**: [MoRe (AAAI 2025)](https://arxiv.org/pdf/2412.11076)
- **Datasets**: PASCAL VOC 2012, MS COCO 2014
- **Task**: Weakly Supervised Semantic Segmentation (WSSS)

### 🔷 CDTR: CLIP-Driven Transformer for Weakly Supervised Object Localization
- **Paper**: [CDTR](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Category-aware_Allocation_Transformer_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf)
- **Datasets**: CUB-200-2011, ILSVRC
- **Task**: Weakly Supervised Object Localization (WSOL)

## 📁 Repository Structure

```
.
├── MoRe/              # MoRe experiment code
│   ├── scripts/       # Training and evaluation scripts
│   ├── model/         # Model implementations
│   ├── datasets/      # Dataset loaders
│   ├── utils/         # Utility functions
│   └── README.md      # MoRe-specific documentation
│
├── CDTR/              # CDTR experiment code
│   ├── scripts/       # Training and evaluation scripts
│   ├── models.py      # Model implementations
│   ├── datasets/      # Dataset loaders
│   └── README.md      # CDTR-specific documentation
│
├── REPRODUCTION_GUIDE.md    # Detailed reproduction guide
└── start_experiments.sh     # Script to start both experiments
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8
- PyTorch 1.10+ (with CUDA support)
- See individual experiment README files for detailed requirements

### Running Experiments

1. **MoRe Experiment**:
   ```bash
   cd MoRe
   bash run_train.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port] train_voc
   ```

2. **CDTR Experiment**:
   ```bash
   cd CDTR
   bash run_repro.sh
   ```

For detailed instructions, please refer to:
- [MoRe README](MoRe/README.md)
- [CDTR README](CDTR/README.md)
- [Reproduction Guide](REPRODUCTION_GUIDE.md)

## 📊 Experiments Status

- ✅ **MoRe**: Completed training on PASCAL VOC 2012 (20,000 iterations)
- ✅ **CDTR**: Completed training on CUB-200-2011 (50 epochs)

## 📝 Notes

- All experimental outputs (checkpoints, logs) are saved to `/root/autodl-tmp/experiments/`
- Each experiment has separate directories for outputs and logs
- See `REPRODUCTION_GUIDE.md` for detailed setup and configuration instructions

## 📚 References

- MoRe: Yang, Z., et al. "MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation." AAAI 2025.
- CDTR: Chen, Z., et al. "Category-aware Allocation Transformer for Weakly Supervised Object Localization." ICCV 2023.

## 📄 License

Please refer to the original papers and repositories for licensing information.
