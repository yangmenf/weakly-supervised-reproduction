# [AAAI2025] MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation [![arXiv](https://img.shields.io/badge/arXiv-2303.02506-b31b1b.svg)](https://arxiv.org/pdf/2412.11076)

MoRe 通过正则化类-补丁注意力，有效解决弱监督语义分割中从类-补丁注意力生成定位注意力图（LAM）时的伪影问题。

## News

* **If you find this work helpful, please give us a :star2: to receive the updation !**
* **` Dec. 10th, 2024`:** MoRe is accepted by AAAI2025.
* **All code is available.** 🔥🔥🔥

## Overview

<p align="middle">
<img src="/sources/main_figs.png" alt="MoRe pipeline" width="1200px">
</p>

弱监督语义分割（WSSS）使用图像级标签时，通常使用类激活图（CAM）实现密集预测。最近，Vision Transformer（ViT）提供了从类-补丁注意力生成定位图的替代方案。然而，由于对建模此类注意力的约束不足，定位注意力图（LAM）经常遇到伪影问题，即语义相关性最小的补丁区域被类标记错误激活。本文提出 MoRe 来解决该问题并进一步探索 LAM 的潜力。研究发现，对类-补丁注意力施加额外的正则化是必要的。为此，首先将注意力视为有向图，提出图类别表示模块，隐式正则化类-补丁实体之间的交互。该模块确保类标记在图级别动态压缩相关补丁信息并抑制无关伪影。其次，基于分类权重生成的 CAM 保持对象平滑定位的观察，设计定位信息正则化模块，显式正则化类-补丁注意力。该模块直接从 CAM 挖掘标记关系，并以可学习方式进一步监督类和补丁标记之间的一致性。在 PASCAL VOC 和 MS COCO 上进行了大量实验，验证了 MoRe 有效解决伪影问题并达到最先进的性能，超越了最近的单阶段甚至多阶段方法。

## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

增强标注来自 [SBD 数据集](http://home.bharathh.info/pubs/codes/SBD/download.html)。增强标注下载链接位于 [DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)。下载 `SegmentationClassAug.zip` 后，解压并移动到 `VOCdevkit/VOC2012/`。

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

为 COCO 生成 VOC 风格的分割标签，可使用 [repo](https://github.com/alicranck/coco2voc) 提供的脚本，或直接从 [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link) 下载生成的掩码。

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```

## Requirement

请参考 requirements.txt。

我们集成了用于分割的正则化损失。请参考该 [python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module) 的说明。

## Train MoRe
``` bash
### train voc
bash run_train.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port]  train_voc

### train coco
bash run_train.sh scripts/train_coco.py [gpu_devices] [gpu_numbers] [master_port] train_coco
```

## Evaluate MoRe
``` bash
### eval voc seg and LAM
bash run_evaluate_voc.sh [gpu_device] [gpu_number] [checkpoint_path]

### eval coco seg
bash run_evaluate_seg_coco.sh tools/infer_seg_coco.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]
```

## Main Results

#### 1. Artifact Issue

<p align="middle">
<img src="/sources/artifact_issue.png" alt="artifact issue" width="1200px">
</p>

#### 2. Semantic Results
VOC 和 COCO 上的语义性能。日志和权重现已可用。
| Dataset | Backbone |  Val  | Test | Log |
|:-------:|:--------:|:-----:|:----:|:---:|
|   PASCAL VOC   |   ViT-B  | 76.4  | [75.0](http://host.robots.ox.ac.uk/anonymous/9QW1IM.html) | [log](logs/voc_train.log) |
|   MS COCO  |   ViT-B  |  47.4 |   -  | [log](logs/coco_train.log) |

## Citation 
如果本工作对您的研究有帮助，请引用我们的工作。:two_hearts:
```bash
@article{yang2024more,
  title={MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation},
  author={Yang, Zhiwei and Meng, Yucong and Fu, Kexue and Wang, Shuo and Song, Zhijian},
  journal={arXiv preprint arXiv:2412.11076},
  year={2024}
}
```
如有任何问题，请通过 zwyang21@m.fudan.edu.cn 联系作者。

## Acknowledgement
本仓库基于 [MCTformer Series](https://github.com/xulianuwa/MCTformer.git) 和 [SeCo](https://github.com/zwyang6/SeCo.git) 构建。感谢他们的出色工作！！！
