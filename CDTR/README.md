# CLIP-Driven Transformer for Weakly Supervised Object Localization

弱监督目标定位的 CLIP 驱动 Transformer 的 PyTorch 实现。

''CLIP-Driven Transformer for Weakly Supervised Object Localization'' 基于我们的会议版本 ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Category-aware_Allocation_Transformer_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf)) 构建。

## 📋 Table of content
 1. [📎 Paper Link](#1)
 2. [💡 Abstract](#2)
 3. [📖 Method](#3)
 4. [📃 Requirements](#4)
 5. [✏️ Usage](#5)
    1. [Start](#51)
    2. [Prepare Datasets](#52)
    2. [Model Zoo](#53)
    3. [Training](#54)
    4. [Inference](#55)
 6. [🔍 Citation](#6)
 7. [❤️ Acknowledgement](#7)

## 📎 Paper Link <a name="1"></a> 

* Category-aware Allocation Transformer for Weakly Supervised Object Localization ([link](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Category-aware_Allocation_Transformer_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf))

    Authors: Zhiwei Chen, Jinren Ding, Liujuan Cao, Yunhang Shen, Shengchuan Zhang, Guannan Jiang, Rongrong Ji
    
    Institution: Xiamen University, Xiamen, China. Tencent Youtu Lab, Shanghai, China. CATL, China.
    
* CLIP-Driven Transformer for Weakly Supervised Object Localization ([link]())

    Authors: Zhiwei Chen, Yunhang Shen, Liujuan Cao, Shengchuan Zhang, Rongrong Ji
    
    Institution: Xiamen University, Xiamen, China. Tencent Youtu Lab, Shanghai, China.


## 💡 Abstract <a name="2"></a> 
弱监督目标定位（WSOL）旨在仅使用图像级标签作为监督来定位目标。尽管最近将 transformer 纳入 WSOL 的进展带来改进，但这些方法通常依赖于类别无关的注意力图，导致次优的目标定位。本文提出一种新的 CLIP 驱动 Transformer（CDTR），学习类别感知表示以实现准确的目标定位。具体而言，首先提出类别感知刺激模块（CSM），将可学习的类别偏置嵌入到自注意力图中，通过辅助监督增强学习过程。此外，设计目标约束模块（OCM），以自监督方式细化目标区域，利用 CSM 提供的自注意力图的判别潜力。为在 CSM 和 OCM 之间建立协同连接，进一步开发语义核积分器（SKI），为自注意力图生成语义核。同时，探索 CLIP 模型并设计语义增强适配器（SBA），通过将语义特定的图像和文本表示集成到自注意力图中来丰富目标表示。在 CUB-200-2011 和 ILSVRC 等基准数据集上的大量实验评估突出了 CDTR 框架的优越性能。本研究的代码和模型可在 https://github.com/zhiweichen0012/CDTR 获取。

## 📖 Method <a name="3"></a> 

<p align="center">
    <img src="./Img/network.png" width="750"/> <br />
    <em> 
    </em>
</p>
提出的 CLIP 驱动 Transformer（CDTR）架构。

## 📃 Requirements <a name="4"></a> 
  - PyTorch==1.10.1  
  - torchvision==0.11.2
  - timm==0.4.12

## ✏️ Usage <a name="5"></a> 

### Start <a name="51"></a> 

```bash  
git clone git@github.com:zhiweichen0012/CDTR.git
cd CDTR
```

### Prepare Datasets <a name="52"></a> 

* CUB ([http://www.vision.caltech.edu/datasets/cub_200_2011/](http://www.vision.caltech.edu/datasets/cub_200_2011/))
* ILSVRC ([https://www.image-net.org/challenges/LSVRC/](https://www.image-net.org/challenges/LSVRC/))

目录结构遵循 torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) 的标准布局，训练和验证数据应分别位于 `train/` 和 `val` 文件夹中：

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Model Zoo <a name="53"></a> 
我们提供训练好的 CDTR 模型。
| Name | Loc. Acc@1 | Loc. Acc@5 | URL |
| --- | --- | --- | --- |
| CDTR_CUB | 81.33     | 94.06     | [model](https://drive.google.com/drive/folders/144yLFl9gJxPp1uC4RThQIqCy3GIz5OsB?usp=sharing) |
| CDTR_ILSVRC | 58.20 | 68.05 | [model](https://drive.google.com/drive/folders/144yLFl9gJxPp1uC4RThQIqCy3GIz5OsB?usp=sharing) |

### Training <a name="54"></a> 

使用 4 个 GPU 在 CUB 上训练 CDTR：

```bash
bash scripts/train.sh deit_small_patch16_224_CDTR_cub CUB 110 /path/to/output_ckpt/CUB
```

使用 4 个 GPU 在 ILSVRC 上训练 CDTR：

```bash
bash scripts/train.sh deit_small_patch16_224_CDTR_imnet IMNET 14 /path/to/output_ckpt/IMNET
```

注意：请检查 ``` scripts/train.sh ``` 中 "torchrun" 命令、数据集和预训练权重的路径。

### Inference <a name="55"></a> 

测试 CUB 模型：

```bash  
bash scripts/test.sh deit_small_patch16_224_CDTR_cub CUB /path/to/CDTR_CUB_model
```

测试 ILSVRC 模型：

```bash  
bash scripts/test.sh deit_small_patch16_224_CDTR_imnet IMNET /path/to/CDTR_IMNET_model
```

注意：请检查 ``` scripts/test.sh ``` 中 "python3" 命令和数据集的路径。

## 🔍 Citation <a name="6"></a> 

```
@inproceedings{chen2023category,
  title={Category-aware Allocation Transformer for Weakly Supervised Object Localization},
  author={Chen, Zhiwei and Ding, Jinren and Cao, Liujuan and Shen, Yunhang and Zhang, Shengchuan and Jiang, Guannan and Ji, Rongrong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6643--6652},
  year={2023}
}
```

## ❤️ Acknowledgement <a name="7"></a> 

我们使用 [deit](https://github.com/facebookresearch/deit) 及其 [预训练权重](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) 作为骨干网络。感谢他们的出色工作！
