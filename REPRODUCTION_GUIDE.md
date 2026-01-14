# 服务器复现指南 (MoRe & CDTR)

本文档记录在 Linux 服务器（如 RTX 4090D）上从零开始配置环境、准备数据并运行 **MoRe** 和 **CDTR** 复现实验的过程。

数据地址：`/root/autodl-tmp`  
代码地址：`/root/Result Reproduction`

---

## 1. 基础环境配置

创建新的虚拟环境以隔离依赖。

```bash
# 创建并激活名为 paper_repro 的环境 (Python 3.8)
conda create -n paper_repro python=3.8 -y
conda activate paper_repro

# 安装 PyTorch (适配 RTX 4090 的 CUDA 11.8/12.1 版本)
# 以 CUDA 12.1 为例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. 数据集解压与整理

数据集已下载到 `autodl-tmp`（`VOCtrainval` 和 `CUB` 压缩包已上传到服务器）。

### 2.1 准备 PASCAL VOC 2012 (用于 MoRe)

```bash
# 创建存放 VOC 的目录
mkdir -p ./data/VOC

# 解压 (注意 tar 包通常会解压出 VOCdevkit 文件夹)
tar -xvf VOCtrainval_11-May-2012.tar -C ./data/VOC

# 最终结构应为:
# ./data/VOC/VOCdevkit/VOC2012/JPEGImages
# ./data/VOC/VOCdevkit/VOC2012/SegmentationClass
```

### 2.2 准备 CUB-200-2011 (用于 CDTR)

```bash
# 创建存放 CUB 的目录
mkdir -p ./data/CUB

# 解压
tar -xzvf CUB_200_2011.tgz -C ./data/CUB

# 最终结构应为:
# ./data/CUB/CUB_200_2011/images
# ./data/CUB/CUB_200_2011/train_test_split.txt
```

---

## 3. MoRe 复现指南 (WSSS)

复现弱监督语义分割的训练流程。

### 3.1 安装依赖

进入 MoRe 代码目录：

```bash
cd MoRe
pip install -r requirements.txt
# 如果缺少 mmcv 等特定库，可能需要单独安装:
# pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

### 3.2 运行训练

MoRe 使用分布式训练脚本 `run_train.sh`。

```bash
# 修改 run_train.sh 中的参数 (如果需要)
# 用法: bash run_train.sh <脚本文件> <GPU编号> <每节点进程数> <主端口> <实验描述>

# 示例: 在 GPU 0 上使用单卡训练 (WSSS 阶段)
# 注意: 需要确认 train_cam.py 的路径 (假设为 tools/train_cam.py)
bash run_train.sh tools/train_cam.py 0 1 29500 "reproduce_voc_cam"
```

如果脚本报错，请检查 `run_train.sh` 中调用的 python 文件路径是否正确（通常是 `tools/train_wsss.py` 或类似，具体看代码目录 `tools/` 下的主训练文件）。

---

## 4. CDTR 复现指南 (WSOL)

复现弱监督目标定位的效果。

### 4.1 安装依赖

进入 CDTR 代码目录：

```bash
cd CDTR
pip install ftfy regex tqdm
# 安装 CLIP
pip install git+https://github.com/openai/CLIP.git
```

### 4.2 运行训练

CDTR 提供了 `scripts/train.sh`。需要指定 CUB 数据集的路径。

```bash
# 修改 scripts/train.sh 或直接运行命令
# 核心训练命令通常如下 (参考 README):

python main.py \
  --data_root ../data/CUB/CUB_200_2011 \
  --dataset CUB \
  --arch transformer \
  --batch_size 32 \
  --lr 5e-5 \
  --epochs 50 \
  --output_dir ./output_cub
```

使用提供的脚本：

```bash
# 确保在 CDTR 根目录下
bash scripts/train.sh main.py 0 1 29501 "reproduce_cub_cdtr"
```

注意检查 `scripts/train.sh` 内部是否硬编码了其他参数，可能需要手动修改它以指向正确的数据集路径。

---

## 5. 常见问题 (Troubleshooting)

1. **显存不足 (OOM)**:
   - 虽然 4090D 有 24GB，但如果 Batch Size 太大仍可能溢出。
   - 解决方法: 在运行命令中调小 `--batch_size`（例如从 32 降到 16）。

2. **数据路径报错 (FileNotFound)**:
   - 请再次确认解压后的路径结构。
   - MoRe 通常通过配置文件 `configs/` 指定路径，建议修改配置文件中的 `data_root`。
   - CDTR 通过命令行参数 `--data_root` 指定。

3. **CUDA 版本不匹配**:
   - 如果遇到 `RuntimeError: CUDA error`，请检查 `nvcc --version` 和 `pip list | grep torch` 是否一致。

---

**祝复现顺利！**
