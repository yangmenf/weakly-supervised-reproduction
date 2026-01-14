# 论文复现实验重新启动总结

**重启时间**: 2025-01-12 16:26

---

## ✅ 已修复的问题

### 1. MoRe验证阶段错误
**问题**: `ValueError: Input and output must have the same number of spatial dimensions`
- **错误位置**: `engine/validatation_engine.py` 第31行
- **原因**: `labels.shape[1:]` 可能包含通道维度，导致维度不匹配
- **修复**: 添加了维度检查逻辑，正确提取空间维度（H, W）

**修复代码**:
```python
# 修复前
resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)

# 修复后
if len(labels.shape) == 4:  # B, C, H, W
    target_size = labels.shape[2:]
else:  # B, H, W or B, H, W, C
    target_size = labels.shape[1:3]
resized_cam = F.interpolate(_cams, size=target_size, mode='bilinear', align_corners=False)
```

### 2. CDTR检查点保存错误
**问题**: `AttributeError: 'NoneType' object has no attribute 'state_dict'`
- **错误位置**: `main.py` 第683行
- **原因**: `model_ema` 为 `None` 时仍尝试保存
- **修复**: 添加了 `None` 检查

**修复代码**:
```python
checkpoint_dict = {
    "model": model_without_ddp.state_dict(),
    "optimizer": optimizer.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
    "epoch": epoch,
    "scaler": loss_scaler.state_dict(),
    "args": args,
}
if model_ema is not None:
    checkpoint_dict["model_ema"] = get_state_dict(model_ema)
utils.save_on_master(checkpoint_dict, checkpoint_path)
```

---

## 🚀 实验启动状态

### MoRe (WSSS) - VOC数据集
- **状态**: 🟢 **运行中**
- **PID**: 32237 (主进程)
- **启动时间**: 2025-01-12 16:26
- **日志文件**: `/root/Result Reproduction/MoRe/logs/train_voc_*.log`
- **输出目录**: `w_outputs/2026-01/voc_reproduce_voc_more_*/`
- **训练参数**: 20000 iterations, batch_size=4

### CDTR (WSOL) - CUB数据集
- **状态**: 🟢 **运行中**
- **PID**: 32348
- **启动方式**: nohup后台运行
- **启动时间**: 2025-01-12 16:26
- **日志文件**: `/root/Result Reproduction/CDTR/logs/train_cub_*.log`
- **输出目录**: `./output_cub/`
- **训练参数**: 50 epochs, batch_size=32

---

## 📊 GPU资源使用

- **GPU型号**: NVIDIA GeForce RTX 4090 D
- **总显存**: 24564 MiB
- **当前使用**: ~21019 MiB (86%)
- **GPU利用率**: 100% ✅
- **状态**: 两个实验共享GPU，正常运行

---

## 🔍 监控命令

### 实时查看训练日志
```bash
# MoRe训练日志
tail -f /root/Result\ Reproduction/MoRe/logs/train_voc_*.log

# CDTR训练日志
tail -f /root/Result\ Reproduction/CDTR/logs/train_cub_*.log
```

### 查看GPU使用情况
```bash
nvidia-smi
# 或实时监控
watch -n 1 nvidia-smi
```

### 查看训练进程
```bash
ps aux | grep -E "(train_voc|main.py.*CUB)" | grep -v grep
```

### 查看PID文件
```bash
cat /root/Result\ Reproduction/MoRe/.train_pid
cat /root/Result\ Reproduction/CDTR/.train_pid
```

---

## 🛑 停止实验

如果需要停止实验：

```bash
# 停止MoRe训练
kill $(cat /root/Result\ Reproduction/MoRe/.train_pid)

# 停止CDTR训练
kill $(cat /root/Result\ Reproduction/CDTR/.train_pid)

# 或使用启动脚本中的PID
kill 32237  # MoRe
kill 32348  # CDTR
```

---

## 📝 预计训练时间

### MoRe训练
- **总迭代数**: 20000 iterations
- **当前进度**: 刚开始（约0%）
- **预计时间**: 
  - 每个iteration约0.2-0.4秒
  - 总时间约1-2小时（取决于验证频率）

### CDTR训练
- **总epoch数**: 50 epochs
- **每个epoch**: 约34秒（184个batch）
- **预计总时间**: 约29分钟（已测试）

---

## 📂 输出文件位置

### MoRe输出
- **检查点**: `w_outputs/2026-01/voc_reproduce_voc_more_*/checkpoints/`
- **预测结果**: `w_outputs/2026-01/voc_reproduce_voc_more_*/predictions/`
- **日志**: `logs/train_voc_*.log`

### CDTR输出
- **检查点**: `./output_cub/model_epoch*.pth`
- **日志**: `logs/train_cub_*.log`

---

## ⚠️ 重要提示

1. **后台运行**: 实验使用 `nohup` 在后台运行，即使SSH断开也会继续运行
2. **GPU共享**: 两个实验共享GPU，GPU利用率100%表示训练正常
3. **日志监控**: 建议定期查看日志，确认训练正常进行
4. **检查点**: 训练过程中会自动保存检查点，可用于恢复训练

---

## 🔄 重新启动实验

如果需要重新启动实验，使用启动脚本：

```bash
cd /root/Result\ Reproduction
bash start_experiments.sh
```

---

**实验状态**: ✅ **两个实验均在后台正常运行中！**

*最后更新: 2025-01-12 16:26*
