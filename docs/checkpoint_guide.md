# Checkpoint 保存与恢复指南

本指南介绍如何使用 checkpoint 功能进行断点续训。

## 目录

1. [概述](#概述)
2. [自动保存机制](#自动保存机制)
3. [恢复训练](#恢复训练)
4. [Checkpoint内容](#checkpoint内容)
5. [最佳实践](#最佳实践)

---

## 概述

Temporal Encoder 训练脚本支持两种类型的 checkpoint:

1. **Epoch Checkpoint**: 每个 epoch 结束后自动保存
2. **Best Model**: 验证 loss 最低时保存

两种训练模式都支持:
- ✅ 单GPU训练 (`pretrain_temporal_encoder.py`)
- ✅ 分布式训练 (`pretrain_temporal_encoder_ddp.py`)

---

## 自动保存机制

### 保存位置

默认保存到 `checkpoints/temporal_pretrain/`:

```
checkpoints/temporal_pretrain/
├── epoch_1_step_1000.pt      # Epoch 1结束
├── epoch_2_step_2000.pt      # Epoch 2结束
├── epoch_3_step_3000.pt      # Epoch 3结束
├── ...
├── epoch_10_step_10000.pt    # Epoch 10结束
└── best_model.pt             # 验证loss最低的模型
```

### 保存时机

**每个 Epoch 结束后**:
1. 完成训练
2. 完成验证
3. 自动保存 checkpoint

**验证 Loss 改善时**:
- 额外保存一份 `best_model.pt`

### 配置保存路径

在 `configs/pretrain_temporal.yaml` 中:

```yaml
paths:
  output_dir: "checkpoints/temporal_pretrain"
  log_dir: "logs/temporal_pretrain"
  tensorboard_dir: "logs/temporal_pretrain/tensorboard"
```

或者通过命令行参数覆盖:

```bash
python script/pretrain_temporal_encoder.py \
    --config configs/pretrain_temporal.yaml \
    --output_dir checkpoints/my_custom_run
```

---

## 恢复训练

### 单GPU训练恢复

```bash
python script/pretrain_temporal_encoder.py \
    --config configs/pretrain_temporal.yaml \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

### 分布式训练恢复

```bash
torchrun --nproc_per_node=4 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

### 恢复行为

训练将从 checkpoint 的下一个 epoch 开始:

**示例**: 从 `epoch_5_step_5000.pt` 恢复
- 加载的 epoch: 5 (已完成)
- 开始训练: Epoch 6
- 继续训练直到配置的总 epoch 数

**日志输出**:
```
INFO - Resuming from checkpoint: checkpoints/temporal_pretrain/epoch_5_step_5000.pt
INFO - Resumed from epoch 6, step 5000
INFO - Starting training
INFO - Epoch 6/10
...
```

---

## Checkpoint内容

### 完整状态保存

每个 checkpoint 包含:

```python
{
    'epoch': 5,                          # 已完成的epoch编号
    'global_step': 5000,                 # 全局训练步数
    'model_state_dict': {...},           # 模型参数
    'optimizer_state_dict': {...},       # 优化器状态 (Adam momentum等)
    'scheduler_state_dict': {...},       # 学习率调度器状态
    'train_loss': 1.234,                 # 该epoch的训练loss
    'val_loss': 1.456                    # 该epoch的验证loss
}
```

### 模型参数

**单GPU**:
```python
checkpoint['model_state_dict']  # 直接是模型参数
```

**DDP**:
```python
checkpoint['model_state_dict']  # model.module的参数 (已去除DDP wrapper)
```

### 优化器状态

保存 AdamW 的内部状态:
- Momentum buffers (first moment estimates)
- Variance buffers (second moment estimates)
- Step count

**重要**: 恢复训练时会继续使用这些状态，确保优化平滑过渡。

### 学习率调度器状态

保存当前学习率和 warmup 进度:
- 当前 step 计数
- Last learning rate

---

## 最佳实践

### 1. 定期检查 Checkpoint

```bash
# 列出所有checkpoint
ls -lh checkpoints/temporal_pretrain/

# 查看checkpoint信息
python -c "
import torch
ckpt = torch.load('checkpoints/temporal_pretrain/epoch_5_step_5000.pt')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Step: {ckpt[\"global_step\"]}')
print(f'Train Loss: {ckpt[\"train_loss\"]:.4f}')
print(f'Val Loss: {ckpt[\"val_loss\"]:.4f}')
"
```

### 2. 磁盘空间管理

Checkpoint 文件通常较大 (取决于模型):
- Qwen2.5-0.5B: ~500MB/checkpoint
- Qwen2.5-7B: ~5GB/checkpoint

**清理策略**:

```bash
# 只保留最近3个epoch的checkpoint
cd checkpoints/temporal_pretrain
ls -t epoch_*.pt | tail -n +4 | xargs rm

# 或者使用脚本自动清理
python tools/cleanup_old_checkpoints.py \
    --checkpoint_dir checkpoints/temporal_pretrain \
    --keep_last 3
```

### 3. 备份 Best Model

```bash
# 定期备份最佳模型
cp checkpoints/temporal_pretrain/best_model.pt \
   backups/best_model_$(date +%Y%m%d).pt
```

### 4. 验证 Checkpoint 完整性

```python
# tools/verify_checkpoint.py
import torch
import sys

checkpoint_path = sys.argv[1]

try:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    required_keys = ['epoch', 'global_step', 'model_state_dict', 
                     'optimizer_state_dict', 'scheduler_state_dict']
    
    for key in required_keys:
        if key not in ckpt:
            print(f"✗ Missing key: {key}")
            sys.exit(1)
    
    print(f"✓ Checkpoint is valid")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Step: {ckpt['global_step']}")
    print(f"  Train Loss: {ckpt['train_loss']:.4f}")
    print(f"  Val Loss: {ckpt['val_loss']:.4f}")
    
except Exception as e:
    print(f"✗ Checkpoint is corrupted: {str(e)}")
    sys.exit(1)
```

### 5. 从特定Step恢复 (高级)

如果需要从epoch中间恢复 (需要额外实现):

```python
# 修改训练脚本以支持step级别的保存
if global_step % config['training'].get('save_step_interval', 1000) == 0:
    checkpoint_path = output_dir / f"step_{global_step}.pt"
    save_checkpoint(...)
```

### 6. 跨设备迁移

**从单GPU切换到DDP**:
```bash
# 单GPU训练
python script/pretrain_temporal_encoder.py ...

# 切换到4卡DDP继续训练
torchrun --nproc_per_node=4 \
    script/pretrain_temporal_encoder_ddp.py \
    --resume checkpoints/temporal_pretrain/epoch_3_step_3000.pt
```

**注意**: 
- ✅ 模型参数完全兼容
- ✅ 优化器状态可以复用
- ⚠️ 学习率调度器可能需要调整 (总step数改变)

---

## 故障恢复场景

### 场景1: 训练中断 (断电/OOM)

**症状**: 训练突然停止，没有保存最新checkpoint

**解决**: 从最近的epoch checkpoint恢复
```bash
# 查找最新checkpoint
ls -t checkpoints/temporal_pretrain/epoch_*.pt | head -n 1

# 恢复训练
python script/pretrain_temporal_encoder.py \
    --resume checkpoints/temporal_pretrain/epoch_8_step_8000.pt
```

**损失**: 最多损失1个epoch的训练进度

### 场景2: 发现配置错误

**症状**: 训练5个epoch后发现学习率设置错误

**解决**: 
1. 修改配置文件
2. 从checkpoint恢复 (但不加载优化器状态)

```python
# 修改训练脚本
if args.resume and args.reset_optimizer:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 不加载optimizer和scheduler，使用新配置
```

### 场景3: Checkpoint损坏

**症状**: 加载checkpoint时报错

**解决**:
1. 尝试加载前一个checkpoint
2. 验证checkpoint完整性 (见上面的验证脚本)

```bash
# 验证所有checkpoint
for ckpt in checkpoints/temporal_pretrain/epoch_*.pt; do
    echo "Checking $ckpt..."
    python tools/verify_checkpoint.py $ckpt
done
```

### 场景4: 多机训练某节点失败

**症状**: DDP训练时某个节点掉线

**解决**:
1. 重启失败的节点
2. 所有节点从同一checkpoint恢复
3. 确保使用相同的 `--resume` 参数

```bash
# 所有节点执行相同命令
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=X \
    script/pretrain_temporal_encoder_ddp.py \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

---

## 高级功能

### 提取模型权重 (仅推理使用)

```python
# tools/extract_model_only.py
import torch

# 加载完整checkpoint
checkpoint = torch.load('checkpoints/temporal_pretrain/best_model.pt')

# 只保存模型权重
model_only = {
    'model_state_dict': checkpoint['model_state_dict'],
    'epoch': checkpoint['epoch'],
    'val_loss': checkpoint['val_loss']
}

torch.save(model_only, 'models/temporal_encoder_inference.pt')
```

### 合并多个Checkpoint (实验对比)

```python
# 对比不同epoch的性能
checkpoints = [
    'epoch_5_step_5000.pt',
    'epoch_7_step_7000.pt',
    'epoch_10_step_10000.pt'
]

for ckpt_path in checkpoints:
    ckpt = torch.load(f'checkpoints/temporal_pretrain/{ckpt_path}')
    print(f"{ckpt_path}:")
    print(f"  Train Loss: {ckpt['train_loss']:.4f}")
    print(f"  Val Loss: {ckpt['val_loss']:.4f}")
```

---

## 总结

**关键点**:
1. ✅ 每个epoch自动保存checkpoint
2. ✅ 使用 `--resume` 参数恢复训练
3. ✅ Checkpoint包含完整训练状态
4. ✅ 单GPU和DDP都支持
5. ✅ 支持跨设备迁移 (单GPU ↔ DDP)

**建议**:
- 定期备份 `best_model.pt`
- 保留最近3-5个epoch checkpoint
- 在长时间训练前验证checkpoint功能
- 使用 TensorBoard 监控loss曲线，决定从哪个checkpoint恢复
