# 分布式训练指南 (Distributed Data Parallel)

本指南介绍如何使用 PyTorch DDP 进行 Temporal Encoder 的分布式训练。

## 目录

1. [概述](#概述)
2. [环境要求](#环境要求)
3. [快速开始](#快速开始)
4. [配置说明](#配置说明)
5. [断点续训](#断点续训)
6. [多机训练](#多机训练)
7. [性能优化](#性能优化)
8. [故障排查](#故障排查)

---

## 概述

### DDP vs 单GPU训练

**优势**:
- **训练速度**: 4卡可提升约3.5-3.8倍速度 (考虑通信开销)
- **大batch size**: 有效batch size = `batch_size × num_gpus`
- **更快收敛**: 更大的batch size可能带来更稳定的梯度

**开销**:
- **通信成本**: 每个step需要同步梯度 (~5-10% 开销)
- **内存**: 每个GPU都需要加载完整模型

### 关键实现特性

1. **DistributedSampler**: 自动将数据分片到各个GPU
2. **DDP模型包装**: 自动处理梯度同步
3. **Rank 0保存**: 只有主进程保存checkpoint和日志
4. **每轮自动保存**: 每个epoch结束后自动保存checkpoint
5. **断点续训**: 支持从任意checkpoint恢复训练

---

## 环境要求

### 硬件要求

- **单机多卡**: 2-8张GPU (推荐4张或8张)
- **多机多卡**: 每台机器2-8张GPU，机器间需要高速网络 (InfiniBand或10GbE+)
- **显存**: 每张GPU至少16GB (Qwen2.5-0.5B) 或 24GB (Qwen2.5-7B)

### 软件要求

```bash
# PyTorch >= 2.0 with CUDA support
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers
pip install transformers>=4.30.0

# 其他依赖
pip install pyyaml tqdm tensorboard
```

### 验证环境

```bash
# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 检查NCCL (DDP后端)
python -c "import torch; print(f'NCCL available: {torch.distributed.is_nccl_available()}')"
```

---

## 快速开始

### 单机多卡训练

#### 1. 准备配置文件

使用现有的 `configs/pretrain_temporal.yaml`，或者创建自定义配置。

**关键参数调整**:
```yaml
data:
  batch_size: 16  # 每张GPU的batch size (总batch = 16 × 4 = 64)
  num_workers: 4  # 每张GPU的数据加载线程数

training:
  num_epochs: 10
  learning_rate: 1e-4
  gradient_accumulation_steps: 1  # DDP下通常不需要梯度累积
```

#### 2. 启动训练 (4卡示例)

```bash
# 基础训练
torchrun --nproc_per_node=4 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml

# 指定输出目录
torchrun --nproc_per_node=4 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml \
    --output_dir checkpoints/ddp_run1
```

#### 3. 监控训练

```bash
# 在另一个终端查看TensorBoard
tensorboard --logdir logs/temporal_pretrain/tensorboard

# 查看日志
tail -f logs/temporal_pretrain/train.log
```

---

## 配置说明

### Batch Size 计算

**有效 Batch Size**:
```
effective_batch_size = batch_size × num_gpus × gradient_accumulation_steps
```

**示例**:
- 单卡: `batch_size=16` → effective = 16
- 4卡DDP: `batch_size=16` → effective = 64
- 4卡DDP + 梯度累积2: `batch_size=16` → effective = 128

**建议**:
- Qwen2.5-0.5B: effective_batch_size = 64-128
- Qwen2.5-7B: effective_batch_size = 32-64

### 学习率缩放

DDP训练时，有效batch size增大，需要调整学习率:

**线性缩放规则** (推荐):
```
lr_ddp = lr_single × sqrt(num_gpus)
```

**示例**:
- 单卡: `lr=1e-4`
- 4卡DDP: `lr=2e-4` (1e-4 × 2)
- 8卡DDP: `lr=2.83e-4` (1e-4 × 2.83)

**配置修改**:
```yaml
training:
  learning_rate: 2e-4  # 从1e-4调整到2e-4 (4卡)
```

### 数据加载优化

```yaml
data:
  num_workers: 4      # 每张GPU的worker数量
  pin_memory: true    # 启用内存锁定 (加速数据传输)
  prefetch_factor: 2  # 预取batch数量
```

**最佳实践**:
- `num_workers`: 2-4 (根据CPU核心数调整)
- `pin_memory`: 总是启用 (GPU训练)
- 避免 `num_workers` 过大导致CPU瓶颈

---

## 断点续训

### 自动checkpoint保存

训练脚本会在每个epoch结束后自动保存checkpoint:

```
checkpoints/temporal_pretrain/
├── epoch_1_step_1000.pt
├── epoch_2_step_2000.pt
├── epoch_3_step_3000.pt
├── best_model.pt          # 验证loss最低的模型
└── ...
```

### 从checkpoint恢复

```bash
torchrun --nproc_per_node=4 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

**恢复内容**:
- ✅ 模型参数
- ✅ 优化器状态 (Adam的momentum和variance)
- ✅ 学习率调度器状态
- ✅ Epoch和step计数器

**注意事项**:
1. 恢复训练时必须使用**相同数量的GPU**
2. 使用相同的配置文件 (或至少batch_size相同)
3. 如果配置改变，优化器状态可能不兼容

### 手动指定开始epoch

如果需要从特定epoch重新开始:

```python
# 修改 pretrain_temporal_encoder_ddp.py
if args.resume:
    checkpoint = torch.load(args.resume)
    # ... 加载模型 ...
    start_epoch = 10  # 手动指定
```

---

## 多机训练

### 网络配置

**要求**:
- 所有节点必须在同一网络
- 节点间可以通过IP地址互相访问
- 开放端口 (默认29500)

**验证连通性**:
```bash
# 在Node 1上ping Node 0
ping 192.168.1.1
```

### 启动命令

#### Node 0 (Master节点)

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml
```

#### Node 1 (Worker节点)

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml
```

**参数说明**:
- `--nproc_per_node`: 每个节点的GPU数量
- `--nnodes`: 总节点数
- `--node_rank`: 当前节点的rank (从0开始)
- `--master_addr`: Master节点的IP地址
- `--master_port`: 通信端口 (所有节点相同)

### 多机断点续训

```bash
# Node 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr="192.168.1.1" --master_port=29500 \
    script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

---

## 性能优化

### 1. 混合精度训练 (FP16)

**配置**:
```yaml
training:
  fp16: true  # 启用FP16
```

**优势**:
- 显存占用减半 (可增大batch size)
- 速度提升约1.5-2倍 (GPU支持Tensor Cores)

**注意事项**:
- 需要GPU支持 (V100/A100/RTX 30/40系列)
- 可能影响数值稳定性 (Temporal Encoder通常问题不大)

### 2. 梯度累积 (极大模型)

如果单卡显存不足以容纳batch_size=16:

```yaml
data:
  batch_size: 8  # 减半

training:
  gradient_accumulation_steps: 2  # 累积2步
```

有效batch = 8 × 4 GPUs × 2 steps = 64

### 3. Gradient Checkpointing (节省显存)

在 `TemporalLLMWrapper` 中启用:

```python
# src/context/temporal_llm_wrapper.py
from torch.utils.checkpoint import checkpoint

def forward(self, ...):
    # 使用checkpoint包装计算密集的部分
    v_temporal = checkpoint(self.temporal_encoder.forward_batch, curve)
    ...
```

**权衡**:
- 显存节省: 30-50%
- 速度降低: 15-25% (需要重计算)

### 4. 数据预处理缓存

如果数据生成慢，可以预先生成并缓存:

```bash
# 预生成训练数据
python tools/pregenerate_temporal_data.py \
    --num_samples 100000 \
    --output_dir data/temporal_pretrain/cache
```

然后修改 `TemporalPretrainDataset` 从缓存加载。

---

## 故障排查

### 常见错误

#### 1. NCCL错误: "Operation not permitted"

**原因**: 防火墙阻止了NCCL通信

**解决**:
```bash
# 临时关闭防火墙 (仅测试用)
sudo systemctl stop firewalld

# 或者开放端口
sudo firewall-cmd --add-port=29500/tcp --permanent
sudo firewall-cmd --reload
```

#### 2. "CUDA out of memory"

**原因**: batch_size过大或模型太大

**解决**:
1. 减小batch_size: `16 → 8 或 4`
2. 启用FP16: `fp16: true`
3. 启用gradient checkpointing
4. 减小 `num_workers`

#### 3. "Address already in use"

**原因**: 端口29500被占用

**解决**:
```bash
# 查找占用进程
lsof -i:29500

# 杀死进程
kill -9 <PID>

# 或者换一个端口
--master_port=29501
```

#### 4. Hang在初始化阶段

**原因**: 节点间网络不通或rank配置错误

**解决**:
1. 检查网络连通性
2. 确认所有节点的 `--nnodes`, `--master_addr` 一致
3. 确认每个节点的 `--node_rank` 唯一且连续

#### 5. Loss不同步/NaN

**原因**: 学习率过大或数值不稳定

**解决**:
1. 降低学习率: `2e-4 → 1e-4`
2. 启用梯度裁剪: `gradient_clip: 1.0`
3. 检查数据是否有NaN
4. 如果使用FP16，尝试切换回FP32

### 调试技巧

#### 1. 单进程调试模式

先在单GPU上测试:

```bash
python script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml
```

如果报错 "DDP environment not properly set up"，说明需要用torchrun启动。

#### 2. 增加日志详细度

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

torchrun --nproc_per_node=4 script/pretrain_temporal_encoder_ddp.py ...
```

#### 3. 验证数据分片

在训练脚本中添加:

```python
# 在train_epoch中
logger.info(f"[Rank {rank}] Processing {len(dataloader)} batches")

# 确保每个rank处理的batch数量相同
```

---

## 性能基准

### 单机多卡 (A100-40GB)

| GPUs | Batch/GPU | Effective Batch | Samples/sec | Speedup |
|------|-----------|-----------------|-------------|---------|
| 1    | 16        | 16              | 15          | 1.0x    |
| 2    | 16        | 32              | 28          | 1.87x   |
| 4    | 16        | 64              | 54          | 3.6x    |
| 8    | 16        | 128             | 102         | 6.8x    |

### 多机多卡 (2节点, 每节点4×A100)

| Setup        | Effective Batch | Samples/sec | Speedup |
|--------------|-----------------|-------------|---------|
| 单机4卡       | 64              | 54          | 1.0x    |
| 2机8卡       | 128             | 96          | 1.78x   |

**注**: 多机speedup受网络带宽限制

---

## 最佳实践总结

1. **从单机开始**: 先在单机上验证代码正确性
2. **监控GPU利用率**: 使用 `nvidia-smi dmon` 确保GPU充分利用
3. **调整batch size**: 让每张GPU的利用率达到80%+
4. **定期保存checkpoint**: 每个epoch自动保存 (已实现)
5. **验证收敛性**: DDP和单GPU应该收敛到相似的loss
6. **使用FP16**: 在支持的GPU上启用混合精度
7. **网络优化**: 多机训练时使用高速网络 (InfiniBand)

---

## 参考资料

- [PyTorch DDP官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL性能优化](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)
- [torchrun文档](https://pytorch.org/docs/stable/elastic/run.html)
