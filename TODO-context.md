# TODO: Temporal Encoder 模态对齐预训练实现

## 概述

本文档描述 `TemporalEncoder` (1D-CNN) 的模态对齐预训练完整实现，目标是训练 CNN 将系统资源时序曲线"翻译"成自然语言描述，使 LLM 能够理解资源约束。

**训练目标**: 强迫 1D-CNN 学会将资源曲线编码为 LLM 可理解的嵌入空间，而无需涉及具体用户任务。

**核心架构**:
- **Encoder (可训练)**: `TemporalCNN` + `MLP Projector` (映射到 Qwen2.5 嵌入维度 3584)
- **LLM (冻结)**: Qwen2.5-0.5B/7B (仅用于解码，不更新参数)
- **Loss**: Causal Language Modeling Loss (Next Token Prediction)

---

## 模块一: 合成数据生成器 ✅ 已完成

### 1.1 数据生成器基础设施 ✅ 已完成

**文件**: `src/data/temporal_pretrain_dataset.py`

**目的**: 创建 `TemporalPretrainDataset` 类，生成 `(Resource_Timeline, Text_Description)` 训练对。

**实现要点**:

#### 1.1.1 资源曲线合成器 `ResourceCurveSynthesizer` ✅ 已完成

**文件**: `src/data/resource_curve_synthesizer.py`

**功能**: 生成多样化的合成资源时序曲线。

**参数**:
- `num_timesteps`: 时间步数量，范围 [20, 100]
- `time_granularity_ms`: 时间粒度，默认 100ms
- `resource_types`: 4 种资源 `[cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]`

**曲线生成策略**:

1. **趋势模式 (Trend Patterns)**:
   - Linear: 线性增长/下降
   - Exponential: 指数增长/衰减
   - Sinusoidal: 正弦波动
   - Step: 阶跃变化
   - Plateau: 平台期后突变

2. **噪声注入**:
   - Gaussian noise: 添加高斯噪声模拟真实波动
   - Spike injection: 注入随机尖峰模拟资源抢占

3. **物理约束**:
   - 确保所有值在合理范围内 (如 GPU SM 在 0-100%)
   - 添加边界限制防止越界

**输出格式**: `torch.Tensor` 形状 `(num_timesteps, 4)`, 每列对应一种资源。

#### 1.1.2 文本描述生成器 `TextDescriptionGenerator` ✅ 已完成

**文件**: `src/data/text_description_generator.py`

**功能**: 根据资源曲线生成对应的自然语言描述。

**三种任务类型**:

##### Type A: 趋势描述 (Trend Description)

**意图**: 训练 CNN 大尺度卷积核捕获宏观趋势。

**实现逻辑**:
1. 计算资源的起始值和结束值
2. 判断趋势类型 (increasing/decreasing/stable)
3. 计算变化幅度和速率
4. 生成描述模板

**Prompt 模板**:
```
Analyze the [RESOURCE_TYPE] trend over the next [DURATION] seconds.
```

**Target 模板**:
```
[RESOURCE_TYPE] shows a [TREND_TYPE] trend, [CHANGE_DESCRIPTION] from [START_VALUE] to [END_VALUE] over [DURATION] seconds.
```

**变量替换规则**:
- `RESOURCE_TYPE`: "GPU memory", "CPU cores", etc.
- `TREND_TYPE`: "increasing", "decreasing", "stable"
- `CHANGE_DESCRIPTION`: "rising steadily", "dropping sharply", "fluctuating"
- `START_VALUE`, `END_VALUE`: 格式化数值 (如 "8000MB", "16 cores")
- `DURATION`: 时间窗口长度 (秒)

**多样性策略**:
- 随机选择 5-10 种不同的描述词汇
- 随机选择不同的句式结构
- 30% 概率添加定量描述 (如 "at a rate of 200MB/s")

##### Type B: 瓶颈定位 (Bottleneck Spotting)

**意图**: 训练 CNN 小尺度卷积核捕获瞬时资源波动。

**实现逻辑**:
1. 在曲线中注入 1-3 个深度下降的 Spike
2. 记录 Spike 的位置 (时间点) 和最小值
3. 生成精确定位描述

**Prompt 模板**:
```
Identify the minimum available [RESOURCE_TYPE] in the timeline.
```

**Target 模板**:
```
The minimum available [RESOURCE_TYPE] drops to [MIN_VALUE] at t=[TIME_POINT]s.
```

**Spike 生成规则**:
- 深度: 下降至正常值的 20%-50%
- 持续时间: 1-5 个时间步 (0.1-0.5秒)
- 位置: 随机分布，但避免边界 (前后各留 10% 缓冲)

**多样性策略**:
- 单资源单 Spike vs 多资源多 Spike
- 描述中 50% 概率添加持续时间信息

##### Type C: 可行性判别 (Feasibility QA)

**意图**: 训练 CNN 编码数值逻辑和时间逻辑。

**实现逻辑**:
1. 随机生成一个资源需求 (如 "4096MB GPU memory")
2. 检查曲线中是否有时间点满足需求
3. 如果满足，返回最早可用时间；否则返回不可行

**Prompt 模板**:
```
Can a task requiring [REQUIREMENT] be scheduled immediately?
```

**Target 模板 (可行)**:
```
Yes, [RESOURCE_TYPE] has [AVAILABLE_VALUE] available, which exceeds the required [REQUIREMENT].
```

**Target 模板 (延迟可行)**:
```
No, current available [RESOURCE_TYPE] is [CURRENT_VALUE]. It will become available after t=[TIME_POINT]s when it reaches [FUTURE_VALUE].
```

**Target 模板 (不可行)**:
```
No, the required [REQUIREMENT] exceeds the maximum available [RESOURCE_TYPE] ([MAX_VALUE]) throughout the timeline.
```

**需求生成策略**:
- 从曲线的平均值生成 0.8x, 1.2x, 1.5x 倍数的需求
- 确保 33% 立即可行, 33% 延迟可行, 33% 不可行

### 1.2 数据集类实现 ✅ 已完成

**文件**: `src/data/temporal_pretrain_dataset.py`

**类**: `TemporalPretrainDataset(torch.utils.data.Dataset)`

**初始化参数**:
- `num_samples`: 总样本数 (建议 50,000 - 200,000)
- `type_distribution`: 三种任务类型的分布比例 `{"A": 0.4, "B": 0.3, "C": 0.3}`
- `curve_synthesizer`: 资源曲线合成器实例
- `text_generator`: 文本描述生成器实例
- `tokenizer`: Qwen2.5 Tokenizer
- `max_length`: 文本最大长度 (默认 256)

**`__getitem__` 返回**:
```python
{
    'curve': torch.Tensor,        # (num_timesteps, 4)
    'prompt_ids': torch.Tensor,   # (prompt_len,)
    'target_ids': torch.Tensor,   # (target_len,)
    'attention_mask': torch.Tensor,
    'labels': torch.Tensor,       # target_ids with prompt masked as -100
    'task_type': str              # "A", "B", or "C"
}
```

**数据增强**:
- 时间步数量随机化 [20, 100]
- 资源范围随机缩放 (0.5x - 1.5x)
- 文本模板随机选择

### 1.3 数据验证工具 ✅ 已完成

**脚本**: `tools/validate_temporal_dataset.py`

**功能**:
1. 可视化随机抽样的资源曲线
2. 打印对应的 Prompt 和 Target
3. 统计任务类型分布
4. 检查文本长度分布
5. 验证物理约束是否满足

**输出**: 生成 `data/temporal_pretrain/validation_report.html` 可视化报告。

---

## 模块二: 训练框架 ✅ 已完成

### 2.1 模型架构集成 ✅ 已完成

**文件**: `src/context/temporal_llm_wrapper.py`

**目的**: 创建 `TemporalLLMWrapper` 类，集成 TemporalEncoder 和冻结的 LLM。

**架构组件**:

```python
class TemporalLLMWrapper(nn.Module):
    def __init__(self, 
                 temporal_encoder: TemporalEncoder,
                 llm_model: AutoModelForCausalLM,
                 llm_embedding_dim: int = 3584):
        """
        Args:
            temporal_encoder: 可训练的 TemporalEncoder (包含 CNN + Projector)
            llm_model: 冻结的 Qwen2.5 模型
            llm_embedding_dim: LLM 嵌入维度 (Qwen2.5-7B: 3584, 0.5B: 1024)
        """
```

**组件说明**:

1. **TemporalEncoder (可训练)**:
   - `TemporalCNN`: 已在 `temporal_encoder.py` 中实现
   - `MLP Projector`: 添加到 `TemporalEncoder.__init__` 中
     ```python
     self.projector = nn.Sequential(
         nn.Linear(cnn_config['output_dim'], llm_embedding_dim),
         nn.GELU(),
         nn.Linear(llm_embedding_dim, llm_embedding_dim)
     )
     ```

2. **LLM (冻结)**:
   - 加载预训练 Qwen2.5 模型
   - 冻结所有参数: `for param in llm_model.parameters(): param.requires_grad = False`
   - 设置为 eval 模式

**Forward 流程**:

```python
def forward(self, 
            curve: torch.Tensor,      # (batch, num_timesteps, 4)
            prompt_ids: torch.Tensor, # (batch, prompt_len)
            target_ids: torch.Tensor, # (batch, target_len)
            attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        loss: Causal LM loss
    """
    # 1. Encode temporal curve
    v_temporal = self.temporal_encoder.forward_batch(curve)  # (batch, output_dim)
    v_temporal = self.projector(v_temporal)                   # (batch, llm_embedding_dim)
    
    # 2. Get LLM embeddings for prompt
    prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)  # (batch, prompt_len, llm_dim)
    
    # 3. Prepend temporal embedding as a special token
    # Option A: 作为第一个 token
    combined_embeds = torch.cat([
        v_temporal.unsqueeze(1),  # (batch, 1, llm_dim)
        prompt_embeds             # (batch, prompt_len, llm_dim)
    ], dim=1)
    
    # 4. Forward through LLM
    outputs = self.llm_model(
        inputs_embeds=combined_embeds,
        attention_mask=self._extend_attention_mask(attention_mask),
        labels=self._prepare_labels(target_ids, prompt_len=prompt_ids.size(1) + 1)
    )
    
    return outputs.loss
```

**关键实现细节**:

- **Temporal Token 注入位置**: 作为序列的第一个 token (类似 CLS token)
- **Attention Mask 扩展**: 在 prompt 前添加一个位置对应 temporal token
- **Labels 准备**: 只计算 target 部分的 loss，prompt 部分 mask 为 -100

### 2.2 训练脚本 ✅ 已完成

**文件**: `script/pretrain_temporal_encoder.py`

**功能**: 完整的训练流程，包括数据加载、训练循环、验证和检查点保存。

**参数配置** (从 `configs/pretrain_temporal.yaml` 加载):

```yaml
model:
  llm_name: "Qwen/Qwen2.5-7B-Instruct"  # 或 Qwen2.5-7B
  llm_embedding_dim: 3584   # 0.5B: 1024, 7B: 3584
  temporal_encoder:
    hidden_channels: 64
    output_dim: 256
    num_layers: 3
    pooling: "adaptive_avg"

data:
  num_train_samples: 100000
  num_val_samples: 5000
  type_distribution:
    A: 0.4
    B: 0.3
    C: 0.3
  max_length: 256
  batch_size: 16

training:
  num_epochs: 10
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_clip: 1.0
  save_interval: 1000
  eval_interval: 500
  
optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]
  
scheduler:
  type: "cosine"
  min_lr: 1e-6
```

**训练循环实现**:

```python
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader):
        # Move to device
        curve = batch['curve'].to(device)
        prompt_ids = batch['prompt_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward
        loss = model(curve, prompt_ids, target_ids, attention_mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**检查点保存**:
- 保存路径: `assert/temporal_pretrain/epoch_{epoch}_step_{step}.pt`
- 保存内容:
  ```python
  {
      'epoch': epoch,
      'step': global_step,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'train_loss': train_loss,
      'val_loss': val_loss
  }
  ```

### 2.3 分布式训练支持 (可选) ✅ 已完成

**文件**: `script/pretrain_temporal_encoder_ddp.py`

**目的**: 使用 PyTorch DDP 加速多 GPU 训练。

**关键功能**:
1. ✅ 使用 `torchrun` 启动分布式训练
2. ✅ 用 `DistributedSampler` 包装 DataLoader
3. ✅ 用 `DDP` 包装模型
4. ✅ 只在 rank 0 保存检查点和日志
5. ✅ 每个 epoch 结束后自动保存 checkpoint
6. ✅ 支持通过 `--resume` 参数加载 checkpoint 断点续训
7. ✅ 支持单机多卡和多机多卡训练

**使用方法**:
```bash
# 单机4卡训练
torchrun --nproc_per_node=4 script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml

# 从checkpoint恢复训练
torchrun --nproc_per_node=4 script/pretrain_temporal_encoder_ddp.py \
    --config configs/pretrain_temporal.yaml \
    --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt

# 多机训练 (Node 0)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 \
    script/pretrain_temporal_encoder_ddp.py --config configs/pretrain_temporal.yaml
```

---

## 模块三: 验证与评估

### 3.1 定量评估

**文件**: `script/evaluate_temporal_encoder.py`

**指标**:

1. **Perplexity (困惑度)**:
   - 在验证集上计算平均 perplexity
   - 分任务类型统计 (Type A/B/C 各自的 perplexity)

2. **Token Accuracy**:
   - 计算生成的 token 与 ground truth 的匹配率
   - 统计前 K 个 token 的准确率 (K=5, 10, 20)

3. **Numerical Accuracy (数值准确性)**:
   - 从生成的文本中提取数值
   - 与 ground truth 中的数值比较
   - 容忍度: ±5% 相对误差

**实现**:

```python
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Forward (计算 loss)
            loss = model(...)
            total_loss += loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                curve=batch['curve'],
                prompt_ids=batch['prompt_ids'],
                max_length=256
            )
            
            # Calculate token accuracy
            correct = (generated_ids == batch['target_ids']).sum()
            correct_tokens += correct.item()
            total_tokens += batch['target_ids'].numel()
    
    perplexity = torch.exp(torch.tensor(total_loss / len(dataloader)))
    token_accuracy = correct_tokens / total_tokens
    
    return {
        'perplexity': perplexity,
        'token_accuracy': token_accuracy
    }
```

### 3.2 定性验证

**文件**: `tools/visualize_temporal_predictions.py`

**功能**:
1. 随机抽取 20 个验证样本
2. 可视化资源曲线
3. 显示 Prompt, Ground Truth Target, Model Prediction
4. 高亮差异部分

**输出**: 生成 `logs/temporal_pretrain/predictions_visualization.html` 报告。

### 3.3 消融实验 (Ablation Study)

**目的**: 验证设计选择的有效性。

**实验变量**:

1. **CNN 架构**:
   - 不同 kernel sizes: [3,5,7] vs [3,3,3] vs [7,7,7]
   - 不同层数: 2 layers vs 3 layers vs 4 layers
   - 不同 pooling: adaptive_avg vs adaptive_max vs flatten

2. **数据分布**:
   - 纯 Type A vs 纯 Type B vs 纯 Type C vs 混合
   - 不同样本数量: 10k vs 50k vs 100k

3. **Projector 深度**:
   - 单层 Linear vs 双层 MLP vs 无 Projector (直接匹配维度)

**记录**: 所有实验结果记录到 `logs/temporal_pretrain/ablation_results.json`

---

## 模块四: 工程化工具

### 4.1 配置文件 ✅ 已完成

**文件**: `configs/pretrain_temporal.yaml`

**内容**: 参见 2.2 节的参数配置。

### 4.2 日志系统 ✅ 已完成

**使用**: `tensorboard` + `wandb` (可选)

**记录内容**:
- 每个 step 的 loss
- 每个 epoch 的平均 loss
- 验证集 perplexity 和 accuracy
- 学习率变化曲线
- 梯度范数

**启动**:
```bash
tensorboard --logdir logs/temporal_pretrain/tensorboard
```

### 4.3 资源监控

**脚本**: `tools/monitor_training.py`

**功能**:
- 实时监控 GPU 使用率
- 记录内存占用
- 估计剩余训练时间
- 发送异常告警 (OOM, NaN loss)

### 4.4 快速测试脚本 ✅ 已完成

**文件**: `script/quick_test_temporal.py`

**目的**: 在小规模数据上快速验证代码正确性。

**配置**:
- 1000 个训练样本
- 100 个验证样本
- 2 个 epochs
- 使用 Qwen2.5-0.5B (节省显存)

**用途**: 在提交大规模训练前，验证代码无语法错误且能正常运行。

---

## 模块五: 实现顺序与依赖

### Phase 1: 数据基础设施 (优先级: 高) ✅ 已完成

1. ✅ 实现 `ResourceCurveSynthesizer` (模块 1.1.1) - 文件: `src/data/resource_curve_synthesizer.py`
2. ✅ 实现 `TextDescriptionGenerator` (模块 1.1.2) - 文件: `src/data/text_description_generator.py`
3. ✅ 实现 `TemporalPretrainDataset` (模块 1.2) - 文件: `src/data/temporal_pretrain_dataset.py`
4. ✅ 实现数据验证工具 (模块 1.3) - 文件: `tools/validate_temporal_dataset.py`
5. **验证**: 运行 `tools/validate_temporal_dataset.py` 确保数据质量

### Phase 2: 模型修改 (优先级: 高) ✅ 已完成

1. ✅ 修改 `TemporalEncoder` 添加 `projector` (模块 2.1) - 文件: `src/context/temporal_encoder.py`
2. ✅ 实现 `TemporalLLMWrapper` (模块 2.1) - 文件: `src/context/temporal_llm_wrapper.py`
3. ✅ 实现快速测试脚本 (模块 4.4) - 文件: `script/quick_test_temporal.py`
4. **验证**: 运行 `script/quick_test_temporal.py` 确保前向传播无错误

### Phase 3: 训练流程 (优先级: 高) ✅ 已完成

1. ✅ 创建配置文件 `configs/pretrain_temporal.yaml` (模块 4.1)
2. ✅ 实现训练脚本 `script/pretrain_temporal_encoder.py` (模块 2.2)
3. ✅ 实现日志系统集成 (模块 4.2) - TensorBoard集成
4. **验证**: 在小规模数据上运行 1 个 epoch

### Phase 4: 评估框架 (优先级: 中)

1. 实现定量评估脚本 (模块 3.1)
2. 实现定性可视化工具 (模块 3.2)
3. **验证**: 在训练后的检查点上运行评估

### Phase 5: 优化与扩展 (优先级: 低)

1. ✅ 实现分布式训练支持 (模块 2.3) - 文件: `script/pretrain_temporal_encoder_ddp.py`
2. 实现消融实验框架 (模块 3.3)
3. 实现资源监控工具 (模块 4.3)

---

## 模块六: 核心实现细节补充

### 6.1 时间轴对齐策略

**问题**: 合成曲线的时间长度不固定 [20, 100]，但 CNN 需要固定输入。

**解决方案 (在 `TemporalEncoder.forward` 中已处理)**:
- 使用 `adaptive_avg_pool` 自动处理不同长度
- 或在数据预处理阶段统一 resample 到固定长度 (如 50 步)

**推荐**: 保持动态长度，依赖 adaptive pooling。

### 6.2 Projector 训练稳定性

**技巧**:
1. **初始化**: 使用 Xavier 初始化 Projector 的权重
2. **Layer Normalization**: 在 Projector 输出后添加 LayerNorm
3. **梯度缩放**: 如果 LLM 的 embedding 梯度过大，对 Projector 的梯度乘以 0.1

**修改 Projector**:
```python
self.projector = nn.Sequential(
    nn.Linear(cnn_config['output_dim'], llm_embedding_dim),
    nn.GELU(),
    nn.Linear(llm_embedding_dim, llm_embedding_dim),
    nn.LayerNorm(llm_embedding_dim)  # 添加 LayerNorm
)
```

### 6.3 文本生成多样性保证

**问题**: 模板化生成可能导致文本过于单一。

**增强策略**:
1. **同义词替换**: 维护一个同义词词典
   - "increasing" → ["rising", "growing", "ascending", "climbing"]
   - "sharply" → ["rapidly", "steeply", "dramatically", "significantly"]

2. **句式变换**: 每种任务类型准备 5-10 种不同句式
   - Active voice vs Passive voice
   - 简单句 vs 复合句

3. **数值格式**: 随机选择不同表示方式
   - "4096 MB" vs "4.0 GB" vs "4 GB"
   - "16 cores" vs "16 CPU cores" vs "16 available cores"

**实现**: 在 `TextDescriptionGenerator` 中添加 `RandomSynonymReplacer` 和 `SentenceRewriter`。

### 6.4 物理约束验证器

**类**: `PhysicalConstraintValidator`

**功能**: 在数据生成后验证合理性。

**检查项**:
1. CPU cores: 0 ≤ value ≤ 128
2. CPU memory: 0 ≤ value ≤ 512 GB
3. GPU SM: 0 ≤ value ≤ 100%
4. GPU memory: 0 ≤ value ≤ 80 GB
5. 时间序列单调性: 时间戳必须严格递增
6. 突变幅度限制: 单步变化不超过 50% (除非是 Spike)

**集成**: 在 `TemporalPretrainDataset.__getitem__` 中调用验证器。

---

## 模块七: 预期训练结果

### 7.1 收敛指标

**训练曲线预期**:
- **Epoch 1-3**: Loss 快速下降，从 ~5.0 降至 ~2.0
- **Epoch 4-7**: 缓慢下降，从 ~2.0 降至 ~1.2
- **Epoch 8-10**: 趋于平稳，在 ~1.0 附近波动

**验证集表现**:
- Perplexity < 3.0 (优秀)
- Perplexity < 5.0 (可接受)
- Token Accuracy > 60% (优秀)

### 7.2 生成质量检查

**定性标准**:
1. **数值准确性**: 生成的数值与 ground truth 误差 < 10%
2. **时间一致性**: 生成的时间点在合理范围内
3. **趋势正确性**: 能正确描述 increasing/decreasing/stable
4. **流畅度**: 生成的文本语法正确，无明显错误

**案例测试**:
- 准备 10 个手工标注的测试样本
- 在每个 checkpoint 上生成预测
- 人工评分 (1-5 分) 并记录

### 7.3 失败模式分析

**常见问题**:
1. **数值幻觉**: 生成不存在的数值
   - **原因**: Projector 未充分训练
   - **解决**: 增加训练样本，特别是 Type C

2. **时间错乱**: 生成的时间点与曲线不符
   - **原因**: CNN 未学会时间编码
   - **解决**: 添加位置编码到 CNN 输入

3. **趋势误判**: 将 increasing 判断为 decreasing
   - **原因**: 大尺度卷积核未充分训练
   - **解决**: 增加 Type A 样本比例

---

## 模块八: 代码模块清单

### 新增文件

```
src/data/
  temporal_pretrain_dataset.py      # 数据集类 ✅
  resource_curve_synthesizer.py     # 曲线合成器 ✅
  text_description_generator.py     # 文本生成器 ✅
  
src/context/
  temporal_llm_wrapper.py            # LLM 包装器 ✅

script/
  pretrain_temporal_encoder.py      # 训练脚本 ✅
  pretrain_temporal_encoder_ddp.py  # 分布式训练 ✅
  evaluate_temporal_encoder.py      # 评估脚本
  quick_test_temporal.py            # 快速测试 ✅

tools/
  validate_temporal_dataset.py      # 数据验证 ✅
  visualize_temporal_predictions.py # 预测可视化
  monitor_training.py               # 训练监控

configs/
  pretrain_temporal.yaml            # 训练配置 ✅
```

### 修改文件

```
src/context/temporal_encoder.py ✅
  - ✅ 添加 projector 到 TemporalEncoder.__init__
  - ✅ 添加 forward_batch 方法处理批量输入
  - ✅ 添加 forward_with_projection 方法用于投影
```

### 数据目录

```
data/temporal_pretrain/
  train/                            # 训练数据缓存 (可选)
  val/                              # 验证数据缓存 (可选)
  validation_report.html            # 数据验证报告
  
checkpoints/temporal_pretrain/
  epoch_1_step_1000.pt
  epoch_2_step_2000.pt
  ...
  
logs/temporal_pretrain/
  tensorboard/                      # TensorBoard 日志
  predictions_visualization.html    # 预测可视化
  ablation_results.json            # 消融实验结果
```

---

## 模块九: 运行命令

### 9.1 数据验证

```bash
python tools/validate_temporal_dataset.py \
  --num_samples 1000 \
  --output_dir data/temporal_pretrain
```

### 9.2 快速测试

```bash
python script/quick_test_temporal.py \
  --config configs/pretrain_temporal.yaml \
  --num_train 1000 \
  --num_val 100 \
  --epochs 2
```

### 9.3 完整训练

**单GPU训练**:
```bash
python script/pretrain_temporal_encoder.py \
  --config configs/pretrain_temporal.yaml \
  --output_dir checkpoints/temporal_pretrain \
  --log_dir logs/temporal_pretrain

# 从checkpoint恢复训练
python script/pretrain_temporal_encoder.py \
  --config configs/pretrain_temporal.yaml \
  --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

### 9.4 分布式训练 ✅ 已实现

**单机多卡 (4 GPUs)**:
```bash
torchrun --nproc_per_node=4 \
  script/pretrain_temporal_encoder_ddp.py \
  --config configs/pretrain_temporal.yaml

# 从checkpoint恢复训练
torchrun --nproc_per_node=4 \
  script/pretrain_temporal_encoder_ddp.py \
  --config configs/pretrain_temporal.yaml \
  --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

**多机多卡 (2节点, 每节点4卡)**:
```bash
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr="192.168.1.1" --master_port=29500 \
  script/pretrain_temporal_encoder_ddp.py \
  --config configs/pretrain_temporal.yaml

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr="192.168.1.1" --master_port=29500 \
  script/pretrain_temporal_encoder_ddp.py \
  --config configs/pretrain_temporal.yaml
```

### 9.5 评估

```bash
python script/evaluate_temporal_encoder.py \
  --checkpoint checkpoints/temporal_pretrain/epoch_10_step_10000.pt \
  --num_samples 5000 \
  --output_dir logs/temporal_pretrain
```

### 9.6 可视化预测

```bash
python tools/visualize_temporal_predictions.py \
  --checkpoint checkpoints/temporal_pretrain/epoch_10_step_10000.pt \
  --num_samples 20 \
  --output logs/temporal_pretrain/predictions_visualization.html
```

---

## 模块十: 关键技术决策说明

### 10.1 为何冻结 LLM?

**原因**:
1. **参数效率**: LLM 有数十亿参数，全量微调需要大量显存
2. **知识保留**: 冻结 LLM 保留其预训练的语言能力
3. **训练稳定性**: 只训练 Encoder 降低训练复杂度

**理论依据**: 
- Prefix-Tuning, P-Tuning 等方法证明，只训练输入侧的参数即可实现模态对齐
- LLM 的语言生成能力已经足够，只需学会"读懂"资源曲线

### 10.2 为何使用 Causal LM Loss?

**原因**:
1. **自然语言生成**: 目标是生成文本描述，Causal LM 是标准选择
2. **Teacher Forcing**: 训练时可以使用 ground truth 作为输入，加速收敛
3. **与 LLM 对齐**: Qwen2.5 本身就是 Causal LM，直接复用其训练目标

**替代方案**:
- Contrastive Learning: 可以尝试，但需要额外设计负样本
- Regression Loss: 不适合，因为目标是文本而非数值

### 10.3 为何使用 1D-CNN?

**原因**:
1. **时序特征**: 1D-CNN 天然适合提取时序信号的局部和全局特征
2. **多尺度**: 不同 kernel size 捕获不同时间尺度的模式
3. **计算效率**: 比 RNN/Transformer 更快，适合实时推理

**替代方案**:
- Transformer: 可以尝试，但参数量更大，可能过拟合
- LSTM: 适合长序列，但训练更慢

### 10.4 三种任务类型的必要性

**原因**:
1. **Type A (趋势)**: 训练宏观理解能力，对应 Phase 3 中的工具选择
2. **Type B (瓶颈)**: 训练细节捕获能力，对应资源冲突检测
3. **Type C (可行性)**: 训练逻辑推理能力，对应调度决策

**协同作用**:
- 三种任务共同覆盖了资源约束推理的所有方面
- 混合训练提高泛化能力

---

## 附录: 关键代码片段

### A.1 TemporalEncoder 添加 Projector

```python
# 修改 src/context/temporal_encoder.py 的 __init__ 方法
class TemporalEncoder(nn.Module):
    def __init__(self, ..., llm_embedding_dim: int = 3584):
        super().__init__()
        # ... 原有代码 ...
        
        # 添加 Projector
        self.projector = nn.Sequential(
            nn.Linear(cnn_config['output_dim'], llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
            nn.LayerNorm(llm_embedding_dim)
        )
    
    def forward_with_projection(self, t_inf_ms, t_end_ms=None):
        """Forward + Projection to LLM embedding space."""
        v_temporal = self.forward(t_inf_ms, t_end_ms)  # (output_dim,)
        v_projected = self.projector(v_temporal)       # (llm_embedding_dim,)
        return v_projected
```

### A.2 TemporalLLMWrapper Forward

```python
# src/context/temporal_llm_wrapper.py
class TemporalLLMWrapper(nn.Module):
    def forward(self, curve, prompt_ids, target_ids, attention_mask):
        batch_size = curve.size(0)
        
        # 1. Encode curves (batch processing)
        v_temporal_list = []
        for i in range(batch_size):
            # Extract single curve
            single_curve = curve[i]  # (num_timesteps, 4)
            
            # Reshape for CNN: (1, 4, num_timesteps)
            cnn_input = single_curve.transpose(0, 1).unsqueeze(0)
            
            # Forward through CNN
            v = self.temporal_encoder.cnn(cnn_input).squeeze(0)  # (output_dim,)
            
            # Project to LLM space
            v_proj = self.temporal_encoder.projector(v)  # (llm_dim,)
            v_temporal_list.append(v_proj)
        
        v_temporal = torch.stack(v_temporal_list)  # (batch, llm_dim)
        
        # 2. Get LLM embeddings
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)
        
        # 3. Prepend temporal embedding
        combined_embeds = torch.cat([
            v_temporal.unsqueeze(1),  # (batch, 1, llm_dim)
            prompt_embeds             # (batch, prompt_len, llm_dim)
        ], dim=1)
        
        # 4. Extend attention mask
        temporal_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        extended_mask = torch.cat([temporal_mask, attention_mask], dim=1)
        
        # 5. Prepare labels (mask prompt, only compute loss on target)
        # Target IDs 对应的位置是 [1+prompt_len : 1+prompt_len+target_len]
        prompt_len = prompt_ids.size(1)
        labels = torch.full(
            (batch_size, 1 + prompt_len + target_ids.size(1)),
            -100,
            dtype=torch.long,
            device=target_ids.device
        )
        labels[:, 1+prompt_len:] = target_ids
        
        # 6. Forward through LLM
        outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=extended_mask,
            labels=labels
        )
        
        return outputs.loss
```

---

## 总结

本文档提供了 Temporal Encoder 模态对齐预训练的完整实现指南，包括:

1. **数据生成**: 三种任务类型的合成数据生成器
2. **模型架构**: TemporalEncoder + Projector + 冻结 LLM
3. **训练流程**: 完整的训练脚本和配置
4. **评估体系**: 定量指标 + 定性可视化
5. **工程化**: 日志、监控、分布式训练支持

**实现优先级**:
1. 数据基础设施 (Phase 1)
2. 模型修改 + 快速测试 (Phase 2)
3. 完整训练流程 (Phase 3)
4. 评估与优化 (Phase 4-5)

**预期训练时间** (单卡 A100):
- 100k 样本 × 10 epochs ≈ 8-12 小时 (Qwen2.5-0.5B)
- 100k 样本 × 10 epochs ≈ 24-36 小时 (Qwen2.5-7B)

**成功标准**:
- 验证集 Perplexity < 3.0
- 定性测试中 80% 样本质量达标
- 能够准确描述趋势、定位瓶颈、判断可行性
