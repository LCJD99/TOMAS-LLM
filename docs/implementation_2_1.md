# Section 2.1: User Task & Embedding (用户任务编码)

## 概述

本节实现用户任务描述的编码，将自然语言任务转换为LLM可处理的嵌入表示。使用与LLM backbone相同的tokenizer和embedding layer，确保完美兼容。

```
User Task Text → Tokenization → Task Embeddings (seq_len, d_model)
```

## 设计理念

### 为何使用LLM自身的Embedding？

1. **完美兼容**：与LLM backbone共享vocabulary和embedding space
2. **无需额外训练**：直接利用预训练知识
3. **统一表示**：任务文本和LLM输出在同一语义空间
4. **简化架构**：避免引入额外的text encoder

### 与工具编码的区别

| 维度 | 工具编码 (Section 1.2) | 任务编码 (Section 2.1) |
|------|------------------------|----------------------|
| **目的** | 编码工具的语义特征 | 编码用户任务意图 |
| **输入** | 工具名称/描述（固定） | 用户输入文本（动态） |
| **编码器** | 独立的ToolEncoder | LLM自身的embedding |
| **维度** | 768D（可配置） | 896D（Qwen2.5-0.5B） |
| **缓存** | 可缓存（工具固定） | 不缓存（每次不同） |

## 模块设计

### 1. TaskEmbedding（基础嵌入层）

#### 1.1 核心功能

```python
class TaskEmbedding(nn.Module):
    """使用预训练LLM的tokenizer和embeddings编码任务文本"""
    
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",  # 与LLM backbone保持一致
        max_length=512,
        device="cpu",
        use_pretrained_embeddings=True  # 是否使用预训练权重
    )
```

**关键特性**:
- 加载HuggingFace tokenizer
- 提取LLM的embedding layer（冻结权重）
- 自动处理padding/truncation
- 返回token embeddings + attention mask

#### 1.2 Tokenization

```python
def tokenize(texts, padding=True, truncation=True):
    """
    Args:
        texts: str or List[str]
        padding: 是否padding到max_length
        truncation: 是否截断超长文本
    
    Returns:
        {
            'input_ids': (batch, seq_len),
            'attention_mask': (batch, seq_len)
        }
    """
```

**示例**:
```python
texts = [
    "Generate an image of a sunset",
    "Analyze sentiment of reviews"
]
encoding = task_emb.tokenize(texts)
# input_ids: (2, 8) for Qwen2.5-0.5B
# attention_mask: (2, 8)
```

#### 1.3 Forward Pass

```python
def forward(texts=None, input_ids=None, attention_mask=None):
    """
    Returns:
        embeddings: (batch, seq_len, d_model)
        attention_mask: (batch, seq_len)
    """
```

**维度**: d_model根据模型自动确定
- Qwen2.5-0.5B: **896**
- Qwen2.5-7B: **3,584**
- Qwen2.5-14B: **5,120**

### 2. UserTaskEncoder（完整编码器）

#### 2.1 架构

```
Text → TaskEmbedding → Sequence Embeddings (seq_len, d_model)
                              ↓
                    Optional Projection → (seq_len, d_tool)
                              ↓
                         Pooling → Pooled Embedding (d_model or d_tool)
```

#### 2.2 核心功能

```python
class UserTaskEncoder(nn.Module):
    def __init__(
        self,
        task_embedding: TaskEmbedding,
        project_to_tool_dim=False,  # 是否投影到工具维度
        d_tool=None,  # 工具嵌入维度（1024 = 768 + 256）
        pooling_method="mean"  # 池化方法
    )
```

**可选投影层**:
```python
if project_to_tool_dim:
    self.projection = nn.Linear(d_model, d_tool)
    # 896 → 1024 for Qwen2.5-0.5B
```

#### 2.3 Pooling方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| **mean** | 所有token的均值（masked） | 默认选择，适合大多数任务 |
| **max** | 最大值池化 | 强调关键信息 |
| **cls** | 使用首个token（CLS风格） | 如果模型训练了CLS token |
| **last** | 使用最后一个非padding token | 强调句尾信息 |

**实现**（mean pooling）:
```python
def pool_sequence(embeddings, attention_mask):
    # Masked mean pooling
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    pooled = sum_embeddings / sum_mask
    return pooled
```

#### 2.4 Forward Pass

```python
def forward(texts=None, input_ids=None, return_pooled=True):
    """
    Returns:
        if return_pooled=True:
            (sequence_embeddings, pooled_embeddings, attention_mask)
        else:
            (sequence_embeddings, attention_mask)
    """
```

## 实现要点

### 1. Embedding层冻结

预训练embeddings被冻结，不参与训练：

```python
# Freeze pretrained embeddings
for param in self.embeddings.parameters():
    param.requires_grad = False
```

**原因**:
- 保持LLM预训练的语义空间
- 减少训练参数（仅投影层可训练）
- 避免过拟合

### 2. Attention Mask处理

正确处理变长序列的masking：

```python
# 确保padding位置不参与计算
embeddings[attention_mask == 0] = 0  # 或设为-inf（max pooling）
```

### 3. 与工具维度对齐（可选）

如果需要与工具嵌入拼接，使用投影层：

```python
# d_toolaware = d_tool_semantic + d_resource = 768 + 256 = 1024
encoder = UserTaskEncoder.from_config(config, project_to_tool_dim=True)
# 输出: (batch, 1024) 可与 h_toolset 拼接
```

### 4. 配置驱动

所有参数在configs中定义：

```yaml
model:
  backbone:
    name: "Qwen/Qwen2.5-0.5B"  # 模型名称
    device: "cpu"
  
  task_encoder:
    max_length: 512  # 最大token长度
    use_pretrained_embeddings: true  # 使用预训练
    pooling_method: "mean"  # 池化方法
```

## 代码结构

```
src/context/user_task.py (379行)
├── TaskEmbedding (162行)
│   ├── __init__(model_name, max_length, device, use_pretrained_embeddings)
│   ├── tokenize(texts, padding, truncation, return_tensors)
│   ├── forward(texts, input_ids, attention_mask)
│   ├── get_vocab_size()
│   ├── get_embedding_dim()
│   └── from_config(config)
│
└── UserTaskEncoder (217行)
    ├── __init__(task_embedding, project_to_tool_dim, d_tool, pooling_method)
    ├── pool_sequence(embeddings, attention_mask)
    ├── forward(texts, input_ids, return_pooled)
    ├── get_output_dim()
    └── from_config(config, project_to_tool_dim)
```

## 测试验证

### 测试配置（simple-test.yaml）

为方便测试，使用小模型：

```yaml
model:
  backbone:
    name: "Qwen/Qwen2.5-0.5B"  # 小模型（~1GB）
    device: "cpu"
    dtype: "float32"
```

**优势**:
- 快速下载（vs 7B模型的~14GB）
- CPU可运行
- 测试充分覆盖功能

### 测试结果 (test_user_task.py)

```bash
$ python tests/test_user_task.py
```

**测试项目**:
1. ✓ TaskEmbedding初始化
   - 模型: Qwen/Qwen2.5-0.5B
   - Vocab size: **151,665**
   - Embedding dim: **896**

2. ✓ 文本分词
   - Input: 3个测试文本
   - Output: (3, 8) token IDs
   - 正确解码验证

3. ✓ Embedding生成
   - Shape: (3, 8, 896) ✓
   - 统计: mean=0.000034, std=0.015107

4. ✓ UserTaskEncoder（无投影）
   - Sequence: (3, 8, 896)
   - Pooled: (3, 896)

5. ✓ 池化方法对比
   - mean: norm=0.1702
   - max: norm=0.6643
   - cls: norm=0.4470
   - last: norm=0.4203

6. ✓ 投影到工具维度
   - Input: 896D
   - Output: **1024D** (d_toolaware)
   - Trainable projection layer

7. ✓ 梯度流
   - Projection layer: ✓ has gradients
   - Embedding layer: ✓ frozen

8. ✓ 单样本vs批量一致性
   - Max diff: **0.0** (完全一致)

9. ✓ 变长序列处理
   - 2 tokens, 10 tokens, 27 tokens
   - Attention mask正确: [2, 10, 27]
   - 不同长度产生不同embedding ✓

## 使用示例

### 1. 基础使用

```python
import yaml
from context.user_task import TaskEmbedding

# 加载配置
with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

# 创建TaskEmbedding
task_emb = TaskEmbedding.from_config(config)

# 编码任务
user_task = "Generate an image of a sunset over mountains"
embeddings, attention_mask = task_emb(user_task)

print(embeddings.shape)  # (1, seq_len, 896)
```

### 2. 批量处理

```python
tasks = [
    "Analyze sentiment of customer reviews",
    "Transcribe audio to text",
    "Generate product description"
]

embeddings, mask = task_emb(tasks)
print(embeddings.shape)  # (3, max_seq_len, 896)
```

### 3. UserTaskEncoder with Pooling

```python
from context.user_task import UserTaskEncoder

# 创建编码器
encoder = UserTaskEncoder.from_config(config, project_to_tool_dim=False)

# 获取序列和池化表示
seq_emb, pooled_emb, mask = encoder(tasks, return_pooled=True)

print(seq_emb.shape)    # (3, seq_len, 896) - 序列表示
print(pooled_emb.shape) # (3, 896) - 池化表示（送入后续模块）
```

### 4. 投影到工具维度

```python
# 创建带投影的编码器
encoder_proj = UserTaskEncoder.from_config(config, project_to_tool_dim=True)

# 输出与工具嵌入维度一致（1024D）
_, pooled_1024d, _ = encoder_proj(tasks, return_pooled=True)

print(pooled_1024d.shape)  # (3, 1024)
# 可与 h_toolset (num_tools, 1024) 在下游拼接或交互
```

### 5. 预先分词（高级）

```python
# 预先分词（如需要自定义处理）
encoding = task_emb.tokenize(tasks)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# 使用预分词结果
embeddings, mask = task_emb(input_ids=input_ids, attention_mask=attention_mask)
```

## 参数统计

| 组件 | 参数量 | 可训练 |
|------|--------|--------|
| TaskEmbedding（Qwen2.5-0.5B） | ~430M | ❌ (冻结) |
| Projection (896→1024) | 917,504 | ✅ |
| **UserTaskEncoder总计** | ~431M | **0.9M可训练** |

**注**：
- TaskEmbedding使用预训练权重，冻结不训练
- 仅Projection layer参数可训练（如果启用）
- 实际训练参数 < 1M，非常轻量

## 与整体架构的集成

### 在Pipeline中的位置

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CENTER PANEL (Section 2.x)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User Task Text
    ↓
UserTaskEncoder → task_emb (seq_len, 896) ✅ THIS
    ↓
[拼接工具上下文]
    ↓
h_toolset (num_tools, 1024) ← From Left Panel
    ↓
[构建LLM输入]
    ↓
Qwen2.5 Backbone
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 下游使用场景

#### 场景1：任务与工具上下文拼接

```python
# 任务编码
task_emb_seq, task_emb_pooled, _ = user_task_encoder(user_task)
# task_emb_seq: (1, seq_len, 896)

# 工具集合上下文（来自Left Panel）
h_toolset = complete_tool_encoder(tool_names, resource_vectors)
# h_toolset: (8, 1024)

# 构建LLM输入（方案1：直接拼接）
# 需要调整维度或使用adapter
```

#### 场景2：作为LLM的prompt embedding

```python
# 获取任务的sequence embeddings
task_seq_emb, _, mask = user_task_encoder(user_task, return_pooled=False)

# 直接作为LLM的输入embeddings
llm_outputs = llm_backbone(inputs_embeds=task_seq_emb, attention_mask=mask)
```

#### 场景3：任务-工具attention

```python
# 任务作为Query，工具作为Key/Value
# task_emb_pooled: (1, 1024) after projection
# h_toolset: (8, 1024)

attention_scores = torch.matmul(task_emb_pooled, h_toolset.T)
# (1, 8) - 每个工具对任务的相关性
```

## 下一步

完成Section 2.1后，Center Panel继续实现：

- **Section 2.2**: Temporal Encoder (时序特征编码)
- **Section 2.3**: Runtime Context Integration (上下文融合)
- **Section 3.x**: LLM Backbone Integration (Qwen2.5集成)

---

**实现时间**: 2024-12-24  
**测试状态**: ✅ All tests passed (Qwen2.5-0.5B)  
**代码行数**: 379行（模块）+ 263行（测试）= 642行  
**配置文件**: simple-test.yaml (专用测试配置)
