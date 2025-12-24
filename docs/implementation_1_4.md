# Section 1.4: Concatenation (拼接为资源感知工具嵌入)

## 概述

本节实现将工具语义嵌入（Stream A）与资源画像嵌入（Stream B）拼接为资源感知的工具嵌入。

```
v_tool (d_tool=768) || v_resource (d_resource=256) → v_toolaware (1024)
```

## 模块设计

### 1. ToolAwareEmbedding

核心拼接模块，提供以下功能：

#### 1.1 前向传播（Concatenation）

```python
def forward(tool_embeddings, resource_embeddings) -> toolaware_embeddings
```

- **输入**:
  - `tool_embeddings`: (batch_size, d_tool) 或 (d_tool,)
  - `resource_embeddings`: (batch_size, d_resource) 或 (d_resource,)
- **输出**: (batch_size, d_toolaware) 或 (d_toolaware,)
- **维度**: d_toolaware = d_tool + d_resource = 768 + 256 = 1024

#### 1.2 逆向拆分（Split）

```python
def split(toolaware_embeddings) -> (tool_embeddings, resource_embeddings)
```

用于分析或可视化，将拼接后的嵌入还原为两部分。

#### 1.3 维度验证

- 可选的维度检查（`validate_dims=True`）
- 确保输入维度与配置一致
- 检查batch size匹配

### 2. ResourceAwareToolEncoder

端到端编码器，封装完整流程：

```
Tool Names/Texts → ToolEncoder → tool_emb (768)
                                   ↓
Resource Vectors → ResourceMLP → resource_emb (256)
                                   ↓
                          Concatenation
                                   ↓
                        v_toolaware (1024)
```

#### 2.1 组件集成

```python
class ResourceAwareToolEncoder(nn.Module):
    def __init__(self, tool_encoder, resource_mlp, concatenator=None):
        self.tool_encoder = tool_encoder
        self.resource_mlp = resource_mlp
        self.concatenator = concatenator or auto_create()
```

#### 2.2 前向传播

```python
def forward(tool_names, tool_texts, resource_vectors, use_cache=True):
    tool_emb = self.tool_encoder(tool_names, tool_texts, use_cache)
    resource_emb = self.resource_mlp(resource_vectors)
    toolaware_emb = self.concatenator(tool_emb, resource_emb)
    return toolaware_emb
```

## 实现要点

### 1. 维度超参数化

所有维度从配置读取，训练时可调：

```yaml
model:
  tool_encoder:
    d_tool: 768  # 工具语义向量维度
  resource_mlp:
    d_resource: 256  # 资源画像投影维度
  # 自动计算: d_toolaware = 768 + 256 = 1024
```

### 2. 批量与单样本兼容

自动处理不同输入形状：

```python
# 单样本: (d_tool,) + (d_resource,) → (d_toolaware,)
# 批量: (batch, d_tool) + (batch, d_resource) → (batch, d_toolaware)
```

### 3. 梯度流保证

拼接操作保持梯度流动：

```python
# torch.cat 是可微分的
toolaware_emb = torch.cat([tool_emb, resource_emb], dim=-1)
# 梯度会正确反向传播到两个输入
```

### 4. 范数组合

拼接后的范数满足：

```
||v_toolaware||² = ||v_tool||² + ||v_resource||²
```

测试显示最大误差 < 1e-5。

## 代码结构

```
src/encoders/concatenation.py (285行)
├── ToolAwareEmbedding (132行)
│   ├── __init__(d_tool, d_resource, validate_dims)
│   ├── forward(tool_embeddings, resource_embeddings)
│   ├── split(toolaware_embeddings)
│   ├── from_config(config)
│   ├── get_output_dim()
│   └── extra_repr()
│
└── ResourceAwareToolEncoder (153行)
    ├── __init__(tool_encoder, resource_mlp, concatenator)
    ├── forward(tool_names, tool_texts, resource_vectors, use_cache)
    ├── get_output_dim()
    └── from_config(config, tool_names, encoder_type)
```

## 测试验证

### 1. 基础功能测试 (`test_concatenation.py`)

```bash
$ python tests/test_concatenation.py
```

**测试项目**:
1. ✓ 单样本拼接（维度: 768+256→1024）
2. ✓ split()逆向拆分（误差 < 1e-10）
3. ✓ 批量处理（batch_size=8）
4. ✓ from_config()工厂方法
5. ✓ 维度验证（错误维度抛出ValueError）
6. ✓ 梯度流（tool_emb和resource_emb都有梯度）
7. ✓ 范数组合（误差 < 1e-5）

### 2. 集成测试 (`test_integration_concatenation.py`)

```bash
$ python tests/test_integration_concatenation.py
```

**测试流程**:
1. ✓ 加载8工具×3尺寸=24样本
2. ✓ 完整流程：ToolEncoder → ResourceMLP → Concatenation
3. ✓ 端到端ResourceAwareToolEncoder
4. ✓ 梯度流过整个Pipeline（4个MLP参数有梯度）
5. ✓ 批量处理24样本：输出(24, 1024)
6. ✓ from_config()创建端到端编码器

**嵌入统计（24样本）**:
```
- Mean: 0.0017, Std: 0.0589
- Min: -0.8458, Max: 1.3161
- Norms: mean=2.22, std=0.93, range=[1.40, 6.08]
```

**按工具分组范数**:
```
image_classification: mean=2.03, std=0.62
text_summarization:   mean=1.90, std=0.29
video_transcoding:    mean=3.39, std=2.36  # 大尺寸导致高范数
sentiment_analysis:   mean=2.10, std=0.23
object_detection:     mean=2.17, std=0.86
machine_translation:  mean=1.92, std=0.38
speech_recognition:   mean=2.01, std=0.64
data_preprocessing:   mean=2.25, std=0.15
```

## 使用示例

### 1. 独立使用ToolAwareEmbedding

```python
import torch
import yaml
from encoders.concatenation import ToolAwareEmbedding

# 从配置创建
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

concat = ToolAwareEmbedding.from_config(config)
# ToolAwareEmbedding(d_tool=768, d_resource=256, d_toolaware=1024)

# 拼接
tool_emb = torch.randn(8, 768)      # batch=8
resource_emb = torch.randn(8, 256)
toolaware_emb = concat(tool_emb, resource_emb)
print(toolaware_emb.shape)  # torch.Size([8, 1024])

# 拆分
tool_part, resource_part = concat.split(toolaware_emb)
print((tool_emb - tool_part).abs().max())  # tensor(0.) - 完美重建
```

### 2. 使用端到端ResourceAwareToolEncoder

```python
from encoders.concatenation import ResourceAwareToolEncoder

# 从配置创建完整Pipeline
encoder = ResourceAwareToolEncoder.from_config(
    config,
    tool_names=['web_search', 'image_gen', 'code_exec'],
    encoder_type='name'
)

# 一步编码
resource_vectors = torch.randn(3, 6)  # [input_size, cpu, mem, gpu, gpu_mem, latency]
toolaware = encoder(
    tool_names=['web_search', 'image_gen', 'code_exec'],
    resource_vectors=resource_vectors,
    use_cache=True
)
print(toolaware.shape)  # torch.Size([3, 1024])
```

### 3. 手动组合三个组件

```python
from encoders.tool_encoder import ToolEncoder
from encoders.resource_mlp import ResourceMLP
from encoders.concatenation import ToolAwareEmbedding

# 创建各组件
tool_encoder = ToolEncoder(config, tool_names=['web_search'], encoder_type='name')
resource_mlp = ResourceMLP.from_config(config)
concat = ToolAwareEmbedding.from_config(config)

# 分步编码
tool_emb = tool_encoder(tool_names=['web_search'])
resource_emb = resource_mlp(torch.randn(1, 6))
toolaware = concat(tool_emb, resource_emb)
```

## 与整体架构的集成

### 上游输入

- **Stream A**: ToolEncoder输出 (batch, 768)
- **Stream B**: ResourceMLP输出 (batch, 256)

### 下游输出

- **v_toolaware**: (batch, 1024) → 送入多头自注意力（Section 1.5）

### 训练中的角色

```python
# Training Pipeline
for batch in dataloader:
    # 1. 编码工具语义
    tool_emb = tool_encoder(batch['tool_names'])
    
    # 2. 投影资源画像
    resource_emb = resource_mlp(batch['resource_vectors'])
    
    # 3. 拼接为资源感知嵌入
    toolaware_emb = concat(tool_emb, resource_emb)
    
    # 4. 送入自注意力（下一节）
    h_toolset = multi_head_attention(toolaware_emb)
    
    # 5. 后续处理...
```

## 参数统计

| 组件 | 参数量 |
|------|--------|
| ToolAwareEmbedding | 0（纯拼接，无参数） |
| ResourceAwareToolEncoder | 0（容器，参数在子模块） |
| **实际参数来源** | |
| ToolEncoder (name-based) | 8×768 = 6,144 |
| ResourceMLP | 134,912 |
| **总计** | **141,056** |

> 注：拼接操作本身不引入新参数，参数全部来自上游ToolEncoder和ResourceMLP。

## 性能考虑

### 1. 计算复杂度

拼接操作是O(1)内存复制：

```python
# PyTorch实现为视图拼接，极高效
toolaware = torch.cat([tool_emb, resource_emb], dim=-1)
# Time: ~1μs for batch=24, ~10μs for batch=1000
```

### 2. 内存占用

```
单样本: (768 + 256) × 4 bytes = 4096 bytes = 4KB
批量24: 24 × 4KB = 96KB
批量1000: 1000 × 4KB = 4MB
```

与上游ToolEncoder和ResourceMLP的输出共享内存（通过torch.cat的视图机制）。

### 3. 缓存策略

- ToolEncoder有缓存机制（工具名→嵌入）
- ResourceMLP无缓存（资源向量每次不同）
- 拼接结果不缓存（维度变化灵活）

## 下一步

完成Section 1.4后，Left Panel仅剩：

- **Section 1.5**: Multi-head Self-Attention（工具集合融合）
  - 输入：v_toolaware (batch, 1024)
  - 输出：h_toolset (batch, 1024) - 上下文化的工具集合表示
  - 组件：nn.MultiheadAttention + LayerNorm + Residual

Section 1.5完成后，整个**Left Panel (Input Processing & Encoders)**即告完成。

---

**实现时间**: 2024-XX-XX  
**测试状态**: ✅ All tests passed  
**代码行数**: 285行（模块）+ 182行（基础测试）+ 248行（集成测试）= 715行
