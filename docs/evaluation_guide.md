# Temporal Encoder 评估指南

本指南介绍如何评估训练好的 Temporal Encoder 模型。

## 目录

1. [概述](#概述)
2. [定量评估](#定量评估)
3. [定性可视化](#定性可视化)
4. [评估指标解读](#评估指标解读)
5. [最佳实践](#最佳实践)

---

## 概述

评估工具提供两种评估方式：

1. **定量评估** (`evaluate_temporal_encoder.py`): 计算客观指标
2. **定性可视化** (`visualize_temporal_predictions.py`): 生成可视化报告

---

## 定量评估

### 快速开始

```bash
python script/evaluate_temporal_encoder.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 5000
```

### 完整参数

```bash
python script/evaluate_temporal_encoder.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --config configs/pretrain_temporal.yaml \
    --num_samples 5000 \
    --batch_size 16 \
    --output_dir logs/temporal_pretrain/evaluation \
    --device cuda
```

### 评估指标

#### 1. Perplexity (困惑度)

**定义**: 模型预测下一个token的不确定性，越低越好。

**计算**:
- Overall Perplexity: 所有样本的平均困惑度
- Task A/B/C Perplexity: 分任务类型的困惑度

**评估标准**:
- ✅ **优秀**: < 3.0
- ✓ **良好**: 3.0 - 5.0
- ✗ **需改进**: > 5.0

**示例输出**:
```
Overall Perplexity: 2.34
Task A Perplexity: 2.12  (趋势描述)
Task B Perplexity: 2.45  (瓶颈定位)
Task C Perplexity: 2.56  (可行性判断)
```

#### 2. Token Accuracy (Token准确率)

**定义**: 生成的token与ground truth匹配的比例。

**计算**:
```python
accuracy = correct_tokens / total_tokens
```

**评估标准**:
- ✅ **优秀**: > 60%
- ✓ **良好**: 40% - 60%
- ✗ **需改进**: < 40%

**注意事项**:
- 只比较前N个token (N = min(len(generated), len(ground_truth)))
- 对同义词替换不友好 (如 "increasing" vs "rising")

#### 3. Numerical Accuracy (数值准确率)

**定义**: 生成文本中的数值与ground truth数值的匹配度。

**计算**:
- 从文本中提取所有数值
- 计算相对误差: `|generated - truth| / |truth|`
- 容忍度: ±5%

**评估标准**:
- ✅ **优秀**: > 70%
- ✓ **良好**: 50% - 70%
- ✗ **需改进**: < 50%

**示例**:
```
Ground Truth: "GPU memory drops to 4096MB at t=2.5s"
Prediction:   "GPU memory drops to 4100MB at t=2.4s"

数值提取: [4096, 2.5] vs [4100, 2.4]
相对误差: [0.1%, 4%]
结果: 2/2 correct (在5%容忍度内)
```

### 输出文件

评估完成后生成 `evaluation_results.json`:

```json
{
  "checkpoint_path": "checkpoints/temporal_pretrain/best_model.pt",
  "checkpoint_epoch": 10,
  "checkpoint_step": 10000,
  "overall_loss": 0.85,
  "overall_perplexity": 2.34,
  "task_A_perplexity": 2.12,
  "task_B_perplexity": 2.45,
  "task_C_perplexity": 2.56,
  "token_accuracy": 0.623,
  "numerical_accuracy": 0.745,
  "num_samples": 5000
}
```

---

## 定性可视化

### 快速开始

```bash
python tools/visualize_temporal_predictions.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 20
```

### 完整参数

```bash
python tools/visualize_temporal_predictions.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --config configs/pretrain_temporal.yaml \
    --num_samples 20 \
    --output logs/temporal_pretrain/predictions_visualization.html \
    --device cuda
```

### 可视化内容

#### 1. 概览统计

- 总样本数
- 各任务类型分布 (Type A/B/C)
- Checkpoint信息

#### 2. 每个样本展示

**资源曲线可视化**:
- 4条资源曲线 (CPU Cores, CPU Mem, GPU SM, GPU Mem)
- SVG格式，支持缩放
- 不同颜色区分资源类型

**文本对比**:
- Prompt (蓝色背景)
- Ground Truth (绿色背景)
- Model Prediction (橙色背景)

**差异高亮**:
- 不同的单词用红色背景标记
- 方便快速识别错误

### 示例输出

HTML报告示例:

```html
┌─────────────────────────────────────────────┐
│ Sample #1                      [Task A]     │
├─────────────────────────────────────────────┤
│ Resource Timeline:                          │
│ [SVG曲线图]                                 │
│                                             │
│ Prompt:                                     │
│ Analyze the GPU memory trend...            │
│                                             │
│ Ground Truth:                               │
│ GPU memory shows an increasing trend,       │
│ rising steadily from 2048MB to 6144MB      │
│                                             │
│ Model Prediction:                           │
│ GPU memory shows an increasing trend,       │
│ rising gradually from 2000MB to 6200MB     │
└─────────────────────────────────────────────┘
```

### 打开报告

```bash
# Linux
xdg-open logs/temporal_pretrain/predictions_visualization.html

# macOS
open logs/temporal_pretrain/predictions_visualization.html

# Windows
start logs/temporal_pretrain/predictions_visualization.html
```

---

## 评估指标解读

### 综合评估示例

假设评估结果如下:

```
Overall Perplexity: 2.34
Task A Perplexity: 2.12
Task B Perplexity: 2.45
Task C Perplexity: 2.56
Token Accuracy: 62.3%
Numerical Accuracy: 74.5%
```

**解读**:

1. **Perplexity 分析**:
   - 总体2.34 → **优秀** (< 3.0)
   - Task A最低 (2.12) → 趋势描述能力最强
   - Task C最高 (2.56) → 可行性判断略弱，但仍在良好范围

2. **Token Accuracy**:
   - 62.3% → **优秀** (> 60%)
   - 说明模型能准确生成大部分token

3. **Numerical Accuracy**:
   - 74.5% → **优秀** (> 70%)
   - 说明模型能准确预测数值

**总体评价**: 模型表现优秀，可用于生产环境。

### 问题诊断

#### 问题1: Perplexity高但Token Accuracy高

**现象**:
```
Perplexity: 6.5
Token Accuracy: 55%
```

**原因**: 模型对某些token不确定，但最终选择仍然正确

**解决**: 
- 增加训练epoch
- 增加训练数据

#### 问题2: Token Accuracy低但Numerical Accuracy高

**现象**:
```
Token Accuracy: 35%
Numerical Accuracy: 70%
```

**原因**: 模型用了不同的词汇描述，但数值正确

**示例**:
- Ground Truth: "increasing steadily"
- Prediction: "rising gradually"

**解决**: 这不一定是问题，定性检查是否语义正确

#### 问题3: 某个任务类型Perplexity明显高

**现象**:
```
Task A: 2.1
Task B: 2.3
Task C: 5.8  ← 明显偏高
```

**原因**: Task C (可行性判断) 训练不足

**解决**:
- 增加Task C的训练数据比例
- 调整 `type_distribution: {A: 0.3, B: 0.3, C: 0.4}`

---

## 最佳实践

### 1. 定期评估

**训练期间**:
```bash
# 每5个epoch评估一次
for epoch in 5 10 15 20; do
    python script/evaluate_temporal_encoder.py \
        --checkpoint checkpoints/temporal_pretrain/epoch_${epoch}_step_*.pt \
        --num_samples 1000 \
        --output_dir logs/eval_epoch_${epoch}
done
```

**比较不同checkpoint**:
```python
import json
import matplotlib.pyplot as plt

epochs = [5, 10, 15, 20]
perplexities = []

for epoch in epochs:
    with open(f'logs/eval_epoch_{epoch}/evaluation_results.json') as f:
        results = json.load(f)
        perplexities.append(results['overall_perplexity'])

plt.plot(epochs, perplexities)
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Model Performance over Training')
plt.savefig('perplexity_curve.png')
```

### 2. 分层评估

**按任务类型**:
```bash
# 创建只包含Task A的评估集
python script/evaluate_temporal_encoder.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 5000
```

查看JSON结果中的 `task_A_perplexity`, `task_B_perplexity`, `task_C_perplexity`

### 3. 可视化诊断

**生成大量样本**:
```bash
python tools/visualize_temporal_predictions.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 100 \
    --output logs/detailed_analysis.html
```

**重点检查**:
- 查找数值错误的样本
- 查找趋势判断错误的样本
- 查找瓶颈定位错误的样本

### 4. A/B测试

**对比两个checkpoint**:
```bash
# 评估Checkpoint A
python script/evaluate_temporal_encoder.py \
    --checkpoint checkpoints/run_A/best_model.pt \
    --output_dir logs/eval_A

# 评估Checkpoint B
python script/evaluate_temporal_encoder.py \
    --checkpoint checkpoints/run_B/best_model.pt \
    --output_dir logs/eval_B

# 对比结果
diff logs/eval_A/evaluation_results.json logs/eval_B/evaluation_results.json
```

### 5. 错误分析

**提取错误样本**:

```python
# tools/extract_errors.py
import json
import torch
from transformers import AutoTokenizer

# 加载评估结果
with open('logs/evaluation/evaluation_results.json') as f:
    results = json.load(f)

# 重新运行并保存错误样本
# (需要修改evaluate脚本保存详细结果)
```

---

## 评估脚本参数详解

### evaluate_temporal_encoder.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | Checkpoint路径 |
| `--config` | `configs/pretrain_temporal.yaml` | 配置文件 |
| `--num_samples` | 5000 | 评估样本数量 |
| `--batch_size` | 16 | 批大小 |
| `--output_dir` | `logs/temporal_pretrain/evaluation` | 输出目录 |
| `--device` | `cuda` | 设备 (cuda/cpu) |

### visualize_temporal_predictions.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | Checkpoint路径 |
| `--config` | `configs/pretrain_temporal.yaml` | 配置文件 |
| `--num_samples` | 20 | 可视化样本数量 |
| `--output` | `logs/.../predictions_visualization.html` | 输出HTML路径 |
| `--device` | `cuda` | 设备 (cuda/cpu) |

---

## 常见问题

### Q1: 评估时OOM

**解决**:
```bash
# 减小batch_size
python script/evaluate_temporal_encoder.py \
    --batch_size 4 \
    ...

# 或使用CPU
python script/evaluate_temporal_encoder.py \
    --device cpu \
    ...
```

### Q2: 评估速度慢

**解决**:
```bash
# 减少样本数量
python script/evaluate_temporal_encoder.py \
    --num_samples 1000 \
    ...

# 增大batch_size (如果显存足够)
python script/evaluate_temporal_encoder.py \
    --batch_size 32 \
    ...
```

### Q3: 可视化HTML打不开

**检查**:
- 文件路径是否正确
- 浏览器是否支持本地文件
- 尝试用不同浏览器打开

### Q4: 指标波动大

**原因**: 评估数据随机生成，种子不同导致波动

**解决**:
```bash
# 固定种子 (需修改脚本)
# 或增加评估样本数量
python script/evaluate_temporal_encoder.py \
    --num_samples 10000 \
    ...
```

---

## 总结

**评估工作流**:

1. **训练完成后**:
   ```bash
   python script/evaluate_temporal_encoder.py \
       --checkpoint checkpoints/temporal_pretrain/best_model.pt
   ```

2. **查看定量结果**:
   ```bash
   cat logs/temporal_pretrain/evaluation/evaluation_results.json
   ```

3. **生成可视化**:
   ```bash
   python tools/visualize_temporal_predictions.py \
       --checkpoint checkpoints/temporal_pretrain/best_model.pt
   ```

4. **分析结果**:
   - Perplexity < 3.0 → 优秀
   - Token Accuracy > 60% → 优秀
   - Numerical Accuracy > 70% → 优秀

5. **如有问题**:
   - 检查可视化报告找出典型错误
   - 调整训练参数或数据分布
   - 重新训练

**目标指标**:
- ✅ Overall Perplexity < 3.0
- ✅ Token Accuracy > 60%
- ✅ Numerical Accuracy > 70%
- ✅ 80%的可视化样本质量达标
