# Tools Directory

本目录包含 Temporal Encoder 项目的辅助工具和脚本。

## 文件列表

| 工具 | 用途 | 状态 |
|------|------|------|
| `validate_temporal_dataset.py` | 数据集验证和可视化 | ✅ 已完成 |
| `visualize_temporal_predictions.py` | 模型预测可视化 | ✅ 已完成 |
| `generate_profiling_data.py` | 性能分析数据生成 | 已存在 |
| `simple.py` | 简单工具脚本 | 已存在 |
| `monitor_training.py` | 训练监控工具 | ⏳ 待实现 |

---

## 快速使用

### 1. 数据集验证

在训练前验证合成数据的质量:

```bash
python tools/validate_temporal_dataset.py \
    --num_samples 1000 \
    --output_dir data/temporal_pretrain
```

**输出**: `data/temporal_pretrain/validation_report.html`

**功能**:
- ✅ 可视化资源曲线
- ✅ 显示文本描述
- ✅ 统计任务类型分布
- ✅ 验证物理约束
- ✅ 检查文本长度分布

---

### 2. 预测可视化

评估训练好的模型:

```bash
python tools/visualize_temporal_predictions.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 20
```

**输出**: `logs/temporal_pretrain/predictions_visualization.html`

**功能**:
- ✅ 资源曲线SVG可视化
- ✅ Prompt、Ground Truth、Prediction对比
- ✅ 差异高亮显示
- ✅ 任务类型统计
- ✅ 交互式HTML报告

---

## 详细文档

### validate_temporal_dataset.py

**目的**: 在训练前验证数据生成器的正确性。

**参数**:
```bash
python tools/validate_temporal_dataset.py \
    --num_samples 1000 \              # 验证样本数量
    --output_dir data/temporal_pretrain  # 输出目录
```

**验证项**:
1. **资源曲线**:
   - 数值范围是否合理
   - 趋势是否正确
   - Spike注入是否正常

2. **文本描述**:
   - 模板是否正确
   - 数值是否匹配
   - 语法是否通顺

3. **任务分布**:
   - Type A/B/C比例
   - 文本长度分布
   - Token数量统计

**HTML报告内容**:
- 20个随机样本的可视化
- 每个样本包含:
  - 资源曲线图
  - Prompt文本
  - Target文本
  - 任务类型标签
- 统计信息汇总

**使用建议**:
- 在首次运行训练前必须验证
- 修改数据生成器后重新验证
- 定期抽查数据质量

---

### visualize_temporal_predictions.py

**目的**: 可视化模型预测，进行定性分析。

**参数**:
```bash
python tools/visualize_temporal_predictions.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \  # Checkpoint路径
    --config configs/pretrain_temporal.yaml \                    # 配置文件
    --num_samples 20 \                                           # 样本数量
    --output logs/predictions.html \                             # 输出路径
    --device cuda                                                # 设备
```

**可视化内容**:

1. **概览统计**:
   - 总样本数
   - 各任务类型数量
   - Checkpoint信息

2. **每个样本**:
   - 资源曲线 (4条线: CPU cores, CPU mem, GPU SM, GPU mem)
   - Prompt (蓝色背景)
   - Ground Truth (绿色背景)
   - Model Prediction (橙色背景)
   - 差异高亮 (红色背景)

**分析方法**:

1. **查找典型错误**:
   - 数值偏差
   - 趋势判断错误
   - 瓶颈定位不准

2. **评估整体质量**:
   - 语法流畅度
   - 数值准确性
   - 逻辑一致性

3. **对比不同checkpoint**:
   ```bash
   # Epoch 5
   python tools/visualize_temporal_predictions.py \
       --checkpoint checkpoints/temporal_pretrain/epoch_5_step_5000.pt \
       --output logs/epoch_5_viz.html
   
   # Epoch 10
   python tools/visualize_temporal_predictions.py \
       --checkpoint checkpoints/temporal_pretrain/epoch_10_step_10000.pt \
       --output logs/epoch_10_viz.html
   ```

**使用建议**:
- 训练后立即可视化
- 每5个epoch生成一次
- 重点查看预测错误的样本
- 用于演示和报告

---

## 工作流示例

### 完整评估流程

```bash
# 1. 训练模型
python script/pretrain_temporal_encoder.py \
    --config configs/pretrain_temporal.yaml

# 2. 定量评估
python script/evaluate_temporal_encoder.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 5000

# 3. 定性可视化
python tools/visualize_temporal_predictions.py \
    --checkpoint checkpoints/temporal_pretrain/best_model.pt \
    --num_samples 20

# 4. 查看结果
cat logs/temporal_pretrain/evaluation/evaluation_results.json
xdg-open logs/temporal_pretrain/predictions_visualization.html
```

### 数据质量检查流程

```bash
# 1. 验证数据集
python tools/validate_temporal_dataset.py \
    --num_samples 1000

# 2. 查看报告
xdg-open data/temporal_pretrain/validation_report.html

# 3. 如有问题，调整数据生成器
# 编辑 src/data/resource_curve_synthesizer.py
# 或 src/data/text_description_generator.py

# 4. 重新验证
python tools/validate_temporal_dataset.py \
    --num_samples 1000
```

---

## 输出文件

### 验证报告

**位置**: `data/temporal_pretrain/validation_report.html`

**内容**:
- 样本可视化 (曲线 + 文本)
- 统计信息 (任务分布、长度分布)
- 质量检查结果

### 预测可视化

**位置**: `logs/temporal_pretrain/predictions_visualization.html`

**内容**:
- Checkpoint信息
- 样本可视化 (曲线 + Prompt + GT + Prediction)
- 差异高亮
- 任务统计

---

## 高级用法

### 批量可视化

```bash
# 可视化多个checkpoint
for epoch in 5 10 15 20; do
    python tools/visualize_temporal_predictions.py \
        --checkpoint checkpoints/temporal_pretrain/epoch_${epoch}_step_*.pt \
        --num_samples 20 \
        --output logs/viz_epoch_${epoch}.html
done
```

### 自定义样本

如果想可视化特定类型的样本，可以修改脚本:

```python
# 在 visualize_temporal_predictions.py 中
dataset = TemporalPretrainDataset(
    num_samples=args.num_samples,
    type_distribution={'A': 1.0, 'B': 0.0, 'C': 0.0},  # 只生成Type A
    ...
)
```

### 导出数据

从HTML中提取数据用于分析:

```python
# tools/extract_from_html.py
from bs4 import BeautifulSoup

with open('logs/predictions_visualization.html') as f:
    soup = BeautifulSoup(f, 'html.parser')

# 提取文本内容
samples = soup.find_all('div', class_='sample')
for sample in samples:
    prompt = sample.find('div', class_='prompt').text
    prediction = sample.find('div', class_='prediction').text
    # ... 分析 ...
```

---

## 常见问题

### Q1: HTML报告太大打不开

**原因**: 样本数量过多

**解决**:
```bash
# 减少样本数量
python tools/visualize_temporal_predictions.py \
    --num_samples 10 \
    ...
```

### Q2: 曲线显示不正常

**检查**:
- 数据范围是否合理
- SVG生成是否正确
- 浏览器是否支持SVG

### Q3: 差异高亮不准确

**原因**: 简单的word-level diff

**改进**: 可以使用更复杂的diff算法 (如difflib)

---

## 开发计划

### monitor_training.py (待实现)

**功能**:
- 实时监控GPU使用率
- 记录内存占用
- 估计剩余训练时间
- 发送异常告警

**使用**:
```bash
python tools/monitor_training.py \
    --log_dir logs/temporal_pretrain \
    --alert_email user@example.com
```

---

## 参考资料

- **评估指南**: `docs/evaluation_guide.md`
- **训练指南**: `docs/temporal_pretrain_guide.md`
- **TODO清单**: `TODO-context.md`
