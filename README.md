# TOMAS-LLM

**T**ool **O**rchestration with **M**ulti-dimensional **A**wareness **S**ystem using **LLM**

资源感知的工具规划/资源分配 LLM 系统原型

## 项目概述

TOMAS-LLM 是一个基于 Qwen2.5-7B 的资源感知工具规划系统，能够根据：
- **静态工具知识**：工具语义描述 + profiling 资源画像
- **动态运行时上下文**：用户任务 + 预测的系统资源曲线

自动生成包含工具选择和资源分配的可执行计划。

## 架构设计

系统采用三栏架构：

### Left Panel: Input Processing & Encoders
- **Static Tool Knowledge Base**：工具注册表 + profiling 数据
  - Tool Encoder (语义编码)
  - Resource MLP (资源画像投影)
  - Multi-head Self-Attention (特征融合)
- **Dynamic Runtime Context**：动态上下文
  - User Task Embedding
  - Latency Prediction Module
  - System Timeline Snapshot Prediction
  - 1D-CNN Temporal Encoder

### Center Panel: LLM Backbone & Reasoning
- Qwen2.5-7B Transformer Decoder Layers

### Right Panel: Output Generation & Parsing
- Token Type Gate / Switch
- Path A: Standard LM Head (解释性文本)
- Path B: Resource Allocation Head
  - Tool Classifier (Softmax)
  - Resource Regressor (MLP)

## 目录结构

```
TOMAS-LLM/
├── configs/                 # 配置文件
│   └── default.yaml
├── data/                    # 数据目录
│   ├── tool_registry/       # 工具语义库
│   │   └── tools.json
│   └── profiling/           # Profiling 资源画像
│       └── profiling.csv
├── src/                     # 源代码
│   ├── schemas/             # 数据校验 (Pydantic)
│   ├── encoders/            # 编码器模块
│   ├── runtime/             # 运行时模块
│   ├── model/               # 模型模块
│   ├── planner/             # 规划器模块
│   ├── eval/                # 评估模块
│   └── main.py              # 主入口
├── script/                  # 脚本
│   ├── train.py             # 训练脚本
│   └── inference.py         # 推理脚本
├── TODO.md                  # 开发任务清单
└── README.md                # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖（待创建 requirements.txt）
pip install -r requirements.txt
```

### 2. 查看示例数据

```bash
# 工具注册表
cat data/tool_registry/tools.json

# Profiling 数据
cat data/profiling/profiling.csv
```

### 3. 运行最小原型

```bash
# 运行主入口（目前为占位符）
python src/main.py --task "对一批图片进行分类"

# 指定输出路径
python src/main.py --task "翻译一篇英文文档" --output output/plan.json
```

## 开发状态

当前项目处于骨架搭建阶段。详细开发任务和进度请查看 [TODO.md](TODO.md)。

已完成：
- ✅ 项目目录结构
- ✅ 配置文件框架
- ✅ 示例数据（tools.json, profiling.csv）
- ✅ 主入口占位符

待实现：
- ⬜ 数据加载器与校验
- ⬜ 各编码器模块
- ⬜ LLM Backbone 集成
- ⬜ Output Heads
- ⬜ 训练和推理流程

## 配置说明

主要配置文件：[configs/default.yaml](configs/default.yaml)

关键配置项：
- `model.backbone.name`: LLM 模型名称（默认 Qwen/Qwen2.5-7B）
- `model.tool_encoder.d_tool`: 工具语义向量维度
- `model.resource_mlp.d_resource`: 资源画像投影维度
- `training.strategy`: 训练策略（freeze_backbone / lora / full_finetune）
- `runtime.naive_mode.enabled`: 是否启用 naive 固定曲线模拟

## 贡献指南

请参考 [TODO.md](TODO.md) 中的任务列表选择合适的开发任务。

## License

待定

## 联系方式

待定
