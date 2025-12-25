# TOMAS-LLM 架构升级与训练实施指南 (Vibe-Coding Guide)

本指南旨在指导 AI 编码助手完成 TOMAS-LLM 基于 "Weights as Memory" (权重即记忆) 架构的重构与训练流程实现。

## 核心架构变更摘要

*   **设计理念**: 静态资源画像不再作为 Input Context，而是转化为 Output Head 的权重矩阵。
*   **推理模式**: 输入 (Task + Dynamic Context) -> LLM -> 输出 (Tool ID + Config ID) -> 查表 (Resource Plan)。
*   **关键组件**:
    1.  **Phase 0 (离线)**: 生成 `[1701, 3584]` 的配置嵌入矩阵，用于初始化分类头。
    2.  **Phase 3 (解码)**: 两阶段分层解码 (Tool Classification -> Masked Config Pointer)。

---

## 任务清单 (Task List)

### 1. Phase 0: 编码器预训练 (Encoder Pre-training)

此阶段通过"自编码"任务预训练资源编码器，使其学会将 6D 资源向量压缩为 LLM 可理解的语义空间。

#### 1.1 数据准备

- [x] **实现自然语言模板生成器 (`src/offline/text_template.py`)** ✓ COMPLETED
    *   **类名**: `ConfigTextTemplate`
    *   **功能**: 将 (Tool, Resource) 数据转换为自然语言描述。
    *   **示例输出**: 
        ```
        "Tool super_resolution configuration: input size small, requires 4 CPU cores, 
         4 GB memory, 50 GPU SMs, 2 GB GPU memory, and latency is 5800 ms."
        ```
    *   **实现细节**:
        - `format_single(tool_name, resource_vector)`: 单条数据模板化。
        - `format_batch(data_list)`: 批量处理。
        - 支持资源字段的标准化命名 (input_size, cpu_cores, memory_gb, gpu_sm, gpu_memory_gb, latency_ms)。

- [x] **实现预训练数据集构建器 (`src/data/pretrain_dataset.py`)** ✓ COMPLETED
    *   **类名**: `EncoderPretrainDataset`
    *   **功能**: 生成 (Input, Target) 监督数据集用于编码器预训练。
    *   **输入**: 
        - Tool Registry (`data/tool_registry/tools.json`)
        - Profiling Data (`data/profiling/profiling.csv`)
    *   **输出**: PyTorch Dataset
        - `input`: Tool ID (int) + Resource Vector (Tensor `[6]`)
        - `target`: 自然语言描述的 Token IDs (Tensor `[seq_len]`)
    *   **数据增强**:
        - **Option A**: 数值抖动 (Jittering) - 对资源向量添加 ±5% 随机噪声。
        - **Option B**: 多轮重复 - 同一数据重复 100 Epoch（过拟合训练）。
    *   **规模**: 1701 条原始数据 → 增强后数万条训练样本。

#### 1.2 编码器架构

- [x] **实现预训练编码器 (`src/offline/pretrain_encoder.py`)** ✓ COMPLETED
    *   **类名**: `ResourceEncoderForPretraining`
    *   **架构**:
        1. **Stream A (工具语义)**: 
           - 使用 Qwen2.5 的 Embedding 层（冻结梯度）。
           - 输入 Tool Name，输出 `[1, 3584]`。
        2. **Stream B (资源编码)**:
           - `ResourceMLP` (可训练)：6D → 3584D。
        3. **Fusion (语义融合)**:
           - Self-Attention (可训练)：拼接 A + B，输出 `[1, 3584]`。
    *   **前向传播**:
        - `forward(tool_id, resource_vector)` → 返回融合嵌入 `[batch, 3584]`。
    *   **初始化策略**:
        - Stream A: 从预训练 Qwen2.5 加载 + 冻结。
        - Stream B & Fusion: Xavier/He 初始化。

#### 1.3 预训练流程

- [x] **编写编码器预训练脚本 (`script/pretrain_encoder.py`)** ✓ COMPLETED
    *   **训练目标**: 
        - 使用 Causal Language Modeling Loss。
        - 编码器输出嵌入 → 注入 Qwen2.5 → 生成目标文本。
    *   **损失函数**: 
        ```
        Loss = CrossEntropy(LLM_output, target_text_tokens)
        ```
    *   **优化器**: AdamW，仅优化 Stream B + Fusion 参数。
    *   **训练终止条件**: Loss < 0.01 或 100 Epochs（目标是过拟合）。
    *   **流程**:
        1. 加载 `EncoderPretrainDataset`。
        2. 初始化 `ResourceEncoderForPretraining`。
        3. 训练循环：
           - 编码器生成嵌入。
           - 嵌入作为前缀注入 Qwen2.5。
           - 计算 Next Token Prediction Loss。
        4. 保存训练好的编码器权重为 `assets/pretrained_encoder.pt`。

#### 1.4 权重导出

- [x] **编写权重导出脚本 (`script/export_classifier_weights.py`)** ✓ COMPLETED
    *   **流程**:
        1. 加载训练好的 `ResourceEncoderForPretraining`。
        2. 批量处理所有 1701 条原始 (Tool, Resource) 数据。
        3. 编码器推理模式，生成 `[1701, 3584]` 权重矩阵。
        4. 保存为 `assets/config_weights.pt`。
        5. 同时生成 `config_lookup.json` 和 `tool_mask_map.json`（复用现有 `ConfigLookupBuilder`）。
    *   **用途**: 该权重矩阵将用于主模型的 `HierarchicalDecoder.config_head` 初始化。

---

## 已废弃的实现 (Deprecated)

以下模块基于旧的理解实现，需要重构或替换：

- ~~`src/offline/embedding_generator.py`~~ - 将被 `pretrain_encoder.py` 替代。
- ~~`src/offline/lookup_builder.py`~~ - 保留部分逻辑，用于导出阶段生成查找表。

---

### 2. Phase 3: 分层解码器实现 (Hierarchical Decoding)

此模块替代原有的 `OutputParser` 和回归头。

- [ ] **实现分层解码器 (`src/decoders/hierarchical.py`)**
    *   **类名**: `HierarchicalDecoder`
    *   **组件**:
        *   `tool_head`: `nn.Linear(3584, 7)`。
        *   `config_head`: `nn.Linear(3584, 1701)` (权重将由 `config_weights.pt` 初始化)。
    *   **方法 `forward(hidden_state)`**:
        *   返回 `tool_logits` 和 `config_logits`。
    *   **方法 `predict(hidden_state, tool_mask_map)`**:
        *   计算 `tool_id = argmax(tool_logits)`。
        *   根据预测的 `tool_id`，从 `tool_mask_map` 获取合法的 `config_ids`。
        *   将 `config_logits` 中非法的 ID 设为 `-inf`。
        *   计算 `config_id = argmax(masked_config_logits)`。
        *   返回 `tool_id`, `config_id`。

### 3. 模型架构重构 (Model Refactoring)

更新主模型以适配新流程。

- [ ] **更新 TOMAS-LLM 模型 (`src/llm/model_wrapper.py`)**
    *   **移除**: 删除 `tool_encoder` 相关输入和处理逻辑（推理时不再需要）。
    *   **移除**: 删除旧的 `resource_regressor`, `tool_classifier`, `token_gate`。
    *   **新增**: 集成 `HierarchicalDecoder`。
    *   **修改 `forward`**:
        *   仅接收 `user_task` 和 `temporal_context`。
        *   LLM 输出 `hidden_states` 传给 `HierarchicalDecoder`。
    *   **LoRA 集成**: 确保 Qwen Backbone 支持 LoRA 配置 (使用 `peft` 库)。

- [ ] **更新 Context Projector (`src/llm/qwen_backbone.py`)**
    *   **移除**: 删除 `tool_proj` 分支（工具嵌入不再作为输入）。
    *   **保留**: 保留并优化 `temporal_proj`，负责将 1D-CNN 输出的低维时序特征 (256维) 映射到 LLM 输入维度 (3584维)。
    *   **定位**: 明确 Projector 仅作用于 Phase 1 (动态运行时输入)，负责将非文本结构化数据注入 LLM Prompt。

### 4. 训练流程实现 (Training Pipeline)

- [ ] **数据加载器更新 (`src/data/dataset.py`)**
    *   **输入**: User Task, System State (用于时序编码)。
    *   **标签**: `Tool ID` (0-6), `Config ID` (0-1700)。不再需要具体的资源数值作为 Label。

- [ ] **编写训练脚本 (`script/train.py`)**
    *   **初始化**:
        *   加载 Qwen2.5-7B，应用 LoRA。
        *   加载 `assets/config_weights.pt` 初始化 `HierarchicalDecoder.config_head`。
    *   **Loss Function**:
        *   `Loss = CrossEntropy(tool_pred, tool_gt) + lambda * CrossEntropy(config_pred, config_gt)`。
        *   注意：计算 Config Loss 时，只计算 Ground Truth Tool 对应的 Config 分支，或者全局计算皆可（因为有 Masking 机制，全局计算更简单）。
    *   **优化器**: 优化 LoRA 参数 + Decoder Head 参数 + Projector 参数。

### 5. 推理流程更新 (Inference Pipeline)

- [ ] **更新推理脚本 (`script/inference.py`)**
    *   加载 `assets/config_lookup.json` 和 `assets/tool_mask_map.json`。
    *   模型前向传播得到 `tool_id` 和 `config_id`。
    *   使用 `config_id` 查表得到最终 JSON 输出。

---

## 目录结构规划

```text
src/
  offline/              # [NEW] 离线处理模块
    __init__.py
    embedding_generator.py
    lookup_builder.py
  decoders/
    hierarchical.py     # [NEW] 分层解码器
    ...
  llm/
    model_wrapper.py    # [UPDATE] 移除旧Encoder，集成新Decoder
assets/                 # [NEW] 存放离线生成的权重和表
  config_weights.pt
  config_lookup.json
  tool_mask_map.json
script/
  generate_offline_assets.py # [NEW] 离线生成脚本
  train.py              # [NEW] 训练脚本
  inference.py          # [UPDATE] 推理脚本
```

## 注意事项

1.  **维度对齐**: 确保 Stream A, Stream B, Fusion, 和 LLM Hidden Size 均为 3584 (Qwen2.5-7B 维度)。
2.  **Masking 逻辑**: 推理时的 Masking 至关重要，必须确保模型不会预测出属于 Tool A 的 Config ID 却标记为 Tool B。
3.  **LoRA 配置**: 仅微调 Attention 模块 (q_proj, v_proj) 即可，保持主干冻结。
