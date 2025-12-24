# TOMAS-LLM TODO

> 目标：实现“资源感知的工具规划/资源分配”LLM系统原型，对齐三栏架构：
> - Left Panel：Input Processing & Encoders（输入处理与编码端）
> - Center Panel：LLM Backbone & Reasoning（LLM骨干与推理）
> - Right Panel：Output Generation & Parsing（输出生成与解析）
>
> 约定：
> - **离线/训练期**：静态工具知识库（工具语义 + profiling资源画像）进入模型权重/可训练模块。
> - **在线/推理期**：动态运行时上下文（用户任务 + 预测的未来资源曲线）作为额外输入/条件。

---

## 0. 项目骨架与约定

- [x] 定义目录结构（建议）：
  - [x] `data/tool_registry/`：`tools.json`
  - [x] `data/profiling/`：`profiling.csv`
  - [x] `src/schemas/`：数据校验（pydantic / jsonschema）
  - [x] `src/encoders/`：tool encoder、resource MLP、temporal encoder
  - [x] `src/runtime/`：resource snapshot、queue、timeline prediction
  - [x] `src/model/`：Qwen2.5-7B封装、额外heads、token gate
  - [x] `src/planner/`：tool plan解码与可执行计划生成
  - [x] `src/eval/`：离线评估、对齐指标
  - [x] `configs/`：维度、超参、路径
  - [x] `script/train.py`： 训练脚本
  - [x] `script/inference.py`：推理脚本 
- [x] 统一配置方式：`configs/default.yaml`
- [x] 定义最小可运行Demo入口：`src/main.py`（给定一条用户指令，输出 Tool Plan）

---

## 1. Left Panel：Static Tool Knowledge Base（静态工具知识库｜训练期）

### 1.1 Tool Registry & Profiling Data（离线数据库）

- [x] 设计工具语义库数据格式（JSON）
  - [x] 字段：`name`、`desc`
  - [x] 校验：name唯一；desc非空；长度限制
- [x] 设计profiling数据格式（CSV）其中input_size是分为三档的桶，分别代表三种划分等级
  - [x] 列：`tool`、`input_size`、`cpu_core`、`cpu_mem_gb`、`gpu_sm`、`gpu_mem_gb`、`latency_ms`
  - [x] 约束：单位统一；缺失值策略；异常值处理
- [x] 实现数据加载器
  - [x] `load_tool_registry()`：读json → 列表/字典
  - [x] `load_profiling_matrix()`：读csv → 张量/表格
  - [x] join逻辑：profiling的`tool`必须在registry中存在
- [x] 生成训练用样本（最小原型）
  - [x] 每条工具形成：语义文本 + 数值画像向量

### 1.2 Tool Encoder（Stream A：语义编码）

- [x] 选型并实现文本编码器，d_tool 和 d_resourse 构成 llm embedding 的维度，但是其维度分配作为一个训练参数放在配置中
  - [x] 实现1：使用工具映射编码，直接将工具的name映射到一个编码表中，编码表是词表外的一段连续的空间
  - [x] 实现2：用Qwen2.5 tokenizer + embedding层（或单独的sentence encoder）
  - [x] 输出维度参数化：`d_tool`
- [x] 输入模板确定（确保稳定）：
  - [x] 例如：`"Tool: {name}\nDescription: {desc}"`
- [x] 缓存：对静态工具文本做embedding cache（训练/推理复用）

### 1.3 MLP Projection（Stream B：资源画像投影）

- [x] 定义数值特征向量：
  - [x] 基础：`[input_size, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb, latency_ms]`
  - [x] 标准化：z-score 或 min-max（保存统计量供推理期使用）
- [x] 实现 MLP：`Linear -> ReLU -> Linear`
  - [x] 输入维度=6（或扩展后）
  - [x] 输出维度参数化：`d_resource`

### 1.4 Concatenation（拼接为资源感知工具嵌入）

- [x] 拼接：`v_tool || v_resource` → `v_toolaware`
  - [x] 维度：`d_tool + d_resource`
- [x] d_tool 和 d_resource 作为超参数训练时可调，需要确保该维度和 后续llm 的 embedding一致
- [x] 创建ToolAwareEmbedding模块（拼接+split）
- [x] 创建ResourceAwareToolEncoder端到端封装
- [x] 单元测试：维度验证、split()、梯度流（test_concatenation.py）
- [x] 集成测试：完整Pipeline（test_integration_concatenation.py）
- [x] 文档：docs/implementation_1_4.md

### 1.5 Multi-head Self-Attention（特征提取/融合）

- [x] 实现对"工具集合"的self-attention encoder（最小：1层MHSA）
  - [x] 输入：所有工具的 `v_toolaware`
  - [x] 输出：工具上下文化表示 `h_toolset`
- [x] 设计输出接口：给下游 tool classifier / planner 作为检索或条件输入
- [x] 创建ToolSetAttention（单层注意力）
- [x] 创建ToolSetEncoder（多层可选FFN）
- [x] 创建CompleteToolEncoder（完整Left Panel封装）
- [x] 单元测试：注意力权重、梯度流、FFN（test_tool_attention.py）
- [x] 集成测试：完整Pipeline（test_integration_left_panel.py）
- [x] 文档：docs/implementation_1_5.md

**✅ LEFT PANEL (Section 1.x) 全部完成！**
- 总参数量: 4,341,504
- 输入: 工具名/描述 + 6D资源向量
- 输出: 上下文化的1024D工具集合嵌入

**验收（静态库）**
- [ ] 同一工具在不同profiling点能生成不同 `v_resource`
- [ ] `v_toolaware`维度可配置且训练可收敛（至少loss下降）

---

## 2. Center Panel：Dynamic Runtime Context（动态运行时上下文｜推理期）

### 2.1 User Task & Embedding（用户任务编码）

- [x] 定义用户指令输入格式（纯文本 / 可含结构化字段）
- [x] 将用户任务编码为主序列输入（Qwen2.5标准embedding路径）
- [x] 创建TaskEmbedding（使用LLM预训练embeddings）
- [x] 创建UserTaskEncoder（支持pooling和投影）
- [x] 单元测试：分词、embedding、pooling（test_user_task.py）
- [x] 配置文件：simple-test.yaml（使用Qwen2.5-0.5B快速测试）
- [x] 文档：docs/implementation_2_1.md

**关键特性**:
- 使用LLM自身的tokenizer和embeddings（完美兼容）
- 支持4种pooling方法：mean, max, cls, last
- 可选投影到工具维度（896D→1024D）
- 冻结embedding层，仅投影层可训练（<1M参数）

### 2.2 Latency Prediction Module (T_inf) ✅
**Status**: COMPLETE - Naive implementation with flexible interface
- [x] LatencyPredictor with 3 modes (fixed, rule_based, learned)
- [x] Fixed mode: Returns constant latency (500ms default)
- [x] Rule-based mode: Tool lookup table + resource scaling
  - [x] Input size scaling (0.5x/1.0x/2.0x)
  - [x] GPU acceleration modeling (30% reduction)
- [x] Learned mode: MLP predictor (6D→1D, ~17K params, optional)
- [x] LatencyAwareModule wrapper for planning integration
- [x] 10/10 tests passing
- **Lines**: 331 (module) + 252 (tests) + 419 (docs) = 1,002 total
- **Ready**: Interface-ready for system integration

**关键特性**:
- 三种预测模式：fixed（固定值）、rule_based（规则）、learned（MLP）
- 资源感知：考虑输入大小和GPU可用性
- 轻量级：learned模式仅~17K参数（可选）
- 灵活接口：可动态切换模式、更新延迟表

### 2.3 System Timeline Snapshot Prediction（未来资源地图）✅

- [x] 作为系统输入信息提供，提供数据以csv形式给出，数据项包括，[time_ms, cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]，这里给出的time指的是从向系统提交时刻起的时间，该csv也是作为提交的输入存放在 `input/system_profiling.csv`,这是系统剩余资源的画像信息
- [x] 从上述数据中读取出该系统资源画像数据，并提供出 T_inf后的资源画像的信息，用于下一个模块的输入
- [x] SystemTimeline: CSV loading with 3 interpolation methods (linear/nearest/previous)
- [x] ResourcePredictor: Predict resources at T_inf, default fallback
- [x] 12/12 tests passing (test_timeline.py)
- **Lines**: 380 (module) + 380 (tests) + 520 (docs) = 1,280 total
- **Ready**: Integrated with LatencyPredictor, ready for LLM context input

### 2.4 1D-CNN Temporal Encoder（时序特征提取）✅

- [x] 实现 1D-CNN（滑动窗口卷积）
  - [x] 输入：从T_inf到timeline结束的资源数据 `(T, 4)` → 转置为 `(1, 4, T)`
  - [x] 输出：`v_temporal` (256D) - Temporal Resource Embedding
  - [x] 关注特征：连续空闲窗口长度、峰谷、趋势（通过多尺度kernel [3,5,7] 捕获）
- [x] ResourceNormalizer: minmax/standard/none normalization (3种归一化方法)
- [x] TemporalCNN: 3-layer 1D-CNN, multi-scale kernels, 67K parameters
- [x] TemporalEncoder: Complete pipeline (extract → normalize → CNN → pool → project)
- [x] Integration with LatencyPredictor (2.2) and SystemTimeline (2.3)
- [x] 16/16 tests passing (test_temporal_encoder.py)
- **Lines**: 450 (module) + 420 (tests) + 680 (docs) = 1,550 total
- **Parameters**: 67,136 (all trainable, gradient flow verified)
- **Ready**: v_temporal ready for LLM injection (Section 3.x - prefix tokens)

**验收（动态上下文）** ✅
- [x] 能根据不同队列/曲线输入，生成不同的 `v_temporal` (pairwise distance verified)
- [x] `T_inf`变化会导致取不同时间窗口的曲线 (early-late: 0.13, mid: 0.01)
- [x] Batch processing支持 (处理多个T_inf: shape (B, 256))
- [x] Gradient flow正常 (所有层可训练，反向传播测试通过)

**下一步**: 将 `v_temporal` 注入LLM (Section 3.x)
- Recommended: 方案A - 作为prefix token（投影成若干"虚拟token"）

---

## 3. Center Panel：LLM Backbone & Reasoning（Qwen2.5-7B）✅

- [x] **集成 Qwen2.5-7B（推理/训练均可）**
  - [x] tokenizer、权重加载、device映射
  - [x] 支持在forward时接入额外条件向量（工具集合、temporal embedding）
- [x] **双模型配置**
  - [x] 生产：Qwen2.5-7B-Instruct (3584D, CUDA, BF16, flash-attn-2)
  - [x] 测试：Qwen2.5-0.5B-Instruct (896D, CPU, FP32)
- [x] **ContextProjector（上下文投影器）**
  - [x] temporal (256D) → llm_hidden_dim
  - [x] task (896D) → llm_hidden_dim
  - [x] tool (1024D) → llm_hidden_dim
  - [x] 可配置LLM维度（896/3584）
  - [x] 参数量：~3.6M (test) / ~13M (default)
- [x] **QwenBackbone（核心模块）**
  - [x] 模型加载：AutoModelForCausalLM + AutoTokenizer
  - [x] 上下文注入：prepare_inputs_with_context (prefix token方案)
  - [x] 生成接口：generate (baseline / context-aware)
  - [x] 批量推理：单上下文→多提示词扩展
- [x] **TOMASSLLMModel（完整集成）**
  - [x] 组件集成：TaskEncoder + TemporalEncoder + LatencyPredictor + QwenBackbone
  - [x] encode_context：端到端上下文编码
  - [x] forward：前向传播（支持teacher forcing）
  - [x] generate：自回归生成
  - [x] from_config：配置文件工厂方法
- [x] **配置文件**
  - [x] configs/simple-test.yaml: 0.5B配置
  - [x] configs/default.yaml: 7B配置
- [x] **测试验证**
  - [x] test_llm_integration.py: 10项全面测试
  - [x] 所有测试通过（10/10）
- **Lines**: 438 (qwen_backbone) + 220 (model_wrapper) + 300 (tests) = 958 total
- **Ready**: 完整LLM推理链路，可进行训练和生成

**验收（骨干）** ✅
- [x] 给定同一prompt，在不同资源曲线条件下能输出不同资源分配参数（批量测试通过）
- [x] 上下文注入有效（context-aware生成与baseline有明显差异）
- [x] 支持batch generation（单上下文扩展到多个提示词）

**下一步**: 
- [ ] 训练策略选择（LoRA微调 / 全参数SFT）
- [ ] 输出解析（从生成文本提取工具选择和资源配置）

---

## 4. Right Panel：Output Generation & Parsing（输出生成与解析）

### 4.1 Token Type Gate / Switch（工具规划token）

- [ ] 定义特殊token集合（不在普通vocab语义里）：
  - [ ] 例如：`<TOOL_PLAN>`、`<TOOL_ID>`、`<RESOURCE_CFG>`、`</TOOL_PLAN>`
- [ ] 实现token gate：
  - [ ] 当decoder输出落入该token类型 → 走“工具分配分支”
  - [ ] 否则 → 标准LM head生成解释文本

### 4.2 Path A：Standard LM Head（解释性文本）

- [ ] 保持原生LM head行为
- [ ] 约束输出模板（最小）：允许返回简短解释 + tool plan

### 4.3 Path B：Resource Allocation Head（专用资源分配头）

#### 4.3.1 Tool Classifier（Softmax）

- [ ] 输入：LLM隐状态 + 工具集合表示（来自静态工具注意力模块）
- [ ] 输出：工具ID分布（softmax over tools）
- [ ] 训练数据：
  - [ ] 最naive：构造规则数据（指令关键词 → 工具）
  - [ ] 迭代：从真实轨迹/人工标注采样

#### 4.3.2 Resource Regressor（MLP回归）

- [ ] 输入：隐状态 + 选定工具embedding + temporal embedding
- [ ] 输出：资源参数（示例）：`cpu_core`, `cpu_mem_gb`, `gpu_sm`, `gpu_mem_gb`（以及可选 `timeout_ms`）
- [ ] 约束与裁剪：
  - [ ] 资源不得超过未来曲线可用上限（可用clamp或损失惩罚）

### 4.4 Final Executable Tool Plan（最终计划格式）

- [ ] 定义最终可执行计划JSON schema（推理输出解析目标）
  - [ ] 示例字段：
    - [ ] `tool_id`（或`tool_name`）
    - [ ] `resource_config`: `{cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb}`
    - [ ] `input`（工具输入参数，可选）
    - [ ] `expected_latency_ms`（可选）
- [ ] 实现解析器：从模型输出token序列 → 结构化plan
- [ ] 实现执行器stub：暂不真正调用外部工具，仅打印/模拟执行

**验收（输出端）**
- [ ] 模型能输出严格符合schema的Tool Plan（解析无报错）
- [ ] Tool Classifier可在小数据集上达到可用准确率（原型≥70%即可）

---

## 5. 训练/数据/评估（建议最小闭环）

- [ ] 先做“合成数据闭环”跑通：
  - [ ] 合成指令集（N条）+ 标注工具ID + 资源需求
  - [ ] 合成资源曲线（忙/闲/碎片化）
- [ ] 定义loss：
  - [ ] 分类：cross-entropy（tool_id）
  - [ ] 回归：MSE/Huber（资源）
  - [ ] 约束：超配惩罚（超过可用资源）
- [ ] 定义评估指标：
  - [ ] Tool Acc
  - [ ] Resource MAE/MAPE
  - [ ] Feasibility Rate（资源分配可行比例）
  - [ ] End-to-end Plan Parse Success Rate

---

## 6. 最小Demo里程碑（按优先级）

- [ ] M0：数据读写跑通（tools.json + profiling.csv + 校验）
- [ ] M1：静态工具嵌入（Tool Encoder + Resource MLP + concat）
- [ ] M2：动态资源曲线 + 1D-CNN temporal embedding（naive曲线即可）
- [ ] M3：Qwen2.5-7B接入 + token gate（能输出plan骨架）
- [ ] M4：tool classifier + resource regressor（可训练/可推理）
- [ ] M5：plan解析为JSON并通过schema校验

---

## 7. 开放问题（需要你确认的最小集合）

- [ ] 工具集合规模预期（10/100/1000+）？影响classifier与attention实现方式
- [ ] 资源参数是否必须包含GPU（或可选）？
- [ ] Tool Plan输出是“单步一个工具”还是“多步工具链”？（当前TODO按单步最小原型）
