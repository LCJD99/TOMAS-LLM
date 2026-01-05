# TOMAS-LLM: Tool-Aware Resource Management and Selection with LLMs

**TOMAS-LLM** is a framework for tool planning with resource-aware allocation using Large Language Models (Qwen2.5-7B). The system learns to select and plan tool execution while considering resource constraints across multiple dimensions (CPU cores, CPU memory, GPU SMs, GPU memory).

## 🎯 Project Overview

This project tackles the problem of:
- **Tool Planning**: Selecting the right tools for complex tasks
- **Resource Allocation**: Assigning optimal computational resources to each tool
- **Latency Optimization**: Minimizing execution time under resource constraints

### Two-Stage Training Approach

**Stage 1: Tool Token Learning**
- Expand vocabulary with tool+profiling tokens
- Each token represents a tool configuration with specific resource requirements
- Train embeddings using LoRA + Embedding initialization

**Stage 2: Tool Planning** (Framework in development)
- Learn to plan multi-step tool execution
- Optimize for resource utilization and task completion

## 📊 Data Schema

Each tool configuration includes:
- Tool functionality description
- Input size category (small/medium/large)
- Resource allocation: CPU cores, CPU memory, GPU SMs, GPU memory
- Profiling data: Execution latency for the configuration

**Example Virtual Token**: `<IMG_CLS_SMALL_LOW_MED_HIGH_LOW>`
- Tool: Image Classification
- Input: Small
- Resources: Low CPU cores, Medium CPU memory, High GPU SMs, Low GPU memory

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/TOMAS-LLM.git
cd TOMAS-LLM

# Install in editable mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Directory Structure

```
TOMAS-LLM/
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # Processed training data
│   └── registry/               # Tool registry (token mappings)
├── checkpoints/                # Model weights
├── configs/                    # Training configurations
├── scripts/                    # Shell scripts for training
├── src/
│   ├── data/                   # Data processing modules
│   ├── models/                 # Model architectures
│   ├── engine/                 # Training/evaluation engines
│   ├── datasets/               # Dataset implementations
│   ├── tokenization/           # Tokenizer expansion
│   ├── inference/              # Inference utilities
│   └── utils/                  # Helper functions
└── tests/                      # Unit tests
```

## 📝 Development Workflow

Follow the detailed implementation plan in [TODO.md](TODO.md).

### Phase 0: Environment Setup ✅
- [x] Create project structure
- [x] Setup dependencies

### Phase 1: Data Processing
- [ ] Analyze profiling data distribution
- [ ] Build tool registry
- [ ] Generate instruction training data

### Phase 2: Model Preparation
- [ ] Expand tokenizer with virtual tokens
- [ ] Initialize embeddings (non-random)

### Phase 3: Stage 1 Training
- [ ] Implement training pipeline
- [ ] Train with LoRA + Embedding learning

### Phase 4: Stage 2 Framework
- [ ] Design planning task format
- [ ] Implement planning trainer

## 🛠️ Usage

### Data Processing

```bash
# Build tool registry
python src/data/build_registry.py \
    --tools data/raw/tools.json \
    --profiling data/raw/profiling.csv \
    --output data/registry/tool_registry.json

# Generate training data
python src/data/build_instruction_data.py \
    --registry data/registry/tool_registry.json \
    --output_dir data/processed/ \
    --augment_factor 5
```

### Training

```bash
# Stage 1: Tool token learning
bash scripts/train_stage1.sh
```

### Inference

```bash
# Predict optimal tool configuration
python src/inference/predict.py \
    --model checkpoints/stage1/final_model/ \
    --input "Classify images with 4 CPU cores, 8GB RAM, 40 SMs, 4GB VRAM"
```

## 📚 Documentation

- [TODO.md](TODO.md) - Detailed implementation checklist
- [configs/](configs/) - Training configuration examples

## 🔬 Key Features

- **Semantic Token Initialization**: Embeddings initialized from tool descriptions, not random
- **Resource-Aware Planning**: Considers multi-dimensional resource constraints
- **Modular Design**: Extensible for additional tools and resource types
- **Efficient Training**: LoRA-based fine-tuning reduces memory requirements

## 📄 License

MIT License

## 🙏 Acknowledgments

- Based on Qwen2.5-7B foundation model
- Inspired by ToolGen methodology for vocabulary expansion

## 📧 Contact

For questions or collaboration, please open an issue or contact the maintainers.

---

**Status**: 🚧 Active Development | **Last Updated**: 2026-01-05
