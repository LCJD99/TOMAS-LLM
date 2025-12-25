# Configuration Files

This directory contains YAML configuration files for TOMAS-LLM training and inference.

## Available Configurations

### Main Model Configurations

- **`default.yaml`** - Main TOMAS-LLM model configuration
  - LLM backbone settings (Qwen2.5-7B)
  - Context projector dimensions
  - Output heads configuration
  - Training and inference settings

### Phase 0: Encoder Pre-training

- **`pretrain_encoder.yaml`** - Standard encoder pre-training configuration
  - Full 10× data augmentation
  - 100 epochs with early stopping at loss < 0.01
  - Recommended for production use
  - ~2-5 hours training on RTX 4090

- **`pretrain_encoder_quick.yaml`** - Quick test configuration
  - 3× data augmentation (faster)
  - 30 epochs, stops at loss < 0.05
  - For development and testing
  - ~30 minutes training

### Test Configurations

- **`simple-test.yaml`** - Minimal configuration for unit tests

## Usage

### Using a Configuration File

```bash
# Use default pre-training config
python script/pretrain_encoder.py --config configs/pretrain_encoder.yaml

# Use quick test config
python script/pretrain_encoder.py --config configs/pretrain_encoder_quick.yaml
```

### Overriding Configuration Values

Command-line arguments override YAML values:

```bash
# Override batch size and epochs
python script/pretrain_encoder.py \
  --config configs/pretrain_encoder.yaml \
  --batch_size 64 \
  --num_epochs 50

# Override device
python script/pretrain_encoder.py \
  --config configs/pretrain_encoder.yaml \
  --device cpu
```

### Creating Custom Configurations

1. Copy an existing config:
   ```bash
   cp configs/pretrain_encoder.yaml configs/my_config.yaml
   ```

2. Edit the new file with your settings

3. Use it:
   ```bash
   python script/pretrain_encoder.py --config configs/my_config.yaml
   ```

## Configuration Structure

### Encoder Pre-training Config

```yaml
data:
  tool_registry: "path/to/tools.json"
  profiling_data: "path/to/profiling.csv"
  augmentation:
    mode: "both"  # jitter, variation, both, none
    num_copies: 10  # Augmentation factor

model:
  llm_model: "Qwen/Qwen2.5-7B"
  llm_hidden_dim: 3584
  freeze_semantic: true  # Freeze Stream A

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1.0e-4
  target_loss: 0.01  # Early stopping threshold

logging:
  wandb:
    enabled: false
    project: "your-project"

device:
  type: "cuda"  # cuda, cpu, mps
```

## Tips

**For Development:**
- Use `pretrain_encoder_quick.yaml`
- Set `batch_size` higher (64)
- Reduce `num_copies` (3-5)

**For Production:**
- Use `pretrain_encoder.yaml`
- Adjust `batch_size` based on GPU memory
- Increase `num_copies` (15-20) for better quality
- Enable W&B logging

**For Limited GPU Memory:**
```yaml
training:
  batch_size: 8  # Smaller batches
  gradient_accumulation_steps: 4  # Effective batch = 32
```

**For CPU Training:**
```yaml
device:
  type: "cpu"

dataloader:
  num_workers: 0  # Avoid multiprocessing issues
  pin_memory: false
```
