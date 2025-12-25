# Phase 0: Encoder Pre-training - Quick Start Guide

This guide shows how to pre-train the resource encoder and export classifier weights.

## Overview

Phase 0 implements the "self-encoding" approach where the encoder learns to compress 6D resource vectors into LLM-understandable embeddings by forcing the LLM to reconstruct natural language descriptions from these embeddings.

**Architecture:**
- **Stream A**: Frozen Qwen2.5 embeddings (semantic anchoring)
- **Stream B**: Trainable ResourceMLP (6D → 3584D)
- **Fusion**: Trainable 8-head self-attention

**Training Goal:** Overfit the encoder to memorize all 1701 configurations.

## Prerequisites

```bash
# Ensure you have the required packages
pip install torch transformers pandas tqdm wandb

# Verify data files exist
ls data/tool_registry/tools.json
ls data/profiling/profiling.csv
```

## Step 1: Pre-train the Encoder

Train the encoder to learn resource-to-semantic mappings:

### Using YAML Configuration (Recommended)

```bash
# Use default configuration
python script/pretrain_encoder.py --config configs/pretrain_encoder.yaml

# Override specific parameters
python script/pretrain_encoder.py \
  --config configs/pretrain_encoder.yaml \
  --batch_size 64 \
  --num_epochs 50 \
  --lr 2e-4
```

### Configuration File

Edit `configs/pretrain_encoder.yaml` to customize:

```yaml
# Data settings
data:
  augmentation:
    mode: "both"  # jitter + variation
    num_copies: 10  # 10x augmentation

# Training settings
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1.0e-4
  target_loss: 0.01  # Early stopping

# Logging
logging:
  wandb:
    enabled: false  # Set to true for W&B tracking
```

### Legacy: Command-line Only (Still Supported)

```bash
python script/pretrain_encoder.py \
  --tool_registry data/tool_registry/tools.json \
  --profiling_data data/profiling/profiling.csv \
  --llm_model Qwen/Qwen2.5-7B \
  --batch_size 32 \
  --num_epochs 100 \
  --lr 1e-4 \
  --target_loss 0.01 \
  --augmentation_mode both \
  --num_augmented_copies 10 \
  --output_dir assets \
  --checkpoint_freq 10 \
  --device cuda
```

**Note:** Command-line arguments override YAML config values.

**Key Parameters:**
- YAML Config: See `configs/pretrain_encoder.yaml` for all options
- `--config`: Path to YAML configuration file (default: `configs/pretrain_encoder.yaml`)
- `--batch_size`: Override batch size (adjust based on GPU memory)
- `--num_epochs`: Override max epochs (default: 100, stops early if target reached)
- `--lr`: Override learning rate (default: 1e-4)
- `--device`: Override device (cuda/cpu)

**Optional: Enable W&B Logging**

Edit `configs/pretrain_encoder.yaml`:
```yaml
logging:
  wandb:
    enabled: true
    project: "tomas-encoder-pretrain"
    entity: "your-username"  # Optional
```

Or use command-line:
```bash
python script/pretrain_encoder.py \
  --config configs/pretrain_encoder.yaml \
  --log_wandb
```

**Expected Output:**
```
Loading configuration from: configs/pretrain_encoder.yaml
================================================================================
TOMAS-LLM Encoder Pre-training
================================================================================
Config: configs/pretrain_encoder.yaml
Device: cuda
LLM Model: Qwen/Qwen2.5-7B
Batch Size: 32
Learning Rate: 0.0001
Max Epochs: 100
Target Loss: 0.01
Augmentation: both (×10)
================================================================================

[1/5] Loading dataset...
  ✓ Loaded 17010 samples from 1701 configs
  ✓ Augmentation factor: 10×

...

Epoch 50/100 - Train Loss: 0.1234
  ✓ New best model saved (loss: 0.1234)

🎉 Target loss 0.01 reached! (current: 0.0095)
Training complete - encoder has successfully memorized all configurations!
```

**Saved Files:**
- `assets/pretrained_encoder.pt` - Best model checkpoint
- `assets/encoder_checkpoint_epoch*.pt` - Periodic checkpoints

## Step 2: Export Classifier Weights

Convert the trained encoder into a weight matrix for the main model:

```bash
python script/export_classifier_weights.py \
  --encoder_checkpoint assets/pretrained_encoder.pt \
  --tool_registry data/tool_registry/tools.json \
  --profiling_data data/profiling/profiling.csv \
  --llm_model Qwen/Qwen2.5-7B \
  --output_dir assets \
  --batch_size 64 \
  --device cuda
```

**Expected Output:**
```
[3/4] Generating weight matrix...
100%|████████████████| 27/27 [00:15<00:00,  1.74it/s]
  ✓ Generated weight matrix: [1701, 3584]

Saving Assets:
  ✓ Saved weight matrix to assets/config_weights.pt
    Shape: [1701, 3584]
  ✓ Saved config lookup to assets/config_lookup.json
    Entries: 1701
  ✓ Saved tool mask map to assets/tool_mask_map.json
    Tools: 7
```

**Generated Files:**

1. **`config_weights.pt`** - Weight matrix [1701, 3584]
   - Used to initialize `HierarchicalDecoder.config_head.weight`
   
2. **`config_lookup.json`** - Config ID → Resource mapping
   ```json
   {
     "0": {
       "tool_name": "image_classification",
       "input_size": "small",
       "cpu_cores": 2,
       "memory_gb": 4.0,
       "gpu_sm": 20,
       "gpu_memory_gb": 2.0,
       "latency_ms": 150.0
     },
     ...
   }
   ```

3. **`tool_mask_map.json`** - Tool → Valid Config IDs
   ```json
   {
     "image_classification": [0, 1, 2, ..., 242],
     "text_summarization": [243, 244, ..., 485],
     ...
   }
   ```

## Validation

Verify the exported weights:

```python
import torch
import json

# Load weight matrix
data = torch.load("assets/config_weights.pt")
print(f"Weight matrix shape: {data['weights'].shape}")  # [1701, 3584]
print(f"Number of configs: {data['num_configs']}")      # 1701
print(f"Hidden dimension: {data['hidden_dim']}")        # 3584

# Load lookup tables
with open("assets/config_lookup.json") as f:
    config_lookup = json.load(f)
    print(f"Config lookup entries: {len(config_lookup)}")

with open("assets/tool_mask_map.json") as f:
    tool_mask_map = json.load(f)
    print(f"Tools in mask map: {list(tool_mask_map.keys())}")
    for tool, configs in tool_mask_map.items():
        print(f"  {tool}: {len(configs)} configurations")
```

## Next Steps

After completing Phase 0, proceed to:

1. **Phase 3**: Implement `HierarchicalDecoder` in `src/decoders/hierarchical.py`
   - Initialize `config_head.weight` with `config_weights.pt`
   - Implement masking logic using `tool_mask_map.json`

2. **Phase 4**: Update main model training pipeline
   - Load pre-trained weights into decoder
   - Train end-to-end with task → (tool_id, config_id) → lookup

## Troubleshooting

### OOM (Out of Memory)
- Reduce `--batch_size` (try 16 or 8)
- Use gradient accumulation
- Use mixed precision training (add `--fp16`)

### Slow Training
- Increase `--batch_size` if GPU memory allows
- Reduce `--num_augmented_copies` (try 5x instead of 10x)
- Use fewer workers in dataloader

### Loss Not Decreasing
- Check learning rate (try 5e-5 or 2e-4)
- Verify data loading is correct
- Ensure Stream A is actually frozen
- Check that embeddings are properly injected

### Export Fails
- Verify checkpoint file exists and is valid
- Ensure tool registry and profiling data match pre-training
- Check device compatibility (CPU vs GPU)

## Configuration Tips

**For Fast Prototyping:**

Edit `configs/pretrain_encoder.yaml`:
```yaml
data:
  augmentation:
    num_copies: 3  # Less augmentation
    
training:
  batch_size: 64  # Larger batches
  num_epochs: 50  # Fewer epochs
  target_loss: 0.05  # Less strict
```

Or use command-line overrides:
```bash
python script/pretrain_encoder.py \
  --config configs/pretrain_encoder.yaml \
  --batch_size 64 \
  --num_epochs 50
```

**For Production Quality:**

```yaml
data:
  augmentation:
    num_copies: 20  # More augmentation
    
training:
  batch_size: 16  # Smaller batches (better convergence)
  num_epochs: 200  # More epochs
  target_loss: 0.005  # Stricter threshold
  
logging:
  wandb:
    enabled: true  # Track experiments
```

## Expected Training Time

**On NVIDIA RTX 4090:**
- Dataset: ~17,000 samples (1701 × 10x augmentation)
- Batch size: 32
- Time per epoch: ~2-3 minutes
- Total training: ~2-5 hours (to reach loss < 0.01)

**On CPU:**
- Not recommended (expect 10-20× slower)

## Files Summary

```
script/
  pretrain_encoder.py          # Main pre-training script
  export_classifier_weights.py # Weight export script

src/offline/
  text_template.py             # Natural language templates
  pretrain_encoder.py          # Encoder architecture
  
src/data/
  pretrain_dataset.py          # Dataset with augmentation

assets/
  pretrained_encoder.pt        # Trained encoder checkpoint
  config_weights.pt            # [1701, 3584] weight matrix
  config_lookup.json           # Config ID → Resource mapping
  tool_mask_map.json           # Tool → Config IDs mapping
```
