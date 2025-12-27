# Temporal Encoder Pretraining

## Overview

This module implements modality alignment pretraining for the Temporal Encoder, training a 1D-CNN to translate system resource timeline curves into natural language descriptions that a frozen LLM can understand.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Dataset Generation

Test the synthetic data generation:

```bash
python tools/validate_temporal_dataset.py \
  --num_samples 1000 \
  --num_visualize 20 \
  --output_dir data/temporal_pretrain
```

This will generate a validation report at `data/temporal_pretrain/validation_report.html`.

### 3. Quick Test

Verify the training pipeline works correctly:

```bash
python script/quick_test_temporal.py \
  --config configs/pretrain_temporal.yaml \
  --num_train 100 \
  --num_val 20 \
  --epochs 2
```

This runs a quick test with minimal data to ensure everything is working.

### 4. Full Training

Start full pretraining:

```bash
python script/pretrain_temporal_encoder.py \
  --config configs/pretrain_temporal.yaml \
  --output_dir checkpoints/temporal_pretrain
```

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/temporal_pretrain/tensorboard
```

### 5. Resume Training

Resume from a checkpoint:

```bash
python script/pretrain_temporal_encoder.py \
  --config configs/pretrain_temporal.yaml \
  --resume checkpoints/temporal_pretrain/epoch_5_step_5000.pt
```

## Configuration

Edit `configs/pretrain_temporal.yaml` to customize:

- **Model**: LLM name, embedding dimension, encoder architecture
- **Data**: Number of samples, batch size, task distribution
- **Training**: Learning rate, epochs, gradient clipping
- **Paths**: Output directories, logging

### Key Configuration Options

```yaml
model:
  llm_name: "Qwen/Qwen2.5-0.5B-Instruct"  # or 7B
  llm_embedding_dim: 1024  # 1024 for 0.5B, 3584 for 7B
  
data:
  num_train_samples: 100000
  num_val_samples: 5000
  batch_size: 16
  
training:
  num_epochs: 10
  learning_rate: 1e-4
```

## Architecture

### Components

1. **TemporalEncoder** (`src/context/temporal_encoder.py`)
   - 1D-CNN with multi-scale convolutions
   - MLP Projector to LLM embedding space
   - **Trainable**

2. **TemporalLLMWrapper** (`src/context/temporal_llm_wrapper.py`)
   - Integrates TemporalEncoder with frozen LLM
   - Handles forward pass and loss computation
   - **Partially trainable** (only encoder)

3. **Data Generation** (`src/data/`)
   - `ResourceCurveSynthesizer`: Generates diverse timeline curves
   - `TextDescriptionGenerator`: Creates natural language descriptions
   - `TemporalPretrainDataset`: PyTorch Dataset

### Training Objective

Causal Language Modeling Loss on three task types:

- **Type A (40%)**: Trend Description - trains macro pattern recognition
- **Type B (30%)**: Bottleneck Spotting - trains spike detection
- **Type C (30%)**: Feasibility QA - trains numerical/temporal logic

## File Structure

```
src/
  context/
    temporal_encoder.py          # CNN encoder with projector
    temporal_llm_wrapper.py      # LLM wrapper for training
  data/
    resource_curve_synthesizer.py  # Curve generation
    text_description_generator.py  # Text generation
    temporal_pretrain_dataset.py   # Dataset class

script/
  pretrain_temporal_encoder.py   # Main training script
  quick_test_temporal.py         # Quick test script

tools/
  validate_temporal_dataset.py   # Data validation

configs/
  pretrain_temporal.yaml         # Training configuration

checkpoints/temporal_pretrain/   # Saved checkpoints
logs/temporal_pretrain/          # Training logs
data/temporal_pretrain/          # Data cache
```

## Expected Results

### Training Curve

- **Epoch 1-3**: Loss drops from ~5.0 to ~2.0
- **Epoch 4-7**: Loss drops from ~2.0 to ~1.2
- **Epoch 8-10**: Loss stabilizes around ~1.0

### Validation Metrics

- **Perplexity**: < 3.0 (excellent), < 5.0 (acceptable)
- **Token Accuracy**: > 60% (excellent)

## Training Time Estimates

On single A100 GPU:

- **100k samples × 10 epochs**
  - Qwen2.5-0.5B: ~8-12 hours
  - Qwen2.5-7B: ~24-36 hours

## Troubleshooting

### Out of Memory

1. Reduce `batch_size` in config
2. Use smaller model: `Qwen2.5-0.5B-Instruct`
3. Enable mixed precision: `fp16: true` or `bf16: true`
4. Reduce `hidden_channels` in encoder config

### Slow Training

1. Increase `batch_size` if memory allows
2. Use `num_workers > 0` in data config
3. Enable `pin_memory: true`
4. Use gradient accumulation: `gradient_accumulation_steps: 2`

### Loss Not Decreasing

1. Check learning rate (try 5e-5 or 2e-4)
2. Verify data quality with validation tool
3. Increase warmup steps
4. Check gradient clipping value

## Next Steps

After pretraining:

1. Evaluate on held-out test set
2. Visualize predictions with quality check tool
3. Integrate trained encoder into full pipeline
4. Fine-tune on real system profiling data

## References

- Design document: `design.md`
- Implementation guide: `TODO-context.md`
- Phase 0 usage: `docs/phase0_usage.md`
