# Implementation Documentation: Section 2.4 - 1D-CNN Temporal Encoder

**实现日期**: 2024-12-24  
**状态**: ✅ Complete

---

## 1. Overview

Section 2.4 implements **1D-CNN Temporal Encoder** for extracting temporal patterns from system resource timeline. This module:
- Extracts timeline data from T_inf (predicted latency) onwards
- Normalizes resource values to neural network-friendly ranges
- Applies multi-scale 1D-CNN to capture temporal features
- Outputs v_temporal embedding for LLM context injection

**Key Features**:
- Multi-scale temporal pattern detection (short/medium/long-term)
- Configurable normalization (minmax/standard)
- Batch processing support
- Integration with LatencyPredictor (Section 2.2) and SystemTimeline (Section 2.3)

---

## 2. Architecture

### 2.1 Component Hierarchy

```
TemporalEncoder (Complete Pipeline)
    ├── SystemTimeline (Section 2.3)
    │   └── Extract timeline window from T_inf onwards
    ├── ResourceNormalizer
    │   └── Normalize [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]
    └── TemporalCNN (1D-CNN)
        ├── Conv1D (kernel=3) → ReLU → BatchNorm  # Short-term
        ├── Conv1D (kernel=5) → ReLU → BatchNorm  # Medium-term
        ├── Conv1D (kernel=7) → ReLU → BatchNorm  # Long-term
        ├── AdaptiveAvgPool1d(1)
        └── Linear → v_temporal (256D)
```

### 2.2 Data Flow

```
Input: T_inf (predicted latency)
   ↓
Extract Timeline Window (T_inf → T_end)
   ↓ [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb] × T timesteps
Normalize (minmax → [0, 1])
   ↓ (T, 4) → transpose → (4, T) → unsqueeze → (1, 4, T)
1D-CNN (3 layers, kernels [3, 5, 7])
   ↓ (1, 64, T) after each conv layer
AdaptiveAvgPool1d
   ↓ (1, 64, 1) → squeeze → (64,)
Linear Projection
   ↓
Output: v_temporal (256,)
```

---

## 3. Implementation Details

### 3.1 ResourceNormalizer

**Purpose**: Normalize resource values for neural network input

**Methods**:

1. **MinMax Normalization** (default)
   ```python
   normalized = (x - min) / (max - min)
   # Result: values in [0, 1]
   ```
   
   Resource ranges:
   - cpu_cores: [0, 32] cores
   - cpu_mem_gb: [0, 128] GB
   - gpu_sm: [0, 100] %
   - gpu_mem_gb: [0, 80] GB

2. **Standard Normalization**
   ```python
   normalized = (x - mean) / std
   # Result: zero mean, unit variance
   ```

3. **None** (no normalization)

**Key Features**:
- Supports 2D (T, 4) and 3D (B, T, 4) tensors
- Bidirectional: normalize() and denormalize()
- Fit to data statistics (for standard normalization)

### 3.2 TemporalCNN

**Architecture**:
```python
TemporalCNN(
    in_channels=4,          # Resource types
    hidden_channels=64,     # Hidden dimension
    output_dim=256,         # v_temporal dimension
    num_layers=3,           # Number of conv layers
    kernel_sizes=[3, 5, 7], # Multi-scale kernels
    pooling='adaptive_avg'  # Pooling method
)
```

**Convolutional Layers**:
- **Layer 1**: Conv1d(4 → 64, kernel=3) + ReLU + BatchNorm
- **Layer 2**: Conv1d(64 → 64, kernel=5) + ReLU + BatchNorm
- **Layer 3**: Conv1d(64 → 64, kernel=7) + ReLU + BatchNorm

**Padding**: Same padding (kernel_size // 2) to preserve temporal length

**Pooling Options**:
- `adaptive_avg`: Adaptive average pooling (default)
- `adaptive_max`: Adaptive max pooling
- `flatten`: Flatten all timesteps (for very short sequences)

**Parameters**: 67,136 (trainable)

### 3.3 TemporalEncoder

**Complete Pipeline**:

1. **Extract Timeline Window**
   ```python
   timeline_window = extract_timeline_window(t_inf_ms, t_end_ms)
   # Result: (num_timesteps, 4) tensor
   # num_timesteps ∈ [min_timesteps, max_timesteps]
   # Default: [5, 50] with 100ms granularity
   ```

2. **Normalize**
   ```python
   normalized = normalizer.normalize(timeline_window)
   # Result: (T, 4) with values in [0, 1]
   ```

3. **Reshape for CNN**
   ```python
   cnn_input = normalized.transpose(0, 1).unsqueeze(0)
   # (T, 4) → (4, T) → (1, 4, T)
   # batch=1, channels=4, time_steps=T
   ```

4. **Apply CNN**
   ```python
   v_temporal = cnn(cnn_input)
   # (1, 4, T) → (1, 256) → (256,)
   ```

**Tensor Input Handling**:
- Scalar float: Single embedding (256,)
- Tensor(1,): Single embedding (from LatencyPredictor)
- Tensor(B,): Batch of embeddings (B, 256)

---

## 4. Usage Examples

### 4.1 Basic Usage

```python
from context.temporal_encoder import TemporalEncoder
from context.timeline import SystemTimeline

# Load timeline
timeline = SystemTimeline("input/system_profiling.csv")

# Create encoder
encoder = TemporalEncoder(
    timeline=timeline,
    cnn_config={
        'in_channels': 4,
        'hidden_channels': 64,
        'output_dim': 256,
        'num_layers': 3
    }
)

# Encode from T_inf onwards
v_temporal = encoder(t_inf_ms=1000.0)
# Shape: (256,)
```

### 4.2 Integration with Latency Predictor

```python
from context.latency_predictor import LatencyPredictor
from context.temporal_encoder import TemporalEncoder

# Setup
latency_predictor = LatencyPredictor(mode='fixed', fixed_latency_ms=1000)
temporal_encoder = TemporalEncoder.from_config(config)

# Predict latency
t_inf = latency_predictor()  # torch.Tensor([1000.0])

# Encode temporal features
v_temporal = temporal_encoder(t_inf)  # (256,)

print(f"Predicted latency: {t_inf.item():.0f}ms")
print(f"Temporal embedding shape: {v_temporal.shape}")
```

### 4.3 Batch Processing

```python
# Multiple T_inf values
t_inf_batch = torch.tensor([500.0, 1000.0, 1500.0, 2000.0])

# Encode all
v_temporal_batch = encoder(t_inf_batch)
# Shape: (4, 256)

# Different T_inf → different embeddings
print("Embedding norms:")
for i, t in enumerate(t_inf_batch):
    print(f"  T_inf={t:.0f}ms: norm={v_temporal_batch[i].norm():.4f}")
```

### 4.4 Custom Normalization

```python
from context.temporal_encoder import ResourceNormalizer

# Standard normalization (z-score)
normalizer = ResourceNormalizer(method='standard')

encoder = TemporalEncoder(
    timeline=timeline,
    normalizer=normalizer,
    cnn_config={...}
)

# Fit normalizer to data (optional)
sample_data = torch.randn(100, 4)  # Sample timeline data
normalizer.fit(sample_data)
```

### 4.5 From Config

```yaml
# configs/simple-test.yaml
runtime:
  temporal_encoder:
    enabled: true
    normalization: "minmax"
    hidden_channels: 64
    output_dim: 256
    num_layers: 3
    pooling: "adaptive_avg"
    min_timesteps: 5
    max_timesteps: 50
    time_granularity_ms: 100
```

```python
import yaml
from context.temporal_encoder import TemporalEncoder

with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

encoder = TemporalEncoder.from_config(config)
v_temporal = encoder(1000.0)
```

---

## 5. Test Results

**Test File**: `tests/test_temporal_encoder.py`  
**Status**: ✅ All 16 tests passing

### Test Coverage

1. ✅ ResourceNormalizer - MinMax (range [0, 1] verified)
2. ✅ ResourceNormalizer - Standard (mean≈0, std≈1 verified)
3. ✅ ResourceNormalizer - Batch Input (3D tensor handling)
4. ✅ TemporalCNN - Architecture (67,136 parameters)
5. ✅ TemporalCNN - Forward Pass (batch×channels×time → batch×output_dim)
6. ✅ TemporalCNN - Pooling Methods (adaptive_avg/adaptive_max)
7. ✅ Load Timeline (31 snapshots, 0-3000ms)
8. ✅ TemporalEncoder - Extract Timeline Window (26 timesteps from T_inf=500ms)
9. ✅ TemporalEncoder - Forward Pass Scalar (single embedding)
10. ✅ TemporalEncoder - Forward Pass Tensor (LatencyPredictor output)
11. ✅ TemporalEncoder - Batch T_inf (different embeddings verified)
12. ✅ TemporalEncoder - Time Window Variation (early vs late distances)
13. ✅ TemporalEncoder from Config (YAML loading)
14. ✅ Integration with LatencyPredictor (seamless integration)
15. ✅ Parameter Count (67K trainable)
16. ✅ Gradient Flow (all layers have gradients)

### Test Output Summary

```
============================================================
ALL TESTS PASSED ✓
============================================================

Summary:
  - ResourceNormalizer: minmax/standard/none methods
  - TemporalCNN: 3-layer 1D-CNN with configurable kernels
  - TemporalEncoder: Complete pipeline (extract → normalize → CNN)
  - Timeline window extraction from T_inf onwards
  - Batch processing support
  - Integration with LatencyPredictor ✓
  - Total parameters: 67,136
  - v_temporal dimension: 256
  ✓ Ready for LLM injection!
```

### Key Validations

**Normalization**:
- MinMax: Range [0.250, 0.800] → correct
- Standard: Mean ≈ 0, Std ≈ 1 → verified
- Denormalization: Reversible → correct

**CNN Architecture**:
- Input: (batch=2, channels=4, time=20)
- Output: (batch=2, output_dim=256)
- Parameters: 67,136

**Temporal Variation**:
- T_inf=0ms vs T_inf=2500ms: distance = 0.1314
- T_inf=1000ms vs T_inf=1500ms: distance = 0.0120
- **Different time windows produce different embeddings** ✓

**Integration**:
- LatencyPredictor → TemporalEncoder seamless
- Handles tensor(1,) from LatencyPredictor correctly

---

## 6. Integration Points

### 6.1 Inputs

**From Section 2.2 (Latency Prediction)**:
```python
t_inf = latency_predictor()  # torch.Tensor([1000.0])
```

**From Section 2.3 (System Timeline)**:
```python
timeline = SystemTimeline("input/system_profiling.csv")
# Provides: timeline.get_batch_snapshots(time_points)
```

### 6.2 Outputs

**To Section 3.x (LLM Context)**:
```python
v_temporal = temporal_encoder(t_inf)  # (256,)
# This embedding will be injected into LLM as:
# - Option A: Prefix tokens (project to LLM vocab space)
# - Option B: Cross-attention conditioning
```

### 6.3 Dependencies

**Internal Modules**:
- `context.timeline.SystemTimeline`: Timeline data management
- Integration-ready with `context.latency_predictor.LatencyPredictor`

**PyTorch Layers**:
- `nn.Conv1d`: 1D convolution for temporal patterns
- `nn.BatchNorm1d`: Normalization after each conv
- `nn.AdaptiveAvgPool1d`: Pooling to fixed-size representation
- `nn.Linear`: Projection to output dimension

---

## 7. Configuration

### 7.1 YAML Settings

```yaml
runtime:
  # Timeline source (Section 2.3)
  timeline:
    csv_path: "input/system_profiling.csv"
    interpolation: "linear"
  
  # Temporal Encoder (Section 2.4)
  temporal_encoder:
    enabled: true
    normalization: "minmax"      # minmax | standard | none
    hidden_channels: 64          # CNN hidden dimension
    output_dim: 256              # v_temporal dimension
    num_layers: 3                # Number of conv layers
    pooling: "adaptive_avg"      # adaptive_avg | adaptive_max | flatten
    min_timesteps: 5             # Minimum timesteps to extract
    max_timesteps: 50            # Maximum timesteps to extract
    time_granularity_ms: 100     # Time step size (ms)
```

### 7.2 Kernel Size Selection

Default: `[3, 5, 7]` for 3 layers

- **Kernel=3**: Short-term fluctuations (300ms window)
- **Kernel=5**: Medium-term trends (500ms window)
- **Kernel=7**: Long-term patterns (700ms window)

Adjustable via:
```python
cnn_config = {
    'kernel_sizes': [3, 5, 7, 9]  # For 4 layers
}
```

---

## 8. Performance Characteristics

### 8.1 Computational Complexity

**Timeline Extraction**:
- Time: O(T) where T = num_timesteps
- Space: O(T × 4) for timeline window

**Normalization**:
- Time: O(T × 4)
- Space: O(1) auxiliary

**CNN Forward Pass**:
- Time: O(L × C × K × T) where:
  - L = num_layers (3)
  - C = hidden_channels (64)
  - K = kernel_size (3-7)
  - T = num_timesteps (5-50)
- Space: O(C × T) for activations

**Total**:
- Forward pass: ~5ms on CPU for T=26
- Memory: ~100KB per batch

### 8.2 Parameter Count

```
Conv1d(4→64, k=3):   4×64×3 + 64 = 832
BatchNorm1d(64):     64×2 = 128
Conv1d(64→64, k=5):  64×64×5 + 64 = 20,544
BatchNorm1d(64):     128
Conv1d(64→64, k=7):  64×64×7 + 64 = 28,736
BatchNorm1d(64):     128
Linear(64→256):      64×256 + 256 = 16,640
--------------------------------
Total:               67,136 parameters
```

All parameters are trainable.

---

## 9. Design Decisions

### 9.1 Why 1D-CNN over RNN/LSTM?

**Pros of 1D-CNN**:
- **Parallel processing**: All timesteps processed simultaneously
- **Multi-scale**: Different kernel sizes capture different temporal patterns
- **Local patterns**: Effective for detecting resource spikes/drops
- **Efficiency**: Faster than sequential RNN/LSTM

**Cons**:
- Less effective for very long-term dependencies
- Requires sufficient timesteps for meaningful patterns

**Decision**: CNN is sufficient for resource timelines (5-50 timesteps, 100ms granularity)

### 9.2 Why MinMax Normalization?

- **Bounded range**: Resource values have natural bounds
- **Interpretability**: [0, 1] range is intuitive
- **Stability**: No explosion from outliers
- **Reversibility**: Easy to denormalize for inspection

### 9.3 Why 256D Output?

- **LLM compatibility**: Common embedding dimension
- **Sufficient capacity**: Captures temporal patterns without over-parameterization
- **Efficiency**: Reasonable size for prefix token injection

### 9.4 Why Extract from T_inf Onwards?

**Rationale**:
- **Relevance**: Future resource availability matters for tool execution
- **Dynamic**: Different T_inf → different windows → different embeddings
- **Predictive**: Captures resource trends after tool starts

**Example**:
- Tool predicted to complete at T_inf=1000ms
- Extract timeline [1000ms, 3000ms]
- Embedding captures: resource recovery after tool execution

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

1. **Fixed Window**: Extracts from T_inf to timeline end
2. **No Attention**: Equal importance to all timesteps
3. **Single Timeline**: One resource timeline per sample
4. **Untrained**: Random initialization, requires training for meaningful patterns

### 10.2 Future Enhancements

1. **Learnable Window Size**:
   ```python
   window_predictor = nn.Linear(1, 1)  # T_inf → window_size
   t_end = t_inf + window_predictor(t_inf)
   ```

2. **Temporal Attention**:
   ```python
   attention_weights = softmax(Q @ K.T / sqrt(d))
   weighted_features = attention_weights @ V
   ```

3. **Multi-Timeline Fusion**:
   ```python
   # Encode multiple resource timelines
   v_temporal_cpu = cnn_cpu(cpu_timeline)
   v_temporal_gpu = cnn_gpu(gpu_timeline)
   v_temporal = concat([v_temporal_cpu, v_temporal_gpu])
   ```

4. **Hierarchical CNN**:
   ```python
   # Process at multiple time scales
   v_temporal_fine = cnn_100ms(timeline_100ms)
   v_temporal_coarse = cnn_1s(timeline_1s)
   v_temporal = fusion([v_temporal_fine, v_temporal_coarse])
   ```

---

## 11. Files Created

```
src/context/temporal_encoder.py          # 450 lines - Complete implementation
tests/test_temporal_encoder.py           # 420 lines - 16 comprehensive tests
docs/implementation_2_4.md               # This file - Documentation
configs/simple-test.yaml                 # Updated with temporal_encoder config
```

---

## 12. Summary

Section 2.4 is now **complete** with:

✅ **ResourceNormalizer**: MinMax/Standard/None normalization (3 methods)  
✅ **TemporalCNN**: 3-layer 1D-CNN with multi-scale kernels (67K params)  
✅ **TemporalEncoder**: Complete pipeline (extract → normalize → CNN)  
✅ **Timeline Extraction**: Dynamic window from T_inf onwards  
✅ **Batch Processing**: Efficient handling of multiple T_inf values  
✅ **Integration**: Seamless with LatencyPredictor (2.2) and SystemTimeline (2.3)  
✅ **Testing**: 16/16 tests passing, all features validated  
✅ **Documentation**: Complete usage examples and API reference  
✅ **Configuration**: YAML-based setup with sensible defaults  

**v_temporal is ready for LLM injection!**

---

## 13. Next Steps

### 13.1 LLM Integration (Section 3.x)

Two options for injecting v_temporal:

**Option A: Prefix Tokens (Recommended)**
```python
# Project v_temporal to LLM embedding space
v_temporal_proj = nn.Linear(256, llm_hidden_dim)  # 256 → 3584
temporal_tokens = v_temporal_proj(v_temporal).unsqueeze(0)  # (1, 3584)

# Concatenate with input embeddings
input_embeds = torch.cat([temporal_tokens, task_embeds, tool_embeds], dim=0)
# Shape: (1 + task_len + tool_len, 3584)

# Feed to LLM
output = llm(inputs_embeds=input_embeds)
```

**Option B: Cross-Attention**
```python
# Add cross-attention layer in LLM
cross_attn = MultiheadAttention(embed_dim=3584, num_heads=28)

# Query from LLM hidden states, Key/Value from v_temporal
attended = cross_attn(
    query=llm_hidden,
    key=v_temporal_expanded,
    value=v_temporal_expanded
)
```

### 13.2 Training

**Loss Function**:
```python
# Contrastive learning: Similar timelines → similar embeddings
loss = contrastive_loss(v_temporal_i, v_temporal_j, label)

# Or supervised: Predict resource bottleneck
bottleneck = classify(v_temporal)  # CPU | GPU | Memory | None
loss = cross_entropy(bottleneck, target)
```

**Augmentation**:
- Time shifting: Start from different T_inf
- Noise injection: Simulate measurement errors
- Timeline interpolation: Smooth/rough resource curves

### 13.3 Evaluation

**Metrics**:
- Embedding similarity vs timeline similarity
- Resource bottleneck prediction accuracy
- Tool selection improvement with temporal context

---

**Temporal encoding completes the runtime context pipeline. Ready for LLM backbone integration!** 🚀
