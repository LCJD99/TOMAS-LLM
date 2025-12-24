# Section 2.2: Latency Prediction Module (T_inf)

**Status**: ✅ COMPLETE (Naive Implementation)  
**Date**: 2025-12-24  
**Code**: 594 lines (334 module + 260 tests)

## Overview

This section implements a naive latency prediction module that provides a simple, flexible interface for integration into the TOMAS-LLM system. The module predicts tool execution latency (T_inf) in milliseconds using three modes: fixed, rule-based, and learned.

## Design Philosophy

**Naive Approach**: The implementation prioritizes simplicity and interface stability over prediction accuracy. This allows the system to be integrated and tested while more sophisticated latency models are developed in parallel.

**Key Principles**:
- Interface-first design: Clean API for system integration
- Multiple modes: From simplest (fixed) to more complex (learned)
- Easy to replace: Minimal dependencies, clear interfaces
- No backward compatibility: Fresh implementation

## Architecture

### 1. LatencyPredictor

Main prediction module with three operational modes:

```python
class LatencyPredictor(nn.Module):
    modes = ["fixed", "rule_based", "learned"]
```

#### Mode 1: Fixed Latency
Returns constant value for all tools.

```python
predictor = LatencyPredictor(mode='fixed', fixed_latency_ms=500.0)
latencies = predictor()  # tensor([500.])
```

**Use Case**: Initial prototyping, baseline comparisons

#### Mode 2: Rule-based Prediction
Uses simple heuristics:
1. Base latency from tool type lookup table
2. Input size scaling (small: 0.5x, medium: 1.0x, large: 2.0x)
3. GPU acceleration (30% reduction if GPU available)

```python
predictor = LatencyPredictor(mode='rule_based')

# Tool-only prediction
latencies = predictor(tool_names=['image_classification'])
# tensor([150.])

# With resource awareness
resources = torch.tensor([[1.0, 4.0, 8.0, 50.0, 8.0, 0.0]])  # Large input + GPU
latencies = predictor(tool_names=['image_classification'], resource_vectors=resources)
# tensor([210.]) = 150 * 2.0 (large) * 0.7 (GPU)
```

**Use Case**: Development and testing with realistic variations

**Latency Table** (base values in ms):
- image_classification: 150
- text_summarization: 300
- video_transcoding: 2000
- sentiment_analysis: 100
- object_detection: 200
- machine_translation: 400
- speech_recognition: 800
- data_preprocessing: 250
- default (unknown): 500

#### Mode 3: Learned Prediction
Uses small MLP to predict latency from resource features.

```python
predictor = LatencyPredictor(mode='learned', enable_learning=True, hidden_dim=128)

# Input: resource_vectors (batch, 6)
# [input_size, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb, latency_ms]
latencies = predictor(resource_vectors=resources)
```

**Architecture**:
```
Input (6D) → Linear(128D) → ReLU → Linear(64D) → ReLU → Linear(1D) → Softplus
```

**Use Case**: Future development, learned from profiling data

### 2. LatencyAwareModule

Wrapper for integration into planning pipeline.

```python
latency_module = LatencyAwareModule(
    latency_predictor=predictor,
    use_latency_in_planning=True
)

output = latency_module(tool_names, resource_vectors)
# {
#   'latencies': tensor([150., 300., ...]),
#   'latency_penalty': tensor([1.79, 2.09, ...])  # For optimization
# }
```

**Latency Penalty**: `log(1 + latencies/100)` for use in loss functions

## Implementation Details

### File Structure

```
src/context/latency_predictor.py    # Main module (334 lines)
tests/test_latency_predictor.py     # Test suite (260 lines)
configs/default.yaml                # Configuration
```

### Configuration

```yaml
model:
  latency_predictor:
    enabled: false           # Use fixed mode for prototyping
    mode: "fixed"           # "fixed", "rule_based", "learned"
    fixed_t_inf_ms: 500     # Fixed value (ms)
    enable_learning: false  # Enable MLP predictor
    hidden_dim: 128         # MLP hidden dimension
    use_in_planning: true   # Include latency in planning
```

### Resource Vector Format

Input to predictor (6D vector):
1. **input_size**: Normalized input size (-1: small, 0: medium, 1: large)
2. **cpu_core**: CPU cores allocated
3. **cpu_mem_gb**: CPU memory (GB)
4. **gpu_sm**: GPU SM utilization (0-100)
5. **gpu_mem_gb**: GPU memory (GB)
6. **latency_ms**: Historical latency (for training only)

## Usage Examples

### Basic Usage

```python
from context.latency_predictor import LatencyPredictor

# Create predictor
predictor = LatencyPredictor.from_config(config)

# Predict for tools
tool_names = ['image_classification', 'text_summarization']
latencies = predictor(tool_names=tool_names)
print(latencies)  # tensor([500., 500.]) in fixed mode
```

### Rule-based with Resources

```python
predictor = LatencyPredictor(mode='rule_based')

resources = torch.tensor([
    [-1.0, 2.0, 4.0, 0.0, 0.0, 0.0],   # Small, no GPU
    [0.0, 4.0, 8.0, 50.0, 8.0, 0.0],   # Medium, with GPU
    [1.0, 8.0, 16.0, 0.0, 0.0, 0.0]    # Large, no GPU
])

tools = ['image_classification'] * 3
latencies = predictor(tool_names=tools, resource_vectors=resources)
# tensor([75., 105., 300.])
# 150*0.5, 150*1.0*0.7, 150*2.0
```

### Integration into Planning

```python
from context.latency_predictor import LatencyAwareModule

# Create module
latency_module = LatencyAwareModule.from_config(config)

# In planning loop
output = latency_module(
    tool_names=candidate_tools,
    resource_vectors=resource_allocations
)

# Use latencies in decision making
predicted_latencies = output['latencies']
penalty = output['latency_penalty']

# Incorporate into loss
total_loss = task_loss + 0.1 * penalty.mean()
```

### Dynamic Mode Switching

```python
predictor = LatencyPredictor(
    mode='fixed',
    fixed_latency_ms=100.0,
    enable_learning=True  # Enable all modes
)

# Development: fixed mode
predictor.set_mode('fixed')
lat1 = predictor(tool_names=['image_classification'])

# Testing: rule-based
predictor.set_mode('rule_based')
lat2 = predictor(tool_names=['image_classification'])

# Production: learned
predictor.set_mode('learned')
lat3 = predictor(resource_vectors=resources)
```

### Updating Latency Table

```python
predictor = LatencyPredictor(mode='rule_based')

# Collect actual latency from profiling
actual_latency = profile_tool('image_classification')

# Update table
predictor.update_latency_table('image_classification', actual_latency)

# Future predictions use updated value
new_latency = predictor(tool_names=['image_classification'])
```

## Test Results

All 10 tests passing:

```
Test 1: Fixed Mode                      ✓
Test 2: Rule-based Mode                 ✓
Test 3: Rule-based with Resources       ✓
Test 4: Learned Mode                    ✓
Test 5: Gradient Flow                   ✓
Test 6: Mode Switching                  ✓
Test 7: from_config() Factory           ✓
Test 8: LatencyAwareModule              ✓
Test 9: Update Latency Table            ✓
Test 10: Unknown Tool Handling          ✓
```

### Key Test Validations

1. **Fixed Mode**: Returns constant values correctly
2. **Rule-based Scaling**:
   - Small input: 150ms → 75ms (0.5x)
   - Medium + GPU: 150ms → 105ms (1.0x * 0.7x)
   - Large input: 150ms → 300ms (2.0x)
3. **GPU Acceleration**: 30% reduction when GPU available
4. **Learned Mode**: MLP produces positive outputs
5. **Gradient Flow**: Backpropagation works correctly
6. **Unknown Tools**: Use default latency (500ms)

## Integration Points

### 1. Tool Selection Module

```python
# In tool planner
latency_module = LatencyAwareModule.from_config(config)

for candidate_tools in search_space:
    # Predict latencies
    output = latency_module(
        tool_names=candidate_tools,
        resource_vectors=resource_allocations
    )
    
    # Factor into selection
    selection_score = accuracy_score - 0.1 * output['latency_penalty']
```

### 2. Resource Allocation

```python
# Evaluate different resource configurations
resources_cpu = torch.tensor([[0.0, 4.0, 8.0, 0.0, 0.0, 0.0]])
resources_gpu = torch.tensor([[0.0, 4.0, 8.0, 50.0, 8.0, 0.0]])

lat_cpu = predictor(tool_names=['image_classification'], resource_vectors=resources_cpu)
lat_gpu = predictor(tool_names=['image_classification'], resource_vectors=resources_gpu)

# Choose based on latency requirements
if max_latency < 150:
    allocate_gpu()
else:
    allocate_cpu()
```

### 3. SLA Enforcement

```python
# Check if tool can meet SLA
required_latency = task.sla_ms
predicted_latency = predictor(tool_names=[tool_name])

if predicted_latency > required_latency:
    # Scale up resources or select faster tool
    alternative_tool = find_faster_tool()
```

## Performance Characteristics

### Model Size
- **Fixed/Rule-based**: 0 parameters (stateless)
- **Learned (hidden=128)**: ~17K parameters
  - Layer 1: 6 × 128 = 768
  - Layer 2: 128 × 64 = 8,192
  - Layer 3: 64 × 1 = 64
  - Biases: 128 + 64 + 1 = 193
  - Total: 9,217 trainable parameters

### Prediction Speed
- **Fixed**: O(1) - instant
- **Rule-based**: O(1) - lookup + scaling
- **Learned**: O(n) - forward pass through MLP (very fast)

### Memory Footprint
- Minimal: <1MB for all modes
- Latency table: ~10 entries × 8 bytes = 80 bytes
- MLP: ~17K params × 4 bytes = 68KB

## Limitations & Future Work

### Current Limitations

1. **Naive Rules**: Rule-based scaling factors (0.5x, 1.0x, 2.0x) are arbitrary
2. **No Historical Data**: Doesn't learn from actual execution traces
3. **Static Table**: Latency table requires manual updates
4. **Simple Features**: Doesn't consider tool complexity, batch size, etc.

### Future Enhancements

1. **Profiling Integration**
   ```python
   # Collect actual latencies
   profiler.record(tool_name, actual_latency, resources)
   
   # Update predictor
   predictor.fit(profiler.get_data())
   ```

2. **Tool Complexity Features**
   - Model size
   - Computation graph depth
   - Memory bandwidth requirements

3. **Batch-aware Prediction**
   ```python
   latencies = predictor(
       tool_names=tools,
       resource_vectors=resources,
       batch_sizes=batch_sizes  # New feature
   )
   ```

4. **Confidence Intervals**
   ```python
   output = predictor(tools, resources, return_uncertainty=True)
   # {
   #   'latencies': mean predictions,
   #   'std': uncertainty estimates
   # }
   ```

5. **Online Learning**
   ```python
   # Update model with new observations
   predictor.update(tool_name, actual_latency, resources)
   ```

## Design Decisions

### Why Three Modes?

1. **Fixed**: Simplest possible, good for initial integration
2. **Rule-based**: Reasonable approximation without training data
3. **Learned**: Future-proof, can improve with data

### Why Separate LatencyAwareModule?

- Clean separation of concerns
- Easier to integrate into planning pipeline
- Provides additional utilities (penalty computation)
- Can add more features without changing core predictor

### Why Softplus for Learned Mode?

- Ensures positive outputs (latency must be > 0)
- Smooth gradient (better than ReLU for regression)
- Numerically stable

## Summary

Section 2.2 provides a **simple, flexible latency prediction interface** ready for system integration. The naive implementation supports three modes (fixed, rule-based, learned) with a clean API that can be easily replaced as the system evolves.

**Key Achievements**:
- ✅ 3 prediction modes (fixed, rule-based, learned)
- ✅ Resource-aware scaling
- ✅ GPU acceleration modeling
- ✅ Clean integration interface
- ✅ 10/10 tests passing
- ✅ Fully documented
- ✅ Zero backward compatibility burden

**Integration Status**: Ready for use in tool planning pipeline

**Next Steps**: Section 2.3 - Runtime Context Integration (combining task, tools, latency, temporal features)
