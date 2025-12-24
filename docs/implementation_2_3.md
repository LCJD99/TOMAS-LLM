# Implementation Documentation: Section 2.3 - System Timeline & Resource Snapshot

**实现日期**: 2024-01-XX  
**实现者**: TOMAS-LLM Team  
**状态**: ✅ Complete

---

## 1. Overview

Section 2.3 implements **System Timeline Snapshot Prediction**, which:
- Reads system resource profiling data from CSV
- Provides resource snapshots at specific timestamps
- Predicts future resource availability after T_inf latency
- Supports multiple interpolation methods (linear, nearest, previous)

This module integrates with Section 2.2 (Latency Prediction) to provide the runtime context needed for intelligent tool planning.

---

## 2. Architecture

### 2.1 Component Hierarchy

```
SystemTimeline (Data Management)
    ├── Load CSV: time_ms, cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb
    ├── Interpolation: linear / nearest / previous
    └── Query Interface: get_snapshot_at(time_ms)

ResourcePredictor (nn.Module)
    ├── Input: T_inf (predicted latency)
    ├── Timeline Query: SystemTimeline.get_snapshot_at(T_inf)
    ├── Fallback: default_snapshot when timeline unavailable
    └── Output: resource_tensor(4,) or resource_dict
```

### 2.2 Data Format

**Input CSV** (`input/system_profiling.csv`):
```csv
time_ms,cpu_cores,cpu_mem_gb,gpu_sm,gpu_mem_gb
0,16,64.0,80,40.0
100,16,64.0,80,40.0
200,14,62.0,75,38.0
...
```

Where:
- `time_ms`: Time from task submission (milliseconds)
- `cpu_cores`: Available CPU cores
- `cpu_mem_gb`: Available CPU memory (GB)
- `gpu_sm`: Available GPU SM percentage (0-100%)
- `gpu_mem_gb`: Available GPU memory (GB)

**Resource Snapshot Dictionary**:
```python
{
    'cpu_cores': float,
    'cpu_mem_gb': float,
    'gpu_sm': float,  # 0-100 percentage
    'gpu_mem_gb': float
}
```

**Resource Tensor**:
```python
torch.Tensor([cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb])  # Shape: (4,)
```

---

## 3. Implementation Details

### 3.1 SystemTimeline

**Class**: `SystemTimeline`  
**File**: `src/context/timeline.py`

**Key Methods**:

1. **`__init__(csv_path, interpolation)`**
   - Loads timeline from CSV
   - Validates columns
   - Sorts by time
   - Stores time range

2. **`get_snapshot_at(time_ms, allow_extrapolation)`**
   - Queries resource snapshot at specific time
   - Supports 3 interpolation methods:
     - `linear`: Linear interpolation between surrounding points
     - `nearest`: Nearest neighbor
     - `previous`: Step function (previous value)
   - Bounds checking with optional extrapolation

3. **`get_batch_snapshots(times_ms)`**
   - Batch query interface
   - Returns list of snapshots

4. **`to_tensor(snapshot, device)`**
   - Converts dict to tensor
   - Shape: (4,) with [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]

**Interpolation Example**:
```python
# Linear interpolation between t=700ms and t=800ms
# At t=750ms (midpoint):
#   cpu_cores: 10 + (8 - 10) * 0.5 = 9.0
#   gpu_sm: 65 + (60 - 65) * 0.5 = 62.5
```

### 3.2 ResourcePredictor

**Class**: `ResourcePredictor(nn.Module)`  
**File**: `src/context/timeline.py`

**Key Methods**:

1. **`forward(t_inf_ms, return_dict)`**
   - Predicts resources at T_inf timestamp
   - Input: scalar or torch.Tensor
   - Output: dict or tensor (4,)
   - Fallback to default snapshot if timeline unavailable

2. **`set_timeline(timeline)`**
   - Update timeline dynamically

3. **`set_default_snapshot(snapshot)`**
   - Update default resource values

4. **`from_config(config)`**
   - Factory method to create from YAML config
   - Loads timeline if csv_path provided
   - Extracts default snapshot from naive_mode

**Default Snapshot Logic**:
```python
# Priority:
# 1. Timeline query (if timeline loaded and time in range)
# 2. Timeline extrapolation (if allow_extrapolation=True)
# 3. Default snapshot (if timeline unavailable or query fails)
```

---

## 4. Usage Examples

### 4.1 Basic Timeline Usage

```python
from context.timeline import SystemTimeline

# Load timeline
timeline = SystemTimeline("input/system_profiling.csv", interpolation="linear")

# Query exact timestamp
snapshot = timeline.get_snapshot_at(500)  # At t=500ms
print(snapshot)
# {'cpu_cores': 12.0, 'cpu_mem_gb': 60.0, 'gpu_sm': 70.0, 'gpu_mem_gb': 36.0}

# Query with interpolation
snapshot = timeline.get_snapshot_at(750)  # Between t=700 and t=800
print(snapshot)
# {'cpu_cores': 9.0, 'cpu_mem_gb': 57.0, 'gpu_sm': 62.5, 'gpu_mem_gb': 33.0}

# Batch query
snapshots = timeline.get_batch_snapshots([0, 500, 1000, 1500])

# Convert to tensor
tensor = timeline.to_tensor(snapshot)
# torch.Tensor([9.0, 57.0, 62.5, 33.0])
```

### 4.2 ResourcePredictor Usage

```python
from context.timeline import ResourcePredictor, SystemTimeline
import torch

# Create timeline
timeline = SystemTimeline("input/system_profiling.csv")

# Create predictor
predictor = ResourcePredictor(timeline=timeline)

# Predict resources at T_inf
t_inf = torch.tensor(1000.0)  # 1 second latency

# Get as tensor
resource_tensor = predictor(t_inf, return_dict=False)
# torch.Tensor([10.0, 58.0, 65.0, 34.0])

# Get as dict
resource_dict = predictor(t_inf, return_dict=True)
# {'cpu_cores': 10.0, 'cpu_mem_gb': 58.0, 'gpu_sm': 65.0, 'gpu_mem_gb': 34.0}
```

### 4.3 Integration with Latency Predictor

```python
from context.latency import LatencyPredictor
from context.timeline import ResourcePredictor, SystemTimeline

# Setup
timeline = SystemTimeline("input/system_profiling.csv")
latency_predictor = LatencyPredictor(mode="fixed", fixed_latency_ms=1000)
resource_predictor = ResourcePredictor(timeline=timeline)

# Predict latency
t_inf = latency_predictor()  # Returns torch.Tensor([1000.0])

# Predict resources at T_inf
future_resources = resource_predictor(t_inf, return_dict=True)

print(f"Tool will execute in {t_inf.item():.0f}ms")
print(f"Available resources at completion:")
print(f"  CPU cores: {future_resources['cpu_cores']:.0f}")
print(f"  CPU mem: {future_resources['cpu_mem_gb']:.1f} GB")
print(f"  GPU SM: {future_resources['gpu_sm']:.0f}%")
print(f"  GPU mem: {future_resources['gpu_mem_gb']:.1f} GB")
```

### 4.4 Default Mode (No Timeline)

```python
# Create predictor without timeline
default_snapshot = {
    'cpu_cores': 16.0,
    'cpu_mem_gb': 64.0,
    'gpu_sm': 80.0,
    'gpu_mem_gb': 40.0
}

predictor = ResourcePredictor(timeline=None, default_snapshot=default_snapshot)

# All predictions return default values
resource = predictor(500, return_dict=True)
# Always returns: {'cpu_cores': 16.0, 'cpu_mem_gb': 64.0, ...}
```

### 4.5 From Config

```python
import yaml
from context.timeline import ResourcePredictor

# Load config
with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

# Create from config
predictor = ResourcePredictor.from_config(config)

# Automatically loads:
# - csv_path: "input/system_profiling.csv"
# - interpolation: "linear"
# - default_snapshot from naive_mode settings
```

---

## 5. Test Results

**Test File**: `tests/test_timeline.py`  
**Status**: ✅ All 12 tests passing

### Test Coverage

1. ✅ Load Timeline from CSV (31 snapshots, 0-3000ms range)
2. ✅ Query Exact Timestamps (6 queries)
3. ✅ Linear Interpolation (verified midpoint calculation)
4. ✅ Interpolation Methods Comparison (linear/nearest/previous)
5. ✅ Batch Snapshot Queries (5 timestamps)
6. ✅ Snapshot to Tensor Conversion (shape verification)
7. ✅ Out-of-Bounds Handling (error + extrapolation)
8. ✅ ResourcePredictor Basic Usage (5 T_inf values)
9. ✅ ResourcePredictor Tensor Input (torch.Tensor handling)
10. ✅ ResourcePredictor from Config (YAML loading)
11. ✅ ResourcePredictor Default Mode (constant fallback)
12. ✅ Dynamic Update (set_timeline/set_default_snapshot)

### Test Output Summary

```
============================================================
ALL TESTS PASSED ✓
============================================================

Summary:
  - Timeline snapshots: 31
  - Time range: 0ms - 3000ms
  - Interpolation methods: linear, nearest, previous
  - ResourcePredictor: tensor/dict output modes
  - Default mode: fallback when timeline unavailable
  ✓ Ready for system integration!
```

### Key Validations

- **Interpolation Accuracy**: Linear interpolation verified at t=250ms
  - Expected: 14.0 cores, Got: 14.00 cores ✓
  
- **Tensor Shape**: Resource tensor shape=(4,) validated
  
- **Bounds Handling**: Out-of-range queries properly raise ValueError
  
- **Extrapolation**: Beyond-range queries return boundary values
  
- **Config Loading**: Timeline and defaults loaded from YAML

---

## 6. Integration Points

### 6.1 Inputs

From Section 2.2 (Latency Prediction):
```python
t_inf = latency_predictor()  # torch.Tensor, shape (1,)
```

From Configuration:
```yaml
runtime:
  timeline:
    csv_path: "input/system_profiling.csv"
    interpolation: "linear"
  naive_mode:
    fixed_cpu_free_cores: [16, ...]
    fixed_gpu_free_sm_ratio: [0.8, ...]
```

### 6.2 Outputs

To Section 3.x (LLM Context):
```python
# Resource snapshot at T_inf
resource_snapshot = resource_predictor(t_inf, return_dict=True)
# Format: {'cpu_cores': float, 'cpu_mem_gb': float, 'gpu_sm': float, 'gpu_mem_gb': float}

# Or as tensor for neural network input
resource_tensor = resource_predictor(t_inf, return_dict=False)
# Format: torch.Tensor([cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]), shape (4,)
```

### 6.3 Dependencies

**Python Packages**:
- pandas: CSV loading and data manipulation
- numpy: Interpolation calculations
- torch: Tensor conversions

**Internal Modules**:
- None (standalone module)

**Data Files**:
- `input/system_profiling.csv`: System resource timeline

---

## 7. Configuration

### 7.1 YAML Settings

```yaml
# configs/simple-test.yaml
runtime:
  timeline:
    csv_path: "input/system_profiling.csv"  # Path to timeline CSV
    interpolation: "linear"  # Interpolation method
    prediction_window_sec: 60  # For future extensions
    time_granularity_ms: 100  # For future extensions
    
  naive_mode:
    enabled: true
    # Default snapshot values (used when timeline unavailable)
    fixed_cpu_free_cores: [16, 16, 14, 14, 12, ...]
    fixed_cpu_mem_gb: [64, 64, 62, 62, 60, ...]
    fixed_gpu_free_sm_ratio: [0.8, 0.8, 0.75, 0.75, ...]
    fixed_gpu_free_mem_gb: [40, 40, 38, 38, 36, ...]
```

### 7.2 Interpolation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `linear` | Linear interpolation between points | Smooth resource changes |
| `nearest` | Nearest neighbor | Discrete resource levels |
| `previous` | Step function (previous value) | Resource reservations |

---

## 8. Performance Characteristics

### 8.1 Time Complexity

- **CSV Loading**: O(n log n) - sorting by time
- **Query (linear)**: O(log n) - binary search + interpolation
- **Query (nearest/previous)**: O(log n) - binary search
- **Batch Query**: O(m log n) - m queries

Where:
- n = number of timeline snapshots
- m = number of batch queries

### 8.2 Space Complexity

- **Timeline Storage**: O(n) - DataFrame with 5 columns
- **Predictor**: O(1) - no additional storage

### 8.3 Typical Performance

With 31 snapshots (0-3000ms):
- Single query: < 1ms
- Batch of 100 queries: < 10ms
- Memory usage: < 10KB (DataFrame)

---

## 9. Design Decisions

### 9.1 Why Pandas for CSV?

- **Pros**:
  - Built-in CSV parsing with type inference
  - Easy column validation
  - Efficient sorting and indexing
  - Familiar API for data manipulation

- **Alternatives Considered**:
  - CSV module: More manual, less features
  - NumPy loadtxt: Less flexible column handling

### 9.2 Why Multiple Interpolation Methods?

Different resource types have different characteristics:
- **Linear**: Good for gradual resource changes (memory usage)
- **Previous**: Good for discrete resource blocks (CPU cores)
- **Nearest**: Good for approximate queries

### 9.3 Why Default Snapshot?

- Graceful degradation when timeline unavailable
- Enables testing without CSV file
- Fallback for production when profiling data missing

### 9.4 Why Tensor and Dict Outputs?

- **Dict**: Human-readable, debugging, external interfaces
- **Tensor**: Neural network input, batch processing

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

1. **Static Timeline**: CSV loaded once, not updated dynamically
2. **No Prediction**: Uses historical data, doesn't predict future trends
3. **Single CSV**: Can't merge multiple profiling runs
4. **No Uncertainty**: Returns point estimates, no confidence intervals

### 10.2 Future Enhancements

1. **Dynamic Timeline**:
   ```python
   timeline.update_snapshot(time_ms, resource_dict)
   timeline.save_to_csv(path)
   ```

2. **Predictive Modeling**:
   ```python
   # Train model on historical data
   timeline_model = TimelinePredictor(history=timeline)
   future_snapshot = timeline_model.predict(t_inf)
   ```

3. **Multi-Source Fusion**:
   ```python
   timeline.merge([timeline1, timeline2, timeline3])
   ```

4. **Uncertainty Quantification**:
   ```python
   snapshot, std = predictor.predict_with_uncertainty(t_inf)
   ```

---

## 11. Files Created

```
src/context/timeline.py          # 380 lines - SystemTimeline + ResourcePredictor
tests/test_timeline.py           # 380 lines - 12 comprehensive tests
input/system_profiling.csv       # 31 lines - Sample timeline data
docs/implementation_2_3.md       # This file - Documentation
```

---

## 12. Summary

Section 2.3 is now **complete** with:

✅ **SystemTimeline**: CSV loading, interpolation, query interface  
✅ **ResourcePredictor**: T_inf → resource snapshot, default fallback  
✅ **Integration**: Works with Section 2.2 (Latency Predictor)  
✅ **Testing**: 12/12 tests passing, all features validated  
✅ **Documentation**: Complete usage examples and API reference  
✅ **Configuration**: YAML-based setup with defaults  

**Ready for Section 3.x integration!**

---

## 13. Next Steps

1. **Section 3.x**: Integrate timeline snapshots into LLM context
2. **Section 4.x**: Use resource predictions for tool planning decisions
3. **End-to-End**: Combine encoders → context → LLM → outputs

**Timeline prediction provides the runtime context needed for intelligent, resource-aware tool scheduling!**
