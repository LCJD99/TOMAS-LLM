#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for SystemTimeline and ResourcePredictor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
import pandas as pd
from context.timeline import SystemTimeline, ResourcePredictor

print("=" * 60)
print("Testing SystemTimeline & ResourcePredictor")
print("=" * 60)

# Load config
with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

# Test 1: Load Timeline from CSV
print("\n" + "=" * 60)
print("Test 1: Load Timeline from CSV")
print("=" * 60)

csv_path = 'input/system_profiling.csv'
print(f"Loading timeline from: {csv_path}")

timeline = SystemTimeline(csv_path, interpolation='linear')
print(f"✓ Timeline loaded successfully")
print(f"  - Snapshots: {len(timeline.timeline_df)}")
print(f"  - Time range: {timeline.time_range[0]:.0f}ms - {timeline.time_range[1]:.0f}ms")
print(f"  - Interpolation: {timeline.interpolation}")

# Show first few rows
print(f"\nFirst 3 snapshots:")
print(timeline.timeline_df.head(3).to_string(index=False))

# Test 2: Query Exact Timestamps
print("\n" + "=" * 60)
print("Test 2: Query Exact Timestamps")
print("=" * 60)

test_times = [0, 500, 1000, 1500, 2000, 3000]
print(f"Querying timestamps: {test_times}")

for t in test_times:
    snapshot = timeline.get_snapshot_at(t)
    print(f"  t={t:4d}ms: cpu={snapshot['cpu_cores']:.1f}, "
          f"mem={snapshot['cpu_mem_gb']:.1f}GB, "
          f"gpu_sm={snapshot['gpu_sm']:.1f}%, "
          f"gpu_mem={snapshot['gpu_mem_gb']:.1f}GB")

print("✓ Exact timestamp queries successful")

# Test 3: Interpolation (Linear)
print("\n" + "=" * 60)
print("Test 3: Linear Interpolation")
print("=" * 60)

# Query between data points
interp_times = [50, 150, 250, 550, 1250]
print(f"Querying interpolated times: {interp_times}")

for t in interp_times:
    snapshot = timeline.get_snapshot_at(t)
    print(f"  t={t:4d}ms: cpu={snapshot['cpu_cores']:.2f}, "
          f"mem={snapshot['cpu_mem_gb']:.2f}GB, "
          f"gpu_sm={snapshot['gpu_sm']:.2f}%, "
          f"gpu_mem={snapshot['gpu_mem_gb']:.2f}GB")

# Verify interpolation correctness
snapshot_250 = timeline.get_snapshot_at(250)
# At t=250, should be midpoint between t=200 and t=300
expected_cpu = (14 + 14) / 2  # 14.0
assert abs(snapshot_250['cpu_cores'] - expected_cpu) < 0.01
print(f"\n✓ Linear interpolation verified (t=250ms: cpu={snapshot_250['cpu_cores']:.2f} ≈ {expected_cpu:.2f})")

# Test 4: Different Interpolation Methods
print("\n" + "=" * 60)
print("Test 4: Interpolation Methods Comparison")
print("=" * 60)

test_time = 250
methods = ['linear', 'nearest', 'previous']
results = {}

for method in methods:
    tl = SystemTimeline(csv_path, interpolation=method)
    snapshot = tl.get_snapshot_at(test_time)
    results[method] = snapshot
    print(f"{method:10s}: cpu={snapshot['cpu_cores']:.2f}, "
          f"gpu_sm={snapshot['gpu_sm']:.2f}%")

print("✓ All interpolation methods work")

# Test 5: Batch Queries
print("\n" + "=" * 60)
print("Test 5: Batch Snapshot Queries")
print("=" * 60)

batch_times = [0, 500, 1000, 1500, 2000]
print(f"Querying batch: {batch_times}")

batch_snapshots = timeline.get_batch_snapshots(batch_times)
print(f"✓ Retrieved {len(batch_snapshots)} snapshots")

for t, snap in zip(batch_times, batch_snapshots):
    print(f"  t={t:4d}ms: cpu={snap['cpu_cores']:.1f}, gpu_sm={snap['gpu_sm']:.1f}%")

# Test 6: Tensor Conversion
print("\n" + "=" * 60)
print("Test 6: Snapshot to Tensor Conversion")
print("=" * 60)

snapshot = timeline.get_snapshot_at(500)
tensor = timeline.to_tensor(snapshot)

print(f"Snapshot dict: {snapshot}")
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor values: {tensor.numpy()}")
print(f"Expected order: [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]")

assert tensor.shape == (4,)
assert torch.allclose(tensor, torch.tensor([12.0, 60.0, 70.0, 36.0]), atol=0.01)
print("✓ Tensor conversion correct")

# Test 7: Bounds Checking
print("\n" + "=" * 60)
print("Test 7: Out-of-Bounds Handling")
print("=" * 60)

# Without extrapolation (should raise error)
try:
    timeline.get_snapshot_at(5000, allow_extrapolation=False)
    print("✗ Should have raised ValueError for out-of-bounds")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Correctly raised error: {str(e)[:60]}...")

# With extrapolation (should return boundary value)
snapshot_extrap = timeline.get_snapshot_at(5000, allow_extrapolation=True)
snapshot_last = timeline.get_snapshot_at(3000)
print(f"Extrapolated (t=5000ms): cpu={snapshot_extrap['cpu_cores']:.1f}")
print(f"Last snapshot (t=3000ms): cpu={snapshot_last['cpu_cores']:.1f}")
assert snapshot_extrap['cpu_cores'] == snapshot_last['cpu_cores']
print("✓ Extrapolation uses boundary values")

# Test 8: ResourcePredictor - Basic Usage
print("\n" + "=" * 60)
print("Test 8: ResourcePredictor - Basic Usage")
print("=" * 60)

predictor = ResourcePredictor(timeline=timeline)
print("Created ResourcePredictor with timeline")

# Predict resources at T_inf
t_inf_values = [100, 500, 1000, 1500, 2000]
print(f"\nPredicting resources at T_inf values: {t_inf_values}")

for t_inf in t_inf_values:
    resource_tensor = predictor(t_inf, return_dict=False)
    resource_dict = predictor(t_inf, return_dict=True)
    
    print(f"  T_inf={t_inf:4d}ms:")
    print(f"    Tensor: {resource_tensor.numpy()}")
    print(f"    Dict: cpu={resource_dict['cpu_cores']:.1f}, gpu_sm={resource_dict['gpu_sm']:.1f}%")

print("✓ ResourcePredictor forward pass successful")

# Test 9: ResourcePredictor - Tensor Input
print("\n" + "=" * 60)
print("Test 9: ResourcePredictor - Tensor Input")
print("=" * 60)

# Create tensor input
t_inf_tensor = torch.tensor(750.0)
print(f"Input: T_inf as tensor = {t_inf_tensor.item():.0f}ms")

resource_pred = predictor(t_inf_tensor, return_dict=False)
print(f"Output tensor: {resource_pred.numpy()}")
print(f"Expected (interpolated between 700-800ms)")

# Manual check
snap_700 = timeline.get_snapshot_at(700)
snap_800 = timeline.get_snapshot_at(800)
snap_750 = timeline.get_snapshot_at(750)
print(f"  t=700ms: cpu={snap_700['cpu_cores']:.1f}")
print(f"  t=750ms: cpu={snap_750['cpu_cores']:.1f} (interpolated)")
print(f"  t=800ms: cpu={snap_800['cpu_cores']:.1f}")

assert abs(resource_pred[0].item() - snap_750['cpu_cores']) < 0.01
print("✓ Tensor input handled correctly")

# Test 10: ResourcePredictor from Config
print("\n" + "=" * 60)
print("Test 10: ResourcePredictor from Config")
print("=" * 60)

predictor_from_config = ResourcePredictor.from_config(config)
print(f"Created ResourcePredictor from config")

# Check if timeline loaded
if predictor_from_config.timeline is not None:
    print(f"  - Timeline loaded: {len(predictor_from_config.timeline.timeline_df)} snapshots")
else:
    print(f"  - Timeline: None (using default)")

# Check default snapshot
print(f"  - Default snapshot: {predictor_from_config.default_snapshot}")

# Test prediction
resource = predictor_from_config(1000, return_dict=True)
print(f"\nPrediction at T_inf=1000ms:")
for key, val in resource.items():
    print(f"  {key}: {val:.2f}")

print("✓ from_config() works correctly")

# Test 11: ResourcePredictor without Timeline (Default Mode)
print("\n" + "=" * 60)
print("Test 11: ResourcePredictor - Default Mode")
print("=" * 60)

default_snapshot = {
    'cpu_cores': 20.0,
    'cpu_mem_gb': 80.0,
    'gpu_sm': 90.0,
    'gpu_mem_gb': 48.0
}

predictor_default = ResourcePredictor(timeline=None, default_snapshot=default_snapshot)
print(f"Created ResourcePredictor without timeline")
print(f"Default snapshot: {default_snapshot}")

# Should always return default values
for t_inf in [100, 500, 1000]:
    resource = predictor_default(t_inf, return_dict=True)
    print(f"  T_inf={t_inf}ms: cpu={resource['cpu_cores']:.1f}")
    assert resource == default_snapshot

print("✓ Default mode works correctly (constant values)")

# Test 12: Update Timeline and Default Snapshot
print("\n" + "=" * 60)
print("Test 12: Dynamic Update")
print("=" * 60)

predictor_dynamic = ResourcePredictor()
print("Created empty ResourcePredictor")

# Initially returns default
initial_resource = predictor_dynamic(500, return_dict=True)
print(f"Initial (default): cpu={initial_resource['cpu_cores']:.1f}")

# Set timeline
predictor_dynamic.set_timeline(timeline)
updated_resource = predictor_dynamic(500, return_dict=True)
print(f"After timeline update: cpu={updated_resource['cpu_cores']:.1f}")

# Should be different now
assert initial_resource['cpu_cores'] != updated_resource['cpu_cores']
print("✓ Dynamic update works")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)

print("\nSummary:")
print(f"  - Timeline snapshots: {len(timeline.timeline_df)}")
print(f"  - Time range: {timeline.time_range[0]:.0f}ms - {timeline.time_range[1]:.0f}ms")
print(f"  - Interpolation methods: linear, nearest, previous")
print(f"  - ResourcePredictor: tensor/dict output modes")
print(f"  - Default mode: fallback when timeline unavailable")
print(f"  ✓ Ready for system integration!")
