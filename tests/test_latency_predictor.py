#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for LatencyPredictor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from context.latency_predictor import LatencyPredictor, LatencyAwareModule

print("=" * 60)
print("Testing LatencyPredictor")
print("=" * 60)

# Load config
with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

# Test 1: Fixed Mode
print("\n" + "=" * 60)
print("Test 1: Fixed Mode")
print("=" * 60)

predictor_fixed = LatencyPredictor(mode='fixed', fixed_latency_ms=500.0)
print(f"Created: {predictor_fixed}")

# Predict for batch
batch_size = 5
latencies = predictor_fixed()  # Single prediction
latencies_batch = predictor_fixed.predict_fixed(batch_size)

print(f"Single prediction: {latencies}")
print(f"Batch prediction ({batch_size}): {latencies_batch}")
assert latencies_batch.shape == (batch_size,)
assert (latencies_batch == 500.0).all()
print("✓ Fixed mode works correctly")

# Test 2: Rule-based Mode
print("\n" + "=" * 60)
print("Test 2: Rule-based Mode")
print("=" * 60)

predictor_rules = LatencyPredictor(mode='rule_based')
print(f"Created: {predictor_rules}")

# Test with tool names only
tool_names = ['image_classification', 'video_transcoding', 'text_summarization']
latencies_names = predictor_rules(tool_names=tool_names)

print(f"\nTool names: {tool_names}")
print(f"Base latencies: {latencies_names.tolist()}")
print(f"Expected:")
print(f"  - image_classification: 150ms")
print(f"  - video_transcoding: 2000ms")
print(f"  - text_summarization: 300ms")

assert latencies_names[0] == 150.0
assert latencies_names[1] == 2000.0
assert latencies_names[2] == 300.0
print("✓ Tool-based prediction correct")

# Test 3: Rule-based with Resources
print("\n" + "=" * 60)
print("Test 3: Rule-based with Resource Scaling")
print("=" * 60)

# Create resource vectors with different input sizes
# [input_size, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb, latency_ms]
resources = torch.tensor([
    [-1.0, 2.0, 4.0, 0.0, 0.0, 0.0],  # Small input, no GPU
    [ 0.0, 4.0, 8.0, 50.0, 8.0, 0.0], # Medium input, with GPU
    [ 1.0, 8.0, 16.0, 0.0, 0.0, 0.0], # Large input, no GPU
])

tool_names_res = ['image_classification'] * 3
latencies_res = predictor_rules(tool_names=tool_names_res, resource_vectors=resources)

print(f"\nTool: image_classification (base=150ms)")
print(f"Resource configs:")
print(f"  1. Small input, no GPU:  {latencies_res[0]:.1f}ms")
print(f"  2. Medium input, GPU:    {latencies_res[1]:.1f}ms")
print(f"  3. Large input, no GPU:  {latencies_res[2]:.1f}ms")

# Expected scaling:
# 1. Small (0.5x) no GPU (1.0x) = 150 * 0.5 = 75ms
# 2. Medium (1.0x) with GPU (0.7x) = 150 * 1.0 * 0.7 = 105ms
# 3. Large (2.0x) no GPU (1.0x) = 150 * 2.0 = 300ms

assert abs(latencies_res[0] - 75.0) < 1.0
assert abs(latencies_res[1] - 105.0) < 1.0
assert abs(latencies_res[2] - 300.0) < 1.0
print("✓ Resource scaling works correctly")

# Test 4: Learned Mode
print("\n" + "=" * 60)
print("Test 4: Learned Mode")
print("=" * 60)

predictor_learned = LatencyPredictor(mode='learned', enable_learning=True, hidden_dim=64)
print(f"Created: {predictor_learned}")
print(f"Predictor architecture:")
print(f"  {predictor_learned.predictor}")

# Predict with learned model
resources_learned = torch.randn(5, 6)
latencies_learned = predictor_learned(resource_vectors=resources_learned)

print(f"\nPredicted latencies (random weights): {latencies_learned}")
print(f"Shape: {latencies_learned.shape}")
print(f"All positive: {(latencies_learned > 0).all()}")

assert latencies_learned.shape == (5,)
assert (latencies_learned > 0).all()  # Softplus ensures positive
print("✓ Learned mode works correctly")

# Test 5: Gradient Flow
print("\n" + "=" * 60)
print("Test 5: Gradient Flow (Learned Mode)")
print("=" * 60)

resources_grad = torch.randn(3, 6, requires_grad=True)
latencies_grad = predictor_learned(resource_vectors=resources_grad)
loss = latencies_grad.sum()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Resource gradients exist: {resources_grad.grad is not None}")
print(f"Predictor has gradients: {any(p.grad is not None for p in predictor_learned.predictor.parameters())}")

assert resources_grad.grad is not None
print("✓ Gradients flow correctly")

# Test 6: Mode Switching
print("\n" + "=" * 60)
print("Test 6: Mode Switching")
print("=" * 60)

predictor_switch = LatencyPredictor(mode='fixed', fixed_latency_ms=100.0, enable_learning=True)
print(f"Initial mode: {predictor_switch.mode}")

# Fixed mode
lat1 = predictor_switch(tool_names=['image_classification'])
print(f"Fixed mode: {lat1.item()}ms")

# Switch to rule-based
predictor_switch.set_mode('rule_based')
lat2 = predictor_switch(tool_names=['image_classification'])
print(f"Rule-based mode: {lat2.item()}ms")

# Switch to learned
predictor_switch.set_mode('learned')
lat3 = predictor_switch(resource_vectors=torch.randn(1, 6))
print(f"Learned mode: {lat3.item():.1f}ms")

assert lat1 == 100.0
assert lat2 == 150.0
print("✓ Mode switching works correctly")

# Test 7: from_config
print("\n" + "=" * 60)
print("Test 7: from_config() Factory")
print("=" * 60)

predictor_config = LatencyPredictor.from_config(config)
print(f"Created from config: {predictor_config}")
print(f"Mode: {predictor_config.mode}")
print(f"Fixed latency: {predictor_config.fixed_latency_ms}ms")

lat_config = predictor_config(tool_names=['video_transcoding'])
print(f"Prediction: {lat_config.item()}ms (should be 500ms in fixed mode)")

assert predictor_config.mode == 'fixed'
assert lat_config == 500.0
print("✓ from_config() works correctly")

# Test 8: LatencyAwareModule
print("\n" + "=" * 60)
print("Test 8: LatencyAwareModule")
print("=" * 60)

latency_module = LatencyAwareModule.from_config(config)
print(f"Created LatencyAwareModule")
print(f"Use in planning: {latency_module.use_latency_in_planning}")

# Forward pass
output = latency_module(
    tool_names=['image_classification', 'text_summarization'],
    resource_vectors=torch.randn(2, 6)
)

print(f"\nOutput keys: {list(output.keys())}")
print(f"Latencies: {output['latencies']}")
if 'latency_penalty' in output:
    print(f"Latency penalty: {output['latency_penalty']}")

assert 'latencies' in output
print("✓ LatencyAwareModule works correctly")

# Test 9: Update Latency Table
print("\n" + "=" * 60)
print("Test 9: Update Latency Table")
print("=" * 60)

predictor_update = LatencyPredictor(mode='rule_based')

# Original latency
lat_before = predictor_update(tool_names=['image_classification']).item()
print(f"Original latency for image_classification: {lat_before}ms")

# Update
predictor_update.update_latency_table('image_classification', 250.0)

# New latency
lat_after = predictor_update(tool_names=['image_classification']).item()
print(f"Updated latency for image_classification: {lat_after}ms")

assert lat_before == 150.0
assert lat_after == 250.0
print("✓ Latency table update works correctly")

# Test 10: Unknown Tool Handling
print("\n" + "=" * 60)
print("Test 10: Unknown Tool Handling")
print("=" * 60)

predictor_unknown = LatencyPredictor(mode='rule_based')
unknown_tools = ['unknown_tool_1', 'unknown_tool_2']

lat_unknown = predictor_unknown(tool_names=unknown_tools)
print(f"Unknown tools: {unknown_tools}")
print(f"Predicted latencies: {lat_unknown.tolist()}")
print(f"Default latency: {predictor_unknown.latency_table['default']}ms")

assert (lat_unknown == 500.0).all()
print("✓ Unknown tools use default latency")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)

print("\nSummary:")
print(f"  - Modes tested: fixed, rule_based, learned")
print(f"  - Resource scaling: ✓")
print(f"  - GPU acceleration: ✓ (30% reduction)")
print(f"  - Mode switching: ✓")
print(f"  - Gradient flow: ✓")
print(f"  - LatencyAwareModule: ✓")
print(f"  ✓ Ready for integration into planning pipeline!")
