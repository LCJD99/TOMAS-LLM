#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for ToolAwareEmbedding (Concatenation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from encoders.concatenation import ToolAwareEmbedding, ResourceAwareToolEncoder
from encoders.tool_encoder import ToolEncoder
from encoders.resource_mlp import ResourceMLP

print("=" * 60)
print("Testing ToolAwareEmbedding (Concatenation)")
print("=" * 60)

# Load config
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

d_tool = config['model']['tool_encoder']['d_tool']
d_resource = config['model']['resource_mlp']['d_resource']

print(f"Configuration:")
print(f"  d_tool: {d_tool}")
print(f"  d_resource: {d_resource}")
print(f"  d_toolaware (expected): {d_tool + d_resource}")

# Create concatenator
concat = ToolAwareEmbedding(d_tool=d_tool, d_resource=d_resource)
print(f"\nCreated ToolAwareEmbedding:")
print(f"  {concat}")
print(f"  Output dimension: {concat.get_output_dim()}")

# Test single sample
print("\n" + "=" * 60)
print("Test 1: Single Sample")
print("=" * 60)

tool_emb = torch.randn(d_tool)
resource_emb = torch.randn(d_resource)

print(f"Tool embedding shape: {tool_emb.shape}")
print(f"Resource embedding shape: {resource_emb.shape}")

toolaware_emb = concat(tool_emb, resource_emb)
print(f"Tool-aware embedding shape: {toolaware_emb.shape}")
print(f"Expected shape: ({d_tool + d_resource},)")

assert toolaware_emb.shape == (d_tool + d_resource,), "Dimension mismatch!"
print("✓ Dimension check passed")

# Test split
tool_split, resource_split = concat.split(toolaware_emb)
print(f"\nSplit test:")
print(f"  Tool part shape: {tool_split.shape}")
print(f"  Resource part shape: {resource_split.shape}")

diff_tool = (tool_emb - tool_split).abs().max().item()
diff_resource = (resource_emb - resource_split).abs().max().item()
print(f"  Max diff (tool): {diff_tool:.10f}")
print(f"  Max diff (resource): {diff_resource:.10f}")

assert diff_tool < 1e-6 and diff_resource < 1e-6, "Split reconstruction failed!"
print("✓ Split reconstruction correct")

# Test batch
print("\n" + "=" * 60)
print("Test 2: Batch Processing")
print("=" * 60)

batch_size = 8
tool_batch = torch.randn(batch_size, d_tool)
resource_batch = torch.randn(batch_size, d_resource)

print(f"Tool batch shape: {tool_batch.shape}")
print(f"Resource batch shape: {resource_batch.shape}")

toolaware_batch = concat(tool_batch, resource_batch)
print(f"Tool-aware batch shape: {toolaware_batch.shape}")
print(f"Expected shape: ({batch_size}, {d_tool + d_resource})")

assert toolaware_batch.shape == (batch_size, d_tool + d_resource), "Batch dimension mismatch!"
print("✓ Batch dimension check passed")

# Test from_config
print("\n" + "=" * 60)
print("Test 3: from_config()")
print("=" * 60)

concat_from_config = ToolAwareEmbedding.from_config(config)
print(f"Created from config: {concat_from_config}")

toolaware_from_config = concat_from_config(tool_batch, resource_batch)
print(f"Output shape: {toolaware_from_config.shape}")
assert toolaware_from_config.shape == (batch_size, d_tool + d_resource)
print("✓ from_config() works correctly")

# Test dimension validation
print("\n" + "=" * 60)
print("Test 4: Dimension Validation")
print("=" * 60)

wrong_tool = torch.randn(d_tool + 10)  # Wrong dimension
try:
    concat(wrong_tool, resource_emb)
    print("✗ Should have raised ValueError for wrong dimension")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Correctly raised error: {e}")

# Test mismatched batching
wrong_batch = torch.randn(batch_size + 1, d_resource)  # Different batch size
try:
    concat(tool_batch, wrong_batch)
    print("✗ Should have raised ValueError for mismatched batch size")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Correctly raised error: {e}")

# Test gradient flow
print("\n" + "=" * 60)
print("Test 5: Gradient Flow")
print("=" * 60)

tool_with_grad = torch.randn(batch_size, d_tool, requires_grad=True)
resource_with_grad = torch.randn(batch_size, d_resource, requires_grad=True)

toolaware_with_grad = concat(tool_with_grad, resource_with_grad)
loss = toolaware_with_grad.sum()
loss.backward()

print(f"Tool gradient shape: {tool_with_grad.grad.shape}")
print(f"Resource gradient shape: {resource_with_grad.grad.shape}")
print(f"Tool gradient is not None: {tool_with_grad.grad is not None}")
print(f"Resource gradient is not None: {resource_with_grad.grad is not None}")
print("✓ Gradients flow correctly")

# Test norms preservation
print("\n" + "=" * 60)
print("Test 6: Embedding Norms")
print("=" * 60)

tool_norms = tool_batch.norm(dim=1)
resource_norms = resource_batch.norm(dim=1)
toolaware_norms = toolaware_batch.norm(dim=1)

print(f"Tool norms (mean): {tool_norms.mean().item():.4f}")
print(f"Resource norms (mean): {resource_norms.mean().item():.4f}")
print(f"Tool-aware norms (mean): {toolaware_norms.mean().item():.4f}")

# Tool-aware norm should be roughly sqrt(tool_norm^2 + resource_norm^2)
expected_norms = torch.sqrt(tool_norms**2 + resource_norms**2)
norm_diff = (toolaware_norms - expected_norms).abs().max().item()
print(f"Expected combined norm (mean): {expected_norms.mean().item():.4f}")
print(f"Max difference: {norm_diff:.6f}")
assert norm_diff < 1e-5, "Norm combination incorrect!"
print("✓ Norms combine correctly")

print("\n" + "=" * 60)
print("ALL BASIC TESTS PASSED ✓")
print("=" * 60)
