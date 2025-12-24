#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Integration test for complete tool-aware embedding pipeline:
ToolEncoder → ResourceMLP → Concatenation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from data.loader import load_tool_data
from encoders.tool_encoder import ToolEncoder
from encoders.resource_mlp import ResourceMLP
from encoders.concatenation import ToolAwareEmbedding, ResourceAwareToolEncoder

print("=" * 60)
print("Integration Test: Complete Tool-Aware Embedding Pipeline")
print("=" * 60)

# Load configuration
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

print("\n1. Loading tool data...")
tool_dataset = load_tool_data(
    tool_registry_path=config['data']['tool_registry_path'],
    profiling_path=config['data']['profiling_path']
)

# Get registry info
tool_registry = tool_dataset.tool_registry.tools
num_tools = len(tool_registry)
num_samples = len(tool_dataset)
print(f"   - Loaded {num_tools} tools")
print(f"   - Loaded {num_samples} profiling samples")

# Extract tool names
tool_names = [tool.name for tool in tool_registry]
print(f"   - Tools: {', '.join(tool_names)}")

# Get sample data
sample_indices = [0, 8, 16]  # web_search small, image_gen small, code_exec small
print(f"\n2. Testing with samples at indices: {sample_indices}")

for idx in sample_indices:
    sample = tool_dataset[idx]
    print(f"   - {sample['tool_name']}: {sample['input_size']}")

# Create encoders
print("\n3. Initializing encoders...")

# Tool encoder (name-based)
tool_encoder = ToolEncoder(config, tool_names=tool_names, encoder_type="name")
print(f"   - ToolEncoder: d_tool={tool_encoder.d_tool}")

# Resource MLP
resource_mlp = ResourceMLP.from_config(config)
print(f"   - ResourceMLP: d_resource={resource_mlp.d_resource}, params={sum(p.numel() for p in resource_mlp.parameters())}")

# Concatenator
concatenator = ToolAwareEmbedding.from_config(config)
print(f"   - ToolAwareEmbedding: d_toolaware={concatenator.d_toolaware}")

# Test individual components
print("\n" + "=" * 60)
print("Test 1: Individual Component Forward Pass")
print("=" * 60)

# Get a batch of samples
batch_samples = [tool_dataset[i] for i in sample_indices]
batch_tool_names = [s['tool_name'] for s in batch_samples]
batch_resources = torch.stack([torch.tensor(s['resource_vector'], dtype=torch.float32) for s in batch_samples])

print(f"Batch size: {len(batch_samples)}")
print(f"Resource vectors shape: {batch_resources.shape}")

# Tool encoding
tool_embeddings = tool_encoder(tool_names=batch_tool_names)
print(f"\n1. Tool Encoder:")
print(f"   - Output shape: {tool_embeddings.shape}")
print(f"   - Expected: ({len(batch_samples)}, {tool_encoder.d_tool})")
assert tool_embeddings.shape == (len(batch_samples), tool_encoder.d_tool)
print("   ✓ Dimension check passed")

# Resource encoding
resource_embeddings = resource_mlp(batch_resources)
print(f"\n2. Resource MLP:")
print(f"   - Output shape: {resource_embeddings.shape}")
print(f"   - Expected: ({len(batch_samples)}, {resource_mlp.d_resource})")
assert resource_embeddings.shape == (len(batch_samples), resource_mlp.d_resource)
print("   ✓ Dimension check passed")

# Concatenation
toolaware_embeddings = concatenator(tool_embeddings, resource_embeddings)
print(f"\n3. Concatenation:")
print(f"   - Output shape: {toolaware_embeddings.shape}")
print(f"   - Expected: ({len(batch_samples)}, {concatenator.d_toolaware})")
assert toolaware_embeddings.shape == (len(batch_samples), concatenator.d_toolaware)
print("   ✓ Dimension check passed")

# Verify concatenation is correct
tool_part, resource_part = concatenator.split(toolaware_embeddings)
diff_tool = (tool_embeddings - tool_part).abs().max().item()
diff_resource = (resource_embeddings - resource_part).abs().max().item()
print(f"\n4. Verify split:")
print(f"   - Tool reconstruction error: {diff_tool:.10f}")
print(f"   - Resource reconstruction error: {diff_resource:.10f}")
assert diff_tool < 1e-6 and diff_resource < 1e-6
print("   ✓ Concatenation verified correct")

# Test end-to-end wrapper
print("\n" + "=" * 60)
print("Test 2: End-to-End ResourceAwareToolEncoder")
print("=" * 60)

e2e_encoder = ResourceAwareToolEncoder(
    tool_encoder=tool_encoder,
    resource_mlp=resource_mlp,
    concatenator=concatenator
)
print(f"Created ResourceAwareToolEncoder with output_dim={e2e_encoder.get_output_dim()}")

toolaware_e2e = e2e_encoder(
    tool_names=batch_tool_names,
    resource_vectors=batch_resources
)
print(f"Output shape: {toolaware_e2e.shape}")
assert toolaware_e2e.shape == (len(batch_samples), e2e_encoder.get_output_dim())
print("✓ End-to-end forward pass successful")

# Verify same results as individual components
diff = (toolaware_embeddings - toolaware_e2e).abs().max().item()
print(f"Max difference vs component-wise: {diff:.10f}")
assert diff < 1e-6
print("✓ End-to-end matches component-wise processing")

# Test gradient flow through entire pipeline
print("\n" + "=" * 60)
print("Test 3: Gradient Flow Through Pipeline")
print("=" * 60)

# Create fresh batch with gradients enabled
batch_resources_grad = torch.stack([torch.tensor(s['resource_vector'], dtype=torch.float32) for s in batch_samples]).requires_grad_(True)

# Forward pass
toolaware_grad = e2e_encoder(
    tool_names=batch_tool_names,
    resource_vectors=batch_resources_grad
)

# Backward pass
loss = toolaware_grad.sum()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Resource gradients: {batch_resources_grad.grad is not None}")
print(f"ResourceMLP has gradients: {any(p.grad is not None for p in resource_mlp.parameters())}")

# Check resource MLP parameter gradients
mlp_grads = [p.grad for p in resource_mlp.parameters() if p.grad is not None]
print(f"Number of MLP parameters with gradients: {len(mlp_grads)}")
assert len(mlp_grads) > 0
print("✓ Gradients flow through entire pipeline")

# Test all tools
print("\n" + "=" * 60)
print("Test 4: All Tools Processing")
print("=" * 60)

# Get all profiling samples
all_tool_names = [tool_dataset[i]['tool_name'] for i in range(len(tool_dataset))]
all_resources = torch.stack([torch.tensor(tool_dataset[i]['resource_vector'], dtype=torch.float32) for i in range(len(tool_dataset))])

print(f"Processing all {len(tool_dataset)} samples...")

all_toolaware = e2e_encoder(
    tool_names=all_tool_names,
    resource_vectors=all_resources
)

print(f"Output shape: {all_toolaware.shape}")
assert all_toolaware.shape == (len(tool_dataset), e2e_encoder.get_output_dim())
print("✓ All samples processed successfully")

# Compute statistics
print(f"\nEmbedding statistics:")
print(f"  - Mean: {all_toolaware.mean(dim=0).mean().item():.4f}")
print(f"  - Std: {all_toolaware.std(dim=0).mean().item():.4f}")
print(f"  - Min: {all_toolaware.min().item():.4f}")
print(f"  - Max: {all_toolaware.max().item():.4f}")

# Check norms per tool
norms = all_toolaware.norm(dim=1)
print(f"\nNorms per sample:")
print(f"  - Mean: {norms.mean().item():.4f}")
print(f"  - Std: {norms.std().item():.4f}")
print(f"  - Min: {norms.min().item():.4f}")
print(f"  - Max: {norms.max().item():.4f}")

# Group by tool and check consistency
print(f"\nNorms by tool (3 samples each):")
for tool_idx, tool_name in enumerate(tool_names):
    start_idx = tool_idx * 3
    end_idx = start_idx + 3
    tool_norms = norms[start_idx:end_idx]
    print(f"  {tool_name:15s}: mean={tool_norms.mean().item():6.2f}, "
          f"std={tool_norms.std().item():5.2f}, "
          f"range=[{tool_norms.min().item():.2f}, {tool_norms.max().item():.2f}]")

# Test from_config factory method
print("\n" + "=" * 60)
print("Test 5: from_config() Factory Method")
print("=" * 60)

e2e_from_config = ResourceAwareToolEncoder.from_config(
    config,
    tool_names=tool_names,
    encoder_type="name"
)
print(f"Created from config: output_dim={e2e_from_config.get_output_dim()}")

toolaware_from_config = e2e_from_config(
    tool_names=batch_tool_names,
    resource_vectors=batch_resources
)
print(f"Output shape: {toolaware_from_config.shape}")
assert toolaware_from_config.shape == toolaware_e2e.shape
print("✓ from_config() works correctly")

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED ✓")
print("=" * 60)
print(f"\nSummary:")
print(f"  - Input: {num_tools} tools × 3 sizes = {num_samples} samples")
print(f"  - Tool embeddings: {tool_encoder.d_tool}D")
print(f"  - Resource embeddings: {resource_mlp.d_resource}D")
print(f"  - Tool-aware embeddings: {concatenator.d_toolaware}D")
print(f"  - MLP parameters: {sum(p.numel() for p in resource_mlp.parameters())}")
print(f"  ✓ All components integrated successfully!")
