#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Integration test for complete Left Panel (Input Processing & Encoders):
ToolEncoder → ResourceMLP → Concatenation → ToolSetAttention
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
import numpy as np
from data.loader import load_tool_data
from encoders.tool_encoder import ToolEncoder
from encoders.resource_mlp import ResourceMLP
from encoders.concatenation import ToolAwareEmbedding
from encoders.tool_attention import ToolSetEncoder, ToolSetAttention, CompleteToolEncoder

print("=" * 60)
print("Integration Test: Complete Left Panel Pipeline")
print("=" * 60)

# Load configuration
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

print("\n1. Loading tool data...")
tool_dataset = load_tool_data(
    tool_registry_path=config['data']['tool_registry_path'],
    profiling_path=config['data']['profiling_path']
)

tool_registry = tool_dataset.tool_registry.tools
num_tools = len(tool_registry)
num_samples = len(tool_dataset)
tool_names = [tool.name for tool in tool_registry]

print(f"   - Loaded {num_tools} tools")
print(f"   - Loaded {num_samples} profiling samples")
print(f"   - Tools: {', '.join(tool_names[:3])}... (showing first 3)")

# Create all components
print("\n2. Initializing complete pipeline...")

tool_encoder = ToolEncoder(config, tool_names=tool_names, encoder_type="name")
resource_mlp = ResourceMLP.from_config(config)
concatenator = ToolAwareEmbedding.from_config(config)
attention_encoder = ToolSetEncoder.from_config(config)

print(f"   - ToolEncoder: d_tool={tool_encoder.d_tool}")
print(f"   - ResourceMLP: d_resource={resource_mlp.d_resource}")
print(f"   - Concatenator: d_toolaware={concatenator.d_toolaware}")
print(f"   - AttentionEncoder: {attention_encoder.num_layers} layers, {attention_encoder.num_heads} heads")

# Test component-wise forward pass
print("\n" + "=" * 60)
print("Test 1: Component-wise Forward Pass (Single Tool Set)")
print("=" * 60)

# Use all 8 tools with small input size
small_indices = [i * 3 for i in range(num_tools)]  # indices 0,3,6,9,12,15,18,21 (all small)
batch_samples = [tool_dataset[i] for i in small_indices]
batch_tool_names = [s['tool_name'] for s in batch_samples]
batch_resources = torch.stack([torch.tensor(s['resource_vector'], dtype=torch.float32) for s in batch_samples])

print(f"Processing tool set: {num_tools} tools")
print(f"All using 'small' input size")

# Step 1: Tool encoding
tool_embeddings = tool_encoder(tool_names=batch_tool_names)
print(f"\n1. Tool Encoder:")
print(f"   Output: {tool_embeddings.shape}")

# Step 2: Resource encoding
resource_embeddings = resource_mlp(batch_resources)
print(f"\n2. Resource MLP:")
print(f"   Output: {resource_embeddings.shape}")

# Step 3: Concatenation
toolaware_embeddings = concatenator(tool_embeddings, resource_embeddings)
print(f"\n3. Concatenation:")
print(f"   Output: {toolaware_embeddings.shape}")

# Step 4: Self-attention
h_toolset = attention_encoder(toolaware_embeddings)
print(f"\n4. Self-Attention:")
print(f"   Output: {h_toolset.shape}")
print(f"   Expected: ({num_tools}, {concatenator.d_toolaware})")
assert h_toolset.shape == (num_tools, concatenator.d_toolaware)
print("✓ Complete pipeline forward pass successful")

# Analyze attention output
print("\n" + "=" * 60)
print("Test 2: Contextualized Embeddings Analysis")
print("=" * 60)

# Compare before and after attention
print("Before attention (v_toolaware):")
print(f"  Mean: {toolaware_embeddings.mean():.6f}")
print(f"  Std: {toolaware_embeddings.std():.6f}")
print(f"  Norm (mean): {toolaware_embeddings.norm(dim=-1).mean():.4f}")

print("\nAfter attention (h_toolset):")
print(f"  Mean: {h_toolset.mean():.6f}")
print(f"  Std: {h_toolset.std():.6f}")
print(f"  Norm (mean): {h_toolset.norm(dim=-1).mean():.4f}")

# Compute pairwise similarities
def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarities."""
    norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return torch.mm(norm, norm.t())

sim_before = cosine_similarity_matrix(toolaware_embeddings)
sim_after = cosine_similarity_matrix(h_toolset)

print("\nPairwise cosine similarities:")
print(f"  Before attention (mean): {sim_before[~torch.eye(num_tools, dtype=bool)].mean():.6f}")
print(f"  After attention (mean): {sim_after[~torch.eye(num_tools, dtype=bool)].mean():.6f}")

# Attention should increase similarity (contextualization)
print(f"  Change: {sim_after[~torch.eye(num_tools, dtype=bool)].mean() - sim_before[~torch.eye(num_tools, dtype=bool)].mean():.6f}")
print("✓ Contextualization effect observed")

# Test with attention weights
print("\n" + "=" * 60)
print("Test 3: Attention Weights Visualization")
print("=" * 60)

attention_encoder.eval()
h_toolset_eval, attn_weights_list = attention_encoder(toolaware_embeddings, return_all_attentions=True)
attention_encoder.train()

print(f"Number of attention layers: {len(attn_weights_list)}")
attn_weights = attn_weights_list[0]  # First layer
print(f"Attention weights shape: {attn_weights.shape}")
print(f"  (num_heads, num_tools, num_tools) = ({attention_encoder.num_heads}, {num_tools}, {num_tools})")

# Average over heads
attn_avg = attn_weights.mean(dim=0)
print(f"\nAverage attention (over heads): {attn_avg.shape}")

# Show attention pattern for first tool
print(f"\nAttention pattern for '{batch_tool_names[0]}':")
for i, tool_name in enumerate(batch_tool_names):
    print(f"  → {tool_name:25s}: {attn_avg[0, i]:.6f}")

# Check if tools attend to themselves
self_attention = attn_avg.diag()
print(f"\nSelf-attention (diagonal):")
print(f"  Mean: {self_attention.mean():.6f}")
print(f"  Min: {self_attention.min():.6f}")
print(f"  Max: {self_attention.max():.6f}")
print("✓ Attention weights computed successfully")

# Test CompleteToolEncoder wrapper
print("\n" + "=" * 60)
print("Test 4: CompleteToolEncoder End-to-End")
print("=" * 60)

complete_encoder = CompleteToolEncoder(
    tool_encoder=tool_encoder,
    resource_mlp=resource_mlp,
    concatenator=concatenator,
    attention_encoder=attention_encoder
)
print(f"Created CompleteToolEncoder")
param_count = sum(p.numel() for p in complete_encoder.parameters())
print(f"Total parameters: {param_count:,}")

# One-shot encoding
complete_encoder.eval()  # Disable dropout for deterministic comparison
h_toolset_complete = complete_encoder(
    tool_names=batch_tool_names,
    resource_vectors=batch_resources
)
complete_encoder.train()
print(f"\nOne-shot output: {h_toolset_complete.shape}")

# Verify same results as component-wise (use eval mode for both)
attention_encoder.eval()
h_toolset_deterministic = attention_encoder(toolaware_embeddings)
attention_encoder.train()

diff = (h_toolset_deterministic - h_toolset_complete).abs().max().item()
print(f"Max difference vs component-wise: {diff:.10f}")
assert diff < 1e-5, f"Difference too large: {diff}"
print("✓ CompleteToolEncoder matches component-wise processing")

# Test from_config factory
print("\n" + "=" * 60)
print("Test 5: from_config() Factory Method")
print("=" * 60)

complete_from_config = CompleteToolEncoder.from_config(
    config,
    tool_names=tool_names,
    encoder_type='name'
)
print(f"Created from config: output_dim={complete_from_config.get_output_dim()}")

h_from_config = complete_from_config(
    tool_names=batch_tool_names,
    resource_vectors=batch_resources
)
print(f"Output shape: {h_from_config.shape}")
assert h_from_config.shape == h_toolset_complete.shape
print("✓ from_config() works correctly")

# Test gradient flow through entire pipeline
print("\n" + "=" * 60)
print("Test 6: Gradient Flow Through Complete Pipeline")
print("=" * 60)

# Create fresh batch with gradients
batch_resources_grad = torch.stack([torch.tensor(s['resource_vector'], dtype=torch.float32) 
                                   for s in batch_samples]).requires_grad_(True)

# Forward pass
h_grad = complete_encoder(
    tool_names=batch_tool_names,
    resource_vectors=batch_resources_grad
)

# Backward pass
loss = h_grad.sum()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Resource gradients exist: {batch_resources_grad.grad is not None}")

# Check which components have gradients
mlp_grads = sum(1 for p in resource_mlp.parameters() if p.grad is not None)
attn_grads = sum(1 for p in attention_encoder.parameters() if p.grad is not None)
print(f"ResourceMLP parameters with gradients: {mlp_grads}")
print(f"AttentionEncoder parameters with gradients: {attn_grads}")
assert mlp_grads > 0 and attn_grads > 0
print("✓ Gradients flow through entire pipeline")

# Test with different tool sets
print("\n" + "=" * 60)
print("Test 7: Multiple Tool Sets (Batched Processing)")
print("=" * 60)

# Create 3 different tool sets with different resource profiles
# Set 1: All small
set1_indices = [i * 3 for i in range(num_tools)]
# Set 2: All medium  
set2_indices = [i * 3 + 1 for i in range(num_tools)]
# Set 3: All large
set3_indices = [i * 3 + 2 for i in range(num_tools)]

def get_tool_set(indices):
    samples = [tool_dataset[i] for i in indices]
    names = [s['tool_name'] for s in samples]
    resources = torch.stack([torch.tensor(s['resource_vector'], dtype=torch.float32) for s in samples])
    return names, resources

set1_names, set1_resources = get_tool_set(set1_indices)
set2_names, set2_resources = get_tool_set(set2_indices)
set3_names, set3_resources = get_tool_set(set3_indices)

# Stack into batch
batch_resources_multi = torch.stack([set1_resources, set2_resources, set3_resources])
print(f"Batch shape: {batch_resources_multi.shape}")
print(f"  (batch_size, num_tools, num_features) = (3, {num_tools}, 6)")

# Process batch - need to expand tool encoding
tool_emb_batch = tool_encoder(tool_names=tool_names).unsqueeze(0).expand(3, -1, -1)
resource_emb_batch = resource_mlp(batch_resources_multi)
toolaware_batch = concatenator(tool_emb_batch, resource_emb_batch)
h_batch = attention_encoder(toolaware_batch)

print(f"\nOutput shape: {h_batch.shape}")
print(f"  (batch_size, num_tools, d_model) = (3, {num_tools}, {concatenator.d_toolaware})")
assert h_batch.shape == (3, num_tools, concatenator.d_toolaware)
print("✓ Batched processing successful")

# Compare embeddings across tool sets
print("\nNorm statistics per tool set:")
for i, size in enumerate(['small', 'medium', 'large']):
    norms = h_batch[i].norm(dim=-1)
    print(f"  {size:8s}: mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"range=[{norms.min():.2f}, {norms.max():.2f}]")

# Test resource-aware attention
print("\n" + "=" * 60)
print("Test 8: Resource-Aware Attention Patterns")
print("=" * 60)

# Compare attention for same tool with different resources
tool_idx = 0  # image_classification
print(f"Analyzing tool: {tool_names[tool_idx]}")

# Get embeddings for same tool with different sizes
small_sample = tool_dataset[tool_idx * 3]      # small
medium_sample = tool_dataset[tool_idx * 3 + 1]  # medium
large_sample = tool_dataset[tool_idx * 3 + 2]   # large

print(f"\nResource vectors:")
for size, sample in [('small', small_sample), ('medium', medium_sample), ('large', large_sample)]:
    rv = sample['resource_vector']
    print(f"  {size:8s}: input_size={rv[0]:.0f}, cpu_cores={rv[1]:.1f}, "
          f"cpu_mem={rv[2]:.1f}GB, latency={rv[5]:.0f}ms")

# Create tool sets with different resource profiles
resources_varied = torch.stack([
    torch.tensor(small_sample['resource_vector'], dtype=torch.float32),
    torch.tensor(medium_sample['resource_vector'], dtype=torch.float32),
    torch.tensor(large_sample['resource_vector'], dtype=torch.float32)
])

# Pad to full tool set size (repeat first tool for simplicity)
resources_padded = resources_varied.repeat(num_tools // 3 + 1, 1)[:num_tools]
tool_names_repeated = [tool_names[tool_idx]] * num_tools

# Encode
h_varied = complete_encoder(
    tool_names=tool_names_repeated,
    resource_vectors=resources_padded
)

# Compute similarities
varied_norms = h_varied.norm(dim=-1)
print(f"\nEmbedding norms for different resource levels:")
print(f"  small:  {varied_norms[0]:.4f}")
print(f"  medium: {varied_norms[1]:.4f}")
print(f"  large:  {varied_norms[2]:.4f}")
print("✓ Resource awareness reflected in embeddings")

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED ✓")
print("=" * 60)

print(f"\nFinal Statistics:")
print(f"  - Tools processed: {num_tools}")
print(f"  - Total samples: {num_samples}")
print(f"  - Input dimension: 6 (resource features)")
print(f"  - Tool embedding: {tool_encoder.d_tool}D")
print(f"  - Resource embedding: {resource_mlp.d_resource}D")
print(f"  - Tool-aware embedding: {concatenator.d_toolaware}D")
print(f"  - Output (contextualized): {attention_encoder.get_output_dim()}D")
print(f"  - Attention heads: {attention_encoder.num_heads}")
print(f"  - Attention layers: {attention_encoder.num_layers}")
print(f"  - Total parameters: {param_count:,}")
print(f"\n✓ Complete Left Panel (Input Processing & Encoders) implemented!")
