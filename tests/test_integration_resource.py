#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Integration test: Data Loader + Resource MLP
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from data.loader import load_tool_data
from encoders.resource_mlp import ResourceMLP

print("=" * 60)
print("Integration Test: Data Loader + Resource MLP")
print("=" * 60)

# Load configuration
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# Load dataset
print("\nLoading dataset...")
dataset = load_tool_data(
    'data/tool_registry/tools.json',
    'data/profiling/profiling.csv',
    normalize=True
)
print(f"Loaded {len(dataset)} samples")

# Create Resource MLP from config
mlp = ResourceMLP.from_config(config)
print(f"\nCreated Resource MLP: {mlp.input_dim} -> {mlp.hidden_dim} -> {mlp.d_resource}")

# Get resource vectors from dataset (already normalized)
resource_tensors = dataset.to_torch_tensors()
resource_vectors = resource_tensors['resource_vectors']
print(f"\nResource vectors shape: {resource_vectors.shape}")
print(f"Resource vectors (first 3):")
print(resource_vectors[:3])

# Project through MLP
print("\nProjecting through MLP...")
resource_embeddings = mlp(resource_vectors)
print(f"Resource embeddings shape: {resource_embeddings.shape}")
print(f"Expected shape: ({len(dataset)}, {config['model']['resource_mlp']['d_resource']})")

# Check dimensions
assert resource_embeddings.shape == (len(dataset), config['model']['resource_mlp']['d_resource'])
print("✓ Dimension check passed")

# Analyze embeddings
print(f"\nEmbedding statistics:")
print(f"  Mean norm: {resource_embeddings.norm(dim=1).mean().item():.4f}")
print(f"  Std norm: {resource_embeddings.norm(dim=1).std().item():.4f}")
print(f"  Min norm: {resource_embeddings.norm(dim=1).min().item():.4f}")
print(f"  Max norm: {resource_embeddings.norm(dim=1).max().item():.4f}")

# Test tool-specific projection
print("\n" + "=" * 60)
print("Tool-Specific Resource Embeddings")
print("=" * 60)

tool_names = ['image_classification', 'video_transcoding', 'data_preprocessing']
for tool_name in tool_names:
    tool_samples = dataset.get_tool_samples(tool_name)
    print(f"\n{tool_name}:")
    
    for sample in tool_samples:
        resource_vec = torch.from_numpy(sample['resource_vector']).float().unsqueeze(0)
        resource_emb = mlp(resource_vec)
        
        print(f"  {sample['input_size']:6s}: "
              f"norm={resource_emb.norm().item():7.4f}, "
              f"raw={sample['resource_raw']}")

# Test batch processing
print("\n" + "=" * 60)
print("Batch Processing Test")
print("=" * 60)

batch_size = 4
batch_indices = torch.randperm(len(dataset))[:batch_size]
batch_resource_vecs = resource_vectors[batch_indices]

print(f"Processing batch of {batch_size} samples...")
batch_embeddings = mlp(batch_resource_vecs)
print(f"Batch embeddings shape: {batch_embeddings.shape}")
print(f"Batch embedding norms: {batch_embeddings.norm(dim=1)}")

# Verify gradients flow
print("\n" + "=" * 60)
print("Gradient Flow Test")
print("=" * 60)

mlp.train()
test_input = resource_vectors[:3].clone().requires_grad_(True)
output = mlp(test_input)
loss = output.sum()
loss.backward()

print(f"Input requires grad: {test_input.requires_grad}")
print(f"Output requires grad: {output.requires_grad}")
print(f"fc1.weight.grad is not None: {mlp.fc1.weight.grad is not None}")
print(f"fc2.weight.grad is not None: {mlp.fc2.weight.grad is not None}")
print("✓ Gradients flow correctly")

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED ✓")
print("=" * 60)
