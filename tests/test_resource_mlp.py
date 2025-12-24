#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for Resource MLP implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from encoders.resource_mlp import ResourceMLP, ResourceNormalizer

print("=" * 60)
print("Testing ResourceMLP")
print("=" * 60)

# Test basic MLP
input_dim = 6
hidden_dim = 512
d_resource = 256

mlp = ResourceMLP(input_dim=input_dim, hidden_dim=hidden_dim, d_resource=d_resource)
print(f"Created MLP: {input_dim} -> {hidden_dim} -> {d_resource}")
print(f"Parameters: {sum(p.numel() for p in mlp.parameters())}")

# Test single sample
single_input = torch.randn(input_dim)
single_output = mlp(single_input)
print(f"\nSingle sample:")
print(f"  Input shape: {single_input.shape}")
print(f"  Output shape: {single_output.shape}")
print(f"  Output norm: {single_output.norm().item():.4f}")

# Test batch
batch_size = 8
batch_input = torch.randn(batch_size, input_dim)
batch_output = mlp(batch_input)
print(f"\nBatch:")
print(f"  Input shape: {batch_input.shape}")
print(f"  Output shape: {batch_output.shape}")
print(f"  Output norms: {batch_output.norm(dim=1)}")

# Test with dropout and batch norm
mlp_with_reg = ResourceMLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    d_resource=d_resource,
    dropout=0.1,
    use_batch_norm=True
)
print(f"\nMLP with regularization:")
print(f"  Dropout: 0.1")
print(f"  Batch norm: True")

mlp_with_reg.eval()  # Set to eval mode
output_eval = mlp_with_reg(batch_input)
print(f"  Output shape (eval): {output_eval.shape}")

# Test from config
print("\n" + "=" * 60)
print("Testing from_config")
print("=" * 60)

config = {
    'model': {
        'resource_mlp': {
            'input_features': 6,
            'hidden_dim': 512,
            'd_resource': 256,
            'dropout': 0.1,
            'use_batch_norm': False
        }
    }
}

mlp_from_config = ResourceMLP.from_config(config)
print(f"Created MLP from config: {mlp_from_config.input_dim} -> "
      f"{mlp_from_config.hidden_dim} -> {mlp_from_config.d_resource}")

output = mlp_from_config(batch_input)
print(f"Output shape: {output.shape}")

# Test ResourceNormalizer
print("\n" + "=" * 60)
print("Testing ResourceNormalizer")
print("=" * 60)

# Create sample data
n_samples = 100
feature_names = ['input_size', 'cpu_core', 'cpu_mem_gb', 'gpu_sm', 'gpu_mem_gb', 'latency_ms']
raw_features = torch.randn(n_samples, 6) * 10 + 5  # Random data with mean ~5, std ~10

print(f"Raw features shape: {raw_features.shape}")
print(f"Raw mean: {raw_features.mean(dim=0)}")
print(f"Raw std: {raw_features.std(dim=0)}")

# Fit normalizer
normalizer = ResourceNormalizer()
normalizer.fit(raw_features, feature_names)

# Normalize
normalized = normalizer.normalize(raw_features)
print(f"\nNormalized features:")
print(f"  Mean: {normalized.mean(dim=0)}")
print(f"  Std: {normalized.std(dim=0)}")

# Denormalize
denormalized = normalizer.denormalize(normalized)
print(f"\nDenormalized features:")
print(f"  Mean: {denormalized.mean(dim=0)}")
print(f"  Max diff from original: {(denormalized - raw_features).abs().max().item():.6f}")

# Test state dict
state = normalizer.state_dict()
print(f"\nState dict keys: {state.keys()}")

new_normalizer = ResourceNormalizer()
new_normalizer.load_state_dict(state)
print(f"Loaded state into new normalizer")

# Test integration with MLP
print("\n" + "=" * 60)
print("Testing MLP + Normalizer Integration")
print("=" * 60)

# Simulate resource features
test_features = torch.tensor([
    [0.0, 2.0, 4.0, 20.0, 2.0, 150.0],  # Small task
    [1.0, 4.0, 8.0, 40.0, 4.0, 350.0],  # Medium task
    [2.0, 8.0, 16.0, 80.0, 8.0, 800.0]  # Large task
])

print(f"Test features (raw):")
print(test_features)

# Normalize
normalized_features = normalizer.normalize(test_features)
print(f"\nTest features (normalized):")
print(normalized_features)

# Project through MLP
resource_embeddings = mlp(normalized_features)
print(f"\nResource embeddings:")
print(f"  Shape: {resource_embeddings.shape}")
print(f"  Norms: {resource_embeddings.norm(dim=1)}")

# Check output dimension consistency
assert resource_embeddings.shape == (3, d_resource), "Output dimension mismatch"

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
