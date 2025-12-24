#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for ToolSetAttention and ToolSetEncoder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from encoders.tool_attention import ToolSetAttention, ToolSetEncoder

print("=" * 60)
print("Testing ToolSetAttention & ToolSetEncoder")
print("=" * 60)

# Load config
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

d_tool = config['model']['tool_encoder']['d_tool']
d_resource = config['model']['resource_mlp']['d_resource']
d_model = d_tool + d_resource  # d_toolaware

num_heads = config['model']['tool_attention']['num_heads']
num_layers = config['model']['tool_attention']['num_layers']
dropout = config['model']['tool_attention']['dropout']

print(f"Configuration:")
print(f"  d_model (d_toolaware): {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  num_layers: {num_layers}")
print(f"  dropout: {dropout}")

# Test 1: Single ToolSetAttention layer
print("\n" + "=" * 60)
print("Test 1: Single ToolSetAttention Layer")
print("=" * 60)

attention = ToolSetAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
print(f"Created: {attention}")
print(f"Head dimension: {attention.head_dim}")

# Test unbatched input (num_tools, d_model)
num_tools = 8
x_unbatched = torch.randn(num_tools, d_model)
print(f"\nInput (unbatched): {x_unbatched.shape}")

output = attention(x_unbatched)
print(f"Output: {output.shape}")
assert output.shape == x_unbatched.shape, "Shape mismatch!"
print("✓ Unbatched forward pass successful")

# Test batched input (batch, num_tools, d_model)
batch_size = 4
x_batched = torch.randn(batch_size, num_tools, d_model)
print(f"\nInput (batched): {x_batched.shape}")

output_batched = attention(x_batched)
print(f"Output: {output_batched.shape}")
assert output_batched.shape == x_batched.shape, "Shape mismatch!"
print("✓ Batched forward pass successful")

# Test with attention weights
print("\n" + "=" * 60)
print("Test 2: Attention Weights Visualization")
print("=" * 60)

# Set to eval mode to disable dropout for cleaner attention weights
attention.eval()
output, attn_weights = attention(x_unbatched, return_attention=True)
attention.train()  # Back to training mode

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"Expected attention shape: ({num_heads}, {num_tools}, {num_tools})")
assert attn_weights.shape == (num_heads, num_tools, num_tools)
print("✓ Attention weights returned correctly")

# Check attention properties
attn_sum = attn_weights.sum(dim=-1)  # Sum over keys (should be 1.0)
print(f"\nAttention weight properties:")
print(f"  Sum over keys (should be ~1.0): mean={attn_sum.mean():.6f}, std={attn_sum.std():.6f}")
print(f"  Min: {attn_weights.min():.6f}, Max: {attn_weights.max():.6f}")
assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6)
print("✓ Attention weights sum to 1.0")

# Test gradient flow
print("\n" + "=" * 60)
print("Test 3: Gradient Flow")
print("=" * 60)

x_grad = torch.randn(num_tools, d_model, requires_grad=True)
output_grad = attention(x_grad)
loss = output_grad.sum()
loss.backward()

print(f"Input gradient: {x_grad.grad is not None}")
print(f"Attention parameters have gradients: {any(p.grad is not None for p in attention.parameters())}")
print(f"Gradient norm: {x_grad.grad.norm().item():.6f}")
assert x_grad.grad is not None
print("✓ Gradients flow correctly")

# Test ToolSetEncoder (multi-layer)
print("\n" + "=" * 60)
print("Test 4: Multi-layer ToolSetEncoder")
print("=" * 60)

encoder = ToolSetEncoder(
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
)
print(f"Created: {encoder}")
param_count = sum(p.numel() for p in encoder.parameters())
print(f"Total parameters: {param_count:,}")

# Forward pass
x_encoder = torch.randn(num_tools, d_model)
h_toolset = encoder(x_encoder)
print(f"\nInput shape: {x_encoder.shape}")
print(f"Output shape: {h_toolset.shape}")
assert h_toolset.shape == x_encoder.shape
print("✓ Multi-layer encoder forward pass successful")

# Test with batched input
x_encoder_batched = torch.randn(batch_size, num_tools, d_model)
h_toolset_batched = encoder(x_encoder_batched)
print(f"\nBatched input: {x_encoder_batched.shape}")
print(f"Batched output: {h_toolset_batched.shape}")
assert h_toolset_batched.shape == x_encoder_batched.shape
print("✓ Batched multi-layer forward pass successful")

# Test from_config
print("\n" + "=" * 60)
print("Test 5: from_config() Factory Method")
print("=" * 60)

encoder_from_config = ToolSetEncoder.from_config(config)
print(f"Created from config: {encoder_from_config}")
print(f"Output dimension: {encoder_from_config.get_output_dim()}")
assert encoder_from_config.get_output_dim() == d_model
print("✓ from_config() works correctly")

# Test with all attention weights
output_all, all_attentions = encoder_from_config(x_encoder, return_all_attentions=True)
print(f"\nOutput shape: {output_all.shape}")
print(f"Number of attention layers: {len(all_attentions)}")
print(f"Attention per layer shape: {all_attentions[0].shape}")
assert len(all_attentions) == num_layers
print("✓ All attention weights returned correctly")

# Test with feedforward networks
print("\n" + "=" * 60)
print("Test 6: ToolSetEncoder with FFN")
print("=" * 60)

encoder_ffn = ToolSetEncoder(
    d_model=d_model,
    num_heads=num_heads,
    num_layers=2,
    dropout=dropout,
    dim_feedforward=2048,
    use_ffn=True
)
print(f"Created with FFN: {encoder_ffn}")
param_count_ffn = sum(p.numel() for p in encoder_ffn.parameters())
print(f"Total parameters (with FFN): {param_count_ffn:,}")
print(f"Increase vs no FFN: {param_count_ffn - param_count:,}")

h_ffn = encoder_ffn(x_encoder)
print(f"\nOutput shape: {h_ffn.shape}")
assert h_ffn.shape == x_encoder.shape
print("✓ FFN encoder forward pass successful")

# Test gradient flow through multi-layer
print("\n" + "=" * 60)
print("Test 7: Multi-layer Gradient Flow")
print("=" * 60)

x_multilayer = torch.randn(batch_size, num_tools, d_model, requires_grad=True)
h_multilayer = encoder(x_multilayer)
loss = h_multilayer.sum()
loss.backward()

print(f"Input gradient: {x_multilayer.grad is not None}")
print(f"Encoder parameters with gradients: {sum(1 for p in encoder.parameters() if p.grad is not None)}")
print(f"Total encoder parameters: {sum(1 for p in encoder.parameters())}")
assert x_multilayer.grad is not None
print("✓ Gradients flow through all layers")

# Test residual connections
print("\n" + "=" * 60)
print("Test 8: Residual Connection Effect")
print("=" * 60)

# Without attention (just residual + norm)
attention_zero_dropout = ToolSetAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
attention_zero_dropout.eval()  # No dropout

x_residual = torch.randn(num_tools, d_model)
with torch.no_grad():
    output_residual = attention_zero_dropout(x_residual)

# Output should be different from input due to attention, but similar magnitude
input_norm = x_residual.norm().item()
output_norm = output_residual.norm().item()
print(f"Input norm: {input_norm:.4f}")
print(f"Output norm: {output_norm:.4f}")
print(f"Ratio: {output_norm / input_norm:.4f}")

# Check that output is not identical to input
diff = (x_residual - output_residual).abs().max().item()
print(f"Max difference: {diff:.6f}")
assert diff > 0.01, "Output should differ from input due to attention"
print("✓ Residual connection preserves information while adding context")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
