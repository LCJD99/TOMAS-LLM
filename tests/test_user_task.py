#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for UserTaskEncoder and TaskEmbedding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml

print("=" * 60)
print("Testing UserTaskEncoder & TaskEmbedding")
print("=" * 60)

# Load config - use simple-test.yaml for faster testing
config_path = 'configs/simple-test.yaml'
print(f"Loading config from: {config_path}")

with open(config_path) as f:
    config = yaml.safe_load(f)

# Use config settings (already set to CPU and 0.5B model)
print(f"Using model: {config['model']['backbone']['name']}")
print(f"Device: {config['model']['backbone']['device']}")

# Test 1: TaskEmbedding Creation
print("\n" + "=" * 60)
print("Test 1: TaskEmbedding Initialization")
print("=" * 60)

try:
    from context.user_task import TaskEmbedding
    
    print(f"Loading model: {config['model']['backbone']['name']}")
    print("This may take a minute...")
    
    task_emb = TaskEmbedding.from_config(config)
    print(f"✓ TaskEmbedding created successfully")
    print(f"  - Model: {task_emb.model_name}")
    print(f"  - Vocab size: {task_emb.get_vocab_size():,}")
    print(f"  - Embedding dim: {task_emb.get_embedding_dim()}")
    print(f"  - Max length: {task_emb.max_length}")
    print(f"  - Device: {task_emb.device}")
    
except Exception as e:
    print(f"✗ Failed to create TaskEmbedding: {e}")
    print("\nNote: This test requires downloading Qwen2.5-7B (~14GB)")
    print("You can skip this test if the model is not available.")
    sys.exit(0)  # Exit gracefully

# Test 2: Tokenization
print("\n" + "=" * 60)
print("Test 2: Text Tokenization")
print("=" * 60)

test_texts = [
    "Generate an image of a sunset over mountains",
    "Analyze the sentiment of customer reviews",
    "Transcribe this audio file to text"
]

print("Test texts:")
for i, text in enumerate(test_texts, 1):
    print(f"  {i}. {text}")

encoding = task_emb.tokenize(test_texts)
print(f"\nTokenization results:")
print(f"  - Input IDs shape: {encoding['input_ids'].shape}")
print(f"  - Attention mask shape: {encoding['attention_mask'].shape}")
print(f"  - Batch size: {encoding['input_ids'].size(0)}")
print(f"  - Sequence length: {encoding['input_ids'].size(1)}")

# Show tokens for first text
first_tokens = encoding['input_ids'][0]
first_text_decoded = task_emb.tokenizer.decode(first_tokens)
print(f"\nFirst text tokens (first 10): {first_tokens[:10].tolist()}")
print(f"Decoded: {first_text_decoded[:100]}...")

print("✓ Tokenization successful")

# Test 3: Embedding Generation
print("\n" + "=" * 60)
print("Test 3: Embedding Generation")
print("=" * 60)

embeddings, attention_mask = task_emb(test_texts)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Expected: (batch_size={len(test_texts)}, seq_len, d_model={task_emb.d_model})")
print(f"Attention mask shape: {attention_mask.shape}")

assert embeddings.shape[0] == len(test_texts)
assert embeddings.shape[2] == task_emb.d_model
print("✓ Shape validation passed")

# Check embedding statistics
print(f"\nEmbedding statistics:")
print(f"  - Mean: {embeddings.mean().item():.6f}")
print(f"  - Std: {embeddings.std().item():.6f}")
print(f"  - Min: {embeddings.min().item():.6f}")
print(f"  - Max: {embeddings.max().item():.6f}")

print("✓ Embedding generation successful")

# Test 4: UserTaskEncoder
print("\n" + "=" * 60)
print("Test 4: UserTaskEncoder")
print("=" * 60)

from context.user_task import UserTaskEncoder

# Without projection
user_encoder = UserTaskEncoder.from_config(config, project_to_tool_dim=False)
print(f"Created UserTaskEncoder (no projection)")
print(f"  - Output dim: {user_encoder.get_output_dim()}")
print(f"  - Pooling method: {user_encoder.pooling_method}")

seq_emb, pooled_emb, attn_mask = user_encoder(test_texts, return_pooled=True)
print(f"\nOutputs:")
print(f"  - Sequence embeddings: {seq_emb.shape}")
print(f"  - Pooled embeddings: {pooled_emb.shape}")
print(f"  - Attention mask: {attn_mask.shape}")

assert seq_emb.shape == embeddings.shape
assert pooled_emb.shape == (len(test_texts), user_encoder.output_dim)
print("✓ UserTaskEncoder forward pass successful")

# Test 5: Different Pooling Methods
print("\n" + "=" * 60)
print("Test 5: Pooling Methods Comparison")
print("=" * 60)

pooling_methods = ["mean", "max", "cls", "last"]
pooled_results = {}

for method in pooling_methods:
    encoder = UserTaskEncoder(
        task_embedding=task_emb,
        project_to_tool_dim=False,
        pooling_method=method
    )
    _, pooled, _ = encoder(test_texts[:1], return_pooled=True)  # Use only first text
    pooled_results[method] = pooled.squeeze()
    print(f"{method:8s}: shape={pooled.shape}, norm={pooled.norm().item():.4f}")

# Compare pooling results
print("\nPooling method differences:")
mean_pooled = pooled_results['mean']
for method in ['max', 'cls', 'last']:
    diff = (mean_pooled - pooled_results[method]).abs().max().item()
    print(f"  mean vs {method}: max_diff={diff:.6f}")

print("✓ All pooling methods work correctly")

# Test 6: Projection to Tool Dimension
print("\n" + "=" * 60)
print("Test 6: Projection to Tool Dimension")
print("=" * 60)

user_encoder_proj = UserTaskEncoder.from_config(config, project_to_tool_dim=True)
d_toolaware = config['model']['tool_encoder']['d_tool'] + config['model']['resource_mlp']['d_resource']
print(f"Created UserTaskEncoder with projection")
print(f"  - Input dim: {user_encoder_proj.d_model}")
print(f"  - Output dim: {user_encoder_proj.get_output_dim()}")
print(f"  - Expected (d_toolaware): {d_toolaware}")

seq_emb_proj, pooled_emb_proj, _ = user_encoder_proj(test_texts, return_pooled=True)
print(f"\nProjected outputs:")
print(f"  - Sequence embeddings: {seq_emb_proj.shape}")
print(f"  - Pooled embeddings: {pooled_emb_proj.shape}")

assert seq_emb_proj.shape[-1] == d_toolaware
assert pooled_emb_proj.shape[-1] == d_toolaware
print("✓ Projection to tool dimension successful")

# Test 7: Gradient Flow
print("\n" + "=" * 60)
print("Test 7: Gradient Flow (Projection Layer)")
print("=" * 60)

# Only projection layer is trainable (embeddings frozen)
seq_proj, pooled_proj, _ = user_encoder_proj(test_texts, return_pooled=True)
loss = pooled_proj.sum()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Projection layer has gradients: {user_encoder_proj.projection.weight.grad is not None}")
print(f"Embedding layer frozen: {not task_emb.embeddings.weight.requires_grad}")

assert user_encoder_proj.projection.weight.grad is not None
print("✓ Gradients flow through projection layer")

# Test 8: Single vs Batch Processing
print("\n" + "=" * 60)
print("Test 8: Single vs Batch Processing")
print("=" * 60)

single_text = test_texts[0]
print(f"Processing single text: '{single_text[:50]}...'")

# Single text
_, pooled_single, _ = user_encoder(single_text, return_pooled=True)
print(f"Single result shape: {pooled_single.shape}")

# Batch with same text
batch_same = [single_text]
_, pooled_batch, _ = user_encoder(batch_same, return_pooled=True)
print(f"Batch result shape: {pooled_batch.shape}")

# Compare results
diff = (pooled_single - pooled_batch).abs().max().item()
print(f"Max difference: {diff:.10f}")
assert diff < 1e-5
print("✓ Single and batch processing consistent")

# Test 9: Variable Length Sequences
print("\n" + "=" * 60)
print("Test 9: Variable Length Sequences")
print("=" * 60)

varied_texts = [
    "Short task",
    "This is a medium length task description with more words",
    "This is a very long task description that contains many words and should test the model's ability to handle variable length sequences with proper attention masking"
]

print("Text lengths:")
for i, text in enumerate(varied_texts, 1):
    tokens = task_emb.tokenizer.encode(text)
    print(f"  {i}. {len(tokens):3d} tokens: {text[:50]}...")

_, pooled_varied, mask_varied = user_encoder(varied_texts, return_pooled=True)
print(f"\nPooled embeddings: {pooled_varied.shape}")
print(f"Attention mask sum per sequence: {mask_varied.sum(dim=1).tolist()}")

# Check that different lengths produce different embeddings
diff_01 = (pooled_varied[0] - pooled_varied[1]).norm().item()
diff_12 = (pooled_varied[1] - pooled_varied[2]).norm().item()
print(f"Embedding differences:")
print(f"  Text 0 vs 1: {diff_01:.4f}")
print(f"  Text 1 vs 2: {diff_12:.4f}")
assert diff_01 > 0 and diff_12 > 0
print("✓ Variable length sequences handled correctly")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)

print("\nSummary:")
print(f"  - TaskEmbedding: vocab_size={task_emb.get_vocab_size():,}, d_model={task_emb.d_model}")
print(f"  - UserTaskEncoder: output_dim={user_encoder.get_output_dim()}")
print(f"  - Projection enabled: output_dim={user_encoder_proj.get_output_dim()}")
print(f"  - Pooling methods tested: {', '.join(pooling_methods)}")
print(f"  ✓ Ready for integration with LLM backbone!")
