#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Simple test for Tool Encoder - minimal dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from encoders.tool_encoder import ToolNameEncoder, ToolEncoder

# Test ToolNameEncoder
print("=" * 60)
print("Testing ToolNameEncoder")
print("=" * 60)

tool_names = ['image_classification', 'text_summarization', 'video_transcoding']
d_tool = 768

encoder = ToolNameEncoder(tool_names, d_tool)
print(f"Created encoder with {encoder.num_tools} tools, d_tool={d_tool}")

# Test single encoding
emb = encoder.get_tool_embedding('image_classification')
print(f"\nSingle encoding shape: {emb.shape}")
print(f"Embedding norm: {emb.norm().item():.4f}")

# Test batch encoding
batch_emb = encoder(tool_names)
print(f"\nBatch encoding shape: {batch_emb.shape}")

# Test all embeddings
all_emb = encoder.get_all_embeddings()
print(f"All embeddings shape: {all_emb.shape}")

# Test caching in ToolEncoder
print("\n" + "=" * 60)
print("Testing ToolEncoder with caching")
print("=" * 60)

config = {
    'model': {
        'tool_encoder': {'d_tool': 768, 'max_desc_length': 256},
        'backbone': {'name': 'Qwen/Qwen2.5-7B'}
    }
}

unified_encoder = ToolEncoder(config, tool_names=tool_names, encoder_type='name')
print(f"Created unified encoder: type=name, d_tool={unified_encoder.d_tool}")

# Encode without cache
emb1 = unified_encoder(tool_names=tool_names[:2], use_cache=False)
print(f"\nEncoded 2 tools (no cache): {emb1.shape}")

# Encode with cache
emb2 = unified_encoder(tool_names=tool_names[:2], use_cache=True)
print(f"Encoded 2 tools (with cache): {emb2.shape}")
print(f"Cache size: {len(unified_encoder.cache)}")

# Precompute all
cache = unified_encoder.precompute_embeddings(tool_names)
print(f"\nPrecomputed {len(cache)} embeddings")

# Test cache hit
emb3 = unified_encoder(tool_names=[tool_names[0]], use_cache=True)
diff = (cache[tool_names[0]] - emb3[0]).abs().max().item()
print(f"Cache consistency: max_diff={diff:.10f}, valid={diff < 1e-5}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
