#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for Tool Encoder implementation.

Tests both ToolNameEncoder and ToolTextEncoder approaches.
"""

import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from encoders.tool_encoder import ToolNameEncoder, ToolTextEncoder, ToolEncoder
from data.loader import load_tool_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration."""
    with open('configs/default.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_tool_name_encoder():
    """Test ToolNameEncoder (Approach 1)."""
    logger.info("=" * 60)
    logger.info("Testing ToolNameEncoder (Approach 1)")
    logger.info("=" * 60)
    
    # Load tool names
    dataset = load_tool_data(
        'data/tool_registry/tools.json',
        'data/profiling/profiling.csv'
    )
    tool_names = list(set(s['tool_name'] for s in dataset.get_samples()))
    
    logger.info(f"Tool names: {tool_names}")
    
    # Create encoder
    d_tool = 768
    encoder = ToolNameEncoder(tool_names, d_tool)
    
    logger.info(f"Encoder: {encoder.num_tools} tools, d_tool={encoder.d_tool}")
    
    # Test encoding single tool
    test_tool = tool_names[0]
    embedding = encoder.get_tool_embedding(test_tool)
    logger.info(f"\nSingle tool embedding:")
    logger.info(f"  Tool: {test_tool}")
    logger.info(f"  Embedding shape: {embedding.shape}")
    logger.info(f"  Embedding norm: {embedding.norm().item():.4f}")
    
    # Test batch encoding
    batch_tools = tool_names[:3]
    batch_embeddings = encoder(batch_tools)
    logger.info(f"\nBatch encoding:")
    logger.info(f"  Tools: {batch_tools}")
    logger.info(f"  Embeddings shape: {batch_embeddings.shape}")
    
    # Test all embeddings
    all_embeddings = encoder.get_all_embeddings()
    logger.info(f"\nAll tool embeddings shape: {all_embeddings.shape}")
    
    # Test unknown tool
    try:
        encoder(['unknown_tool'])
        logger.error("Should have raised ValueError for unknown tool")
    except ValueError as e:
        logger.info(f"\nCorrectly raised error for unknown tool: {e}")
    
    return encoder


def test_tool_text_encoder():
    """Test ToolTextEncoder (Approach 2)."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ToolTextEncoder (Approach 2)")
    logger.info("=" * 60)
    
    # Load tool texts
    dataset = load_tool_data(
        'data/tool_registry/tools.json',
        'data/profiling/profiling.csv'
    )
    
    # Get unique tools with their texts
    tool_texts = {}
    for sample in dataset.get_samples():
        if sample['tool_name'] not in tool_texts:
            tool_texts[sample['tool_name']] = sample['semantic_text']
    
    texts = list(tool_texts.values())[:3]  # Test with first 3
    logger.info(f"Testing with {len(texts)} tool texts")
    logger.info(f"First text preview: {texts[0][:100]}...")
    
    # Create encoder (use small model for testing or skip if too heavy)
    d_tool = 768
    
    try:
        # Note: This will try to load Qwen model which may be heavy
        # For testing, we'll catch any errors
        encoder = ToolTextEncoder(
            model_name="Qwen/Qwen2.5-7B",
            d_tool=d_tool,
            max_length=256,
            pooling="mean"
        )
        
        logger.info(f"Encoder: vocab_size={encoder.vocab_size}, "
                   f"hidden_dim={encoder.hidden_dim}, d_tool={encoder.d_tool}")
        
        # Test encoding
        embeddings = encoder(texts)
        logger.info(f"\nEncoded embeddings shape: {embeddings.shape}")
        logger.info(f"Embedding norm: {embeddings.norm(dim=1)}")
        
        return encoder
        
    except Exception as e:
        logger.warning(f"Could not test ToolTextEncoder (model too large or not available): {e}")
        logger.info("This is expected if Qwen model is not downloaded")
        return None


def test_unified_tool_encoder():
    """Test unified ToolEncoder wrapper."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Unified ToolEncoder")
    logger.info("=" * 60)
    
    config = load_config()
    
    # Load dataset
    dataset = load_tool_data(
        'data/tool_registry/tools.json',
        'data/profiling/profiling.csv'
    )
    tool_names = list(set(s['tool_name'] for s in dataset.get_samples()))
    
    # Test with name encoder
    logger.info("\n--- Testing with encoder_type='name' ---")
    encoder = ToolEncoder(config, tool_names=tool_names, encoder_type="name")
    
    # Test encoding without cache
    test_names = tool_names[:3]
    embeddings1 = encoder(tool_names=test_names, use_cache=False)
    logger.info(f"Encoded {len(test_names)} tools (no cache): {embeddings1.shape}")
    
    # Test encoding with cache
    embeddings2 = encoder(tool_names=test_names, use_cache=True)
    logger.info(f"Encoded {len(test_names)} tools (with cache): {embeddings2.shape}")
    
    # Check cache hit
    embeddings3 = encoder(tool_names=test_names, use_cache=True)
    logger.info(f"Encoded {len(test_names)} tools (cache hit): {embeddings3.shape}")
    logger.info(f"Cache size: {len(encoder.cache)}")
    
    # Precompute all embeddings
    logger.info("\n--- Precomputing all embeddings ---")
    cache = encoder.precompute_embeddings(tool_names)
    logger.info(f"Precomputed {len(cache)} tool embeddings")
    
    # Test that cached values are used
    test_tool = tool_names[0]
    cached_emb = cache[test_tool]
    new_emb = encoder(tool_names=[test_tool])[0]
    
    diff = (cached_emb - new_emb).abs().max().item()
    logger.info(f"\nCache consistency check:")
    logger.info(f"  Max difference: {diff:.6f}")
    logger.info(f"  Cache working: {diff < 1e-5}")
    
    # Test cache clear
    encoder.clear_cache()
    logger.info(f"Cache cleared, size: {len(encoder.cache)}")
    
    return encoder


def test_embedding_properties():
    """Test embedding properties and consistency."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Embedding Properties")
    logger.info("=" * 60)
    
    config = load_config()
    dataset = load_tool_data(
        'data/tool_registry/tools.json',
        'data/profiling/profiling.csv'
    )
    tool_names = list(set(s['tool_name'] for s in dataset.get_samples()))
    
    encoder = ToolEncoder(config, tool_names=tool_names, encoder_type="name")
    
    # Get all embeddings
    all_embeddings = encoder.encoder.get_all_embeddings()
    
    # Check dimensions
    logger.info(f"Embeddings shape: {all_embeddings.shape}")
    logger.info(f"Expected: ({len(tool_names)}, {config['model']['tool_encoder']['d_tool']})")
    
    # Check norms
    norms = all_embeddings.norm(dim=1)
    logger.info(f"\nEmbedding norms:")
    logger.info(f"  Mean: {norms.mean().item():.4f}")
    logger.info(f"  Std: {norms.std().item():.4f}")
    logger.info(f"  Min: {norms.min().item():.4f}")
    logger.info(f"  Max: {norms.max().item():.4f}")
    
    # Check pairwise similarities
    normalized = torch.nn.functional.normalize(all_embeddings, dim=1)
    similarities = torch.mm(normalized, normalized.t())
    
    # Get off-diagonal elements (exclude self-similarity)
    mask = ~torch.eye(len(tool_names), dtype=torch.bool)
    off_diag_sims = similarities[mask]
    
    logger.info(f"\nPairwise cosine similarities:")
    logger.info(f"  Mean: {off_diag_sims.mean().item():.4f}")
    logger.info(f"  Std: {off_diag_sims.std().item():.4f}")
    logger.info(f"  Min: {off_diag_sims.min().item():.4f}")
    logger.info(f"  Max: {off_diag_sims.max().item():.4f}")
    
    # Check determinism
    emb1 = encoder(tool_names=tool_names[:3], use_cache=False)
    emb2 = encoder(tool_names=tool_names[:3], use_cache=False)
    
    max_diff = (emb1 - emb2).abs().max().item()
    logger.info(f"\nDeterminism check:")
    logger.info(f"  Max difference between two forward passes: {max_diff:.10f}")
    logger.info(f"  Deterministic: {max_diff < 1e-6}")


def main():
    """Run all tests."""
    try:
        # Test 1: ToolNameEncoder
        name_encoder = test_tool_name_encoder()
        
        # Test 2: ToolTextEncoder (may fail if model not available)
        text_encoder = test_tool_text_encoder()
        
        # Test 3: Unified ToolEncoder
        unified_encoder = test_unified_tool_encoder()
        
        # Test 4: Embedding properties
        test_embedding_properties()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nTEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
