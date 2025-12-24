#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for Tool Registry & Profiling Data loaders.

This script validates the data loading functionality.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    ToolRegistryLoader,
    ProfilingDataLoader,
    ToolDataset,
    load_tool_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_tool_registry_loader():
    """Test tool registry loading."""
    logger.info("=" * 60)
    logger.info("Testing Tool Registry Loader")
    logger.info("=" * 60)
    
    registry_path = "data/tool_registry/tools.json"
    loader = ToolRegistryLoader(registry_path)
    tools, tool_dict = loader.load()
    
    logger.info(f"Loaded {len(tools)} tools")
    logger.info(f"Tool names: {loader.get_tool_names()}")
    
    # Display first tool
    if tools:
        first_tool = tools[0]
        logger.info(f"\nFirst tool:")
        logger.info(f"  Name: {first_tool.name}")
        logger.info(f"  Description: {first_tool.desc}")
    
    return loader


def test_profiling_loader(tool_registry):
    """Test profiling data loading."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Profiling Data Loader")
    logger.info("=" * 60)
    
    profiling_path = "data/profiling/profiling.csv"
    loader = ProfilingDataLoader(profiling_path, tool_registry=tool_registry)
    df = loader.load()
    
    logger.info(f"Loaded {len(df)} profiling entries")
    logger.info(f"\nProfiling DataFrame head:")
    logger.info(f"\n{df.head()}")
    
    # Get profiling matrix
    matrix, metadata = loader.get_profiling_matrix(normalize=True)
    logger.info(f"\nProfiling matrix shape: {matrix.shape}")
    logger.info(f"Feature names: {metadata['feature_names']}")
    logger.info(f"Normalization mean: {metadata['mean'][:3]}...")
    logger.info(f"Normalization std: {metadata['std'][:3]}...")
    
    # Test tool-specific profiling
    tool_name = "image_classification"
    tool_prof = loader.get_tool_profiling(tool_name)
    logger.info(f"\nProfiling for '{tool_name}':")
    logger.info(f"\n{tool_prof}")
    
    return loader


def test_tool_dataset(tool_registry, profiling_loader):
    """Test combined tool dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Combined Tool Dataset")
    logger.info("=" * 60)
    
    dataset = ToolDataset(tool_registry, profiling_loader, normalize=True)
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Display first few samples
    logger.info("\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        logger.info(f"\nSample {i}:")
        logger.info(f"  Tool: {sample['tool_name']}")
        logger.info(f"  Input size: {sample['input_size']}")
        logger.info(f"  Semantic text: {sample['semantic_text'][:80]}...")
        logger.info(f"  Resource vector shape: {sample['resource_vector'].shape}")
        logger.info(f"  Resource raw: {sample['resource_raw']}")
    
    # Convert to tensors
    tensors = dataset.to_torch_tensors()
    logger.info(f"\nTensor conversion:")
    logger.info(f"  Resource vectors shape: {tensors['resource_vectors'].shape}")
    logger.info(f"  Resource vectors dtype: {tensors['resource_vectors'].dtype}")
    
    return dataset


def test_convenience_function():
    """Test the convenience load_tool_data function."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Convenience Function")
    logger.info("=" * 60)
    
    dataset = load_tool_data(
        tool_registry_path="data/tool_registry/tools.json",
        profiling_path="data/profiling/profiling.csv",
        normalize=True
    )
    
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Test getting samples for specific tool
    tool_samples = dataset.get_tool_samples("video_transcoding")
    logger.info(f"\nSamples for 'video_transcoding': {len(tool_samples)}")
    for sample in tool_samples:
        logger.info(f"  {sample['input_size']}: {sample['resource_raw']}")
    
    return dataset


def main():
    """Run all tests."""
    try:
        # Test 1: Tool Registry
        tool_registry = test_tool_registry_loader()
        
        # Test 2: Profiling Data
        profiling_loader = test_profiling_loader(tool_registry)
        
        # Test 3: Combined Dataset
        dataset = test_tool_dataset(tool_registry, profiling_loader)
        
        # Test 4: Convenience Function
        dataset2 = test_convenience_function()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nTEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
