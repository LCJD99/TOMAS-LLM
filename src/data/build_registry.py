"""
Build tool registry from raw data.

This script reads tools.json and profiling.csv, then generates a comprehensive
tool_registry.json that maps virtual tokens to their configurations.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from .resource_binning import value_to_level
    from .token_schema import generate_token_name, TOOL_ABBREV
except ImportError:
    from resource_binning import value_to_level
    from token_schema import generate_token_name, TOOL_ABBREV


def load_tools(tools_path: str) -> Dict[str, str]:
    """
    Load tool descriptions from tools.json.
    
    Args:
        tools_path: Path to tools.json
    
    Returns:
        Dictionary mapping tool names to descriptions
    """
    with open(tools_path, 'r', encoding='utf-8') as f:
        tools_list = json.load(f)
    
    return {tool['name']: tool['description'] for tool in tools_list}


def load_profiling(profiling_path: str) -> pd.DataFrame:
    """
    Load profiling data from CSV.
    
    Args:
        profiling_path: Path to profiling.csv
    
    Returns:
        DataFrame with profiling data
    """
    return pd.read_csv(profiling_path)


def generate_semantic_description(
    tool_name: str,
    input_size: str,
    cpu_core: int,
    cpu_mem_gb: float,
    gpu_sm: int,
    gpu_mem_gb: float
) -> str:
    """
    Generate human-readable semantic description for a token.
    
    Args:
        tool_name: Tool name
        input_size: Input size (small/medium/large)
        cpu_core: CPU core count
        cpu_mem_gb: CPU memory in GB
        gpu_sm: GPU SM count
        gpu_mem_gb: GPU memory in GB
    
    Returns:
        Semantic description string
    """
    tool_display = tool_name.replace('_', ' ').title()
    
    return (
        f"{tool_display} for {input_size} inputs with "
        f"{cpu_core} CPU cores, {int(cpu_mem_gb)}GB CPU memory, "
        f"{gpu_sm}% GPU SM units, and {int(gpu_mem_gb)}GB GPU memory"
    )


def build_registry(
    tools: Dict[str, str],
    profiling_df: pd.DataFrame
) -> Dict:
    """
    Build the complete tool registry.
    
    Args:
        tools: Dictionary of tool names to descriptions
        profiling_df: DataFrame with profiling data
    
    Returns:
        Complete registry dictionary
    """
    registry = {
        "tokens": {},
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_count": len(tools),
            "total_tokens": 0
        }
    }
    
    # Process each profiling record
    for idx, row in profiling_df.iterrows():
        tool_name = row['tool']
        
        # Skip if tool not in tools.json
        if tool_name not in tools:
            print(f"Warning: Tool '{tool_name}' not found in tools.json, skipping")
            continue
        
        # Map resource values to levels
        cpu_core_level = value_to_level('cpu_core', row['cpu_core'])
        cpu_mem_level = value_to_level('cpu_mem_gb', row['cpu_mem_gb'])
        gpu_sm_level = value_to_level('gpu_sm', row['gpu_sm'])
        gpu_mem_level = value_to_level('gpu_mem_gb', row['gpu_mem_gb'])
        
        input_size = row['input_size']
        
        # Generate token name
        token_name = generate_token_name(
            tool_name,
            input_size,
            cpu_core_level,
            cpu_mem_level,
            gpu_sm_level,
            gpu_mem_level
        )
        
        # Generate semantic description
        semantic_desc = generate_semantic_description(
            tool_name,
            input_size,
            int(row['cpu_core']),
            float(row['cpu_mem_gb']),
            int(row['gpu_sm']),
            float(row['gpu_mem_gb'])
        )
        
        # Create token entry
        registry["tokens"][token_name] = {
            "tool_name": tool_name,
            "description": tools[tool_name],
            "input_size": input_size,
            "resources": {
                "cpu_core": int(row['cpu_core']),
                "cpu_mem_gb": float(row['cpu_mem_gb']),
                "gpu_sm": int(row['gpu_sm']),
                "gpu_mem_gb": float(row['gpu_mem_gb'])
            },
            "resource_levels": {
                "cpu_core_level": cpu_core_level,
                "cpu_mem_level": cpu_mem_level,
                "gpu_sm_level": gpu_sm_level,
                "gpu_mem_level": gpu_mem_level
            },
            "latency_ms": float(row['latency_ms']),
            "semantic_description": semantic_desc
        }
    
    # Update metadata
    registry["metadata"]["total_tokens"] = len(registry["tokens"])
    
    return registry


def save_registry(registry: Dict, output_path: str):
    """
    Save registry to JSON file.
    
    Args:
        registry: Registry dictionary
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"Registry saved to: {output_path}")


def print_statistics(registry: Dict):
    """
    Print registry statistics.
    
    Args:
        registry: Registry dictionary
    """
    print("\n" + "="*50)
    print("Tool Registry Statistics")
    print("="*50)
    
    metadata = registry["metadata"]
    print(f"Total tokens: {metadata['total_tokens']}")
    print(f"Tool count: {metadata['tool_count']}")
    print(f"Created at: {metadata['created_at']}")
    
    # Count tokens per tool
    tokens_by_tool = {}
    for token_info in registry["tokens"].values():
        tool_name = token_info["tool_name"]
        tokens_by_tool[tool_name] = tokens_by_tool.get(tool_name, 0) + 1
    
    print("\nTokens per tool:")
    for tool_name, count in sorted(tokens_by_tool.items()):
        print(f"  {tool_name}: {count}")
    
    # Sample tokens
    print("\nSample tokens (first 5):")
    for i, (token_name, token_info) in enumerate(list(registry["tokens"].items())[:5]):
        print(f"  {i+1}. {token_name}")
        print(f"     Latency: {token_info['latency_ms']:.1f}ms")
        print(f"     Resources: CPU={token_info['resources']['cpu_core']}, "
              f"GPU_SM={token_info['resources']['gpu_sm']}")
    
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build tool registry from raw data"
    )
    parser.add_argument(
        '--tools',
        type=str,
        default='data/raw/tools.json',
        help='Path to tools.json'
    )
    parser.add_argument(
        '--profiling',
        type=str,
        default='data/raw/profiling.csv',
        help='Path to profiling.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/registry/tool_registry.json',
        help='Output path for registry JSON'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed statistics'
    )
    
    args = parser.parse_args()
    
    print("Building tool registry...")
    print(f"Tools file: {args.tools}")
    print(f"Profiling file: {args.profiling}")
    print(f"Output file: {args.output}")
    print()
    
    # Load data
    print("Loading tools...")
    tools = load_tools(args.tools)
    print(f"Loaded {len(tools)} tools")
    
    print("Loading profiling data...")
    profiling_df = load_profiling(args.profiling)
    print(f"Loaded {len(profiling_df)} profiling records")
    print()
    
    # Build registry
    print("Building registry...")
    registry = build_registry(tools, profiling_df)
    
    # Save registry
    save_registry(registry, args.output)
    
    # Print statistics
    if args.verbose or True:  # Always print stats
        print_statistics(registry)
    
    print("✓ Registry build completed successfully!")


if __name__ == "__main__":
    main()
