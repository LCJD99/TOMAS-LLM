#!/usr/bin/env python3
"""
Generate simulated profiling data for all tools in the registry.
Each tool will have 243 data points (3^5 combinations of 5 parameters with 3 levels each).
"""

import json
import csv
import os
from pathlib import Path
from itertools import product


# Base latencies at reference configuration ["medium", 8, 16, 60, 8]
# input_size, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb
REFERENCE_LATENCIES = {
    "super_resolution": 1.8,
    "image_captioning": 2.1,
    "visual_question_answering": 0.38,
    "image_classification": 0.67,
    "machine_translation": 1.67,  # warm up
    "text_summarization": 1.03,
    "object_detection": 0.42,
}

# Configuration levels for each parameter
# Maximum configuration is ["medium", 8, 16, 60, 8]
CONFIG_LEVELS = {
    "input_size": ["small", "medium", "large"],
    "cpu_core": [2, 4, 8],           # max: 8
    "cpu_mem_gb": [4.0, 8.0, 16.0],  # max: 16.0
    "gpu_sm": [20, 40, 60],          # max: 60
    "gpu_mem_gb": [2.0, 4.0, 8.0],   # max: 8.0
}

# Reference configuration indices (maximum resource configuration)
REFERENCE_CONFIG = {
    "input_size": 1,  # medium
    "cpu_core": 2,    # 8 (highest level)
    "cpu_mem_gb": 2,  # 16.0 (highest level)
    "gpu_sm": 2,      # 60 (highest level)
    "gpu_mem_gb": 2,  # 8.0 (highest level)
}


def calculate_latency(tool_name, input_size_idx, cpu_core_idx, cpu_mem_idx, gpu_sm_idx, gpu_mem_idx):
    """
    Calculate latency based on configuration.
    Uses a simple scaling model based on resource allocation.
    """
    base_latency = REFERENCE_LATENCIES.get(tool_name, 1.0)
    
    # Scaling factors for each parameter
    # Higher resources -> lower latency, larger input -> higher latency
    
    # Input size scaling (0: 0.5x, 1: 1.0x, 2: 2.5x)
    input_size_factors = [0.5, 1.0, 2.5]
    
    # CPU core scaling (inverse relationship)
    cpu_core_factors = [1.5, 1.0, 0.7]
    
    # CPU memory scaling (inverse relationship)
    cpu_mem_factors = [1.3, 1.0, 0.75]
    
    # GPU SM scaling (inverse relationship)
    gpu_sm_factors = [1.4, 1.0, 0.65]
    
    # GPU memory scaling (inverse relationship)
    gpu_mem_factors = [1.35, 1.0, 0.7]
    
    # Calculate final latency
    latency = base_latency
    latency *= input_size_factors[input_size_idx]
    latency *= cpu_core_factors[cpu_core_idx]
    latency *= cpu_mem_factors[cpu_mem_idx]
    latency *= gpu_sm_factors[gpu_sm_idx]
    latency *= gpu_mem_factors[gpu_mem_idx]
    
    # Add some variance to make it more realistic (±5%)
    import random
    variance = random.uniform(0.95, 1.05)
    latency *= variance
    
    return round(latency * 1000, 0)  # Convert to milliseconds


def generate_profiling_data():
    """Generate profiling data for all tools."""
    
    # Get paths
    script_dir = Path(__file__).parent.parent
    tools_json_path = script_dir / "data" / "tool_registry" / "tools.json"
    output_csv_path = script_dir / "data" / "profiling" / "profiling.csv"
    
    # Load tools
    with open(tools_json_path, 'r') as f:
        tools = json.load(f)
    
    # Prepare data rows
    rows = []
    
    for tool in tools:
        tool_name = tool["name"]
        print(f"Generating data for {tool_name}...")
        
        # Generate all combinations (3^5 = 243)
        for input_size_idx, cpu_core_idx, cpu_mem_idx, gpu_sm_idx, gpu_mem_idx in product(
            range(3), range(3), range(3), range(3), range(3)
        ):
            input_size = CONFIG_LEVELS["input_size"][input_size_idx]
            cpu_core = CONFIG_LEVELS["cpu_core"][cpu_core_idx]
            cpu_mem_gb = CONFIG_LEVELS["cpu_mem_gb"][cpu_mem_idx]
            gpu_sm = CONFIG_LEVELS["gpu_sm"][gpu_sm_idx]
            gpu_mem_gb = CONFIG_LEVELS["gpu_mem_gb"][gpu_mem_idx]
            
            latency_ms = calculate_latency(
                tool_name, input_size_idx, cpu_core_idx, 
                cpu_mem_idx, gpu_sm_idx, gpu_mem_idx
            )
            
            rows.append({
                "tool": tool_name,
                "input_size": input_size,
                "cpu_core": cpu_core,
                "cpu_mem_gb": cpu_mem_gb,
                "gpu_sm": gpu_sm,
                "gpu_mem_gb": gpu_mem_gb,
                "latency_ms": latency_ms
            })
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='') as f:
        fieldnames = ["tool", "input_size", "cpu_core", "cpu_mem_gb", "gpu_sm", "gpu_mem_gb", "latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nGenerated {len(rows)} rows of profiling data")
    print(f"Output saved to: {output_csv_path}")
    print(f"\nBreakdown:")
    print(f"  - {len(tools)} tools")
    print(f"  - 243 configurations per tool (3^5)")
    print(f"  - Total: {len(tools)} × 243 = {len(rows)} rows")


if __name__ == "__main__":
    # Set random seed for reproducibility
    import random
    random.seed(42)
    
    generate_profiling_data()
