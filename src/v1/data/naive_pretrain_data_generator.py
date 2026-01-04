"""
Generate training data from profiling.csv for naive encoder pretraining.

This script reads profiling.csv and generates:
1. Combined tokens for each configuration
2. Natural language descriptions (user prompts)
3. JSONL training data file

Output: data/generated/naive_pretrain_data.jsonl
"""

import os
import json
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_combined_token(row: pd.Series) -> str:
    """
    Generate a combined token identifier from a configuration row.
    
    Format: <TOOL_{tool}_{input_size}_{cpu_core}_{cpu_mem}_{gpu_sm}_{gpu_mem}>
    Example: <TOOL_image_classification_small_2_4_20_2>
    
    Args:
        row: DataFrame row containing configuration
        
    Returns:
        Combined token string
    """
    # Convert float values to int for cleaner token names
    cpu_mem = int(row['cpu_mem_gb'])
    gpu_mem = int(row['gpu_mem_gb'])
    
    token = (
        f"<TOOL_{row['tool']}_{row['input_size']}_"
        f"{row['cpu_core']}_{cpu_mem}_{row['gpu_sm']}_{gpu_mem}>"
    )
    
    return token


def generate_user_prompt(row: pd.Series) -> str:
    """
    Generate natural language description for a configuration.
    
    This description serves as the user prompt in training,
    teaching the model to map descriptions to combined tokens.
    
    Args:
        row: DataFrame row containing configuration
        
    Returns:
        Natural language description string
    """
    # Convert float values to int for cleaner descriptions
    cpu_mem = int(row['cpu_mem_gb'])
    gpu_mem = int(row['gpu_mem_gb'])
    latency = int(row['latency_ms'])
    
    prompt = (
        f"This is the {row['tool']} tool with {row['input_size']} input size. "
        f"It uses {row['cpu_core']} CPU cores, {cpu_mem} GB CPU memory, "
        f"{row['gpu_sm']}% GPU SMs, and {gpu_mem} GB GPU memory. "
        f"The expected latency is {latency} ms."
    )
    
    return prompt


def generate_training_data(csv_path: str, output_path: str) -> int:
    """
    Generate training data from profiling CSV.
    
    Args:
        csv_path: Path to profiling.csv
        output_path: Path to output JSONL file
        
    Returns:
        Number of generated training examples
    """
    logger.info(f"Reading CSV from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} configurations from CSV")
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    training_data = []
    
    for idx, row in df.iterrows():
        # Generate combined token
        combined_token = generate_combined_token(row)
        
        # Generate user prompt (natural language description)
        user_prompt = generate_user_prompt(row)
        
        # Response is the combined token
        response = combined_token
        
        # Full prompt is user_prompt + response
        full_prompt = f"{user_prompt} {response}"
        
        # Create training entry
        entry = {
            "combined_token": combined_token,
            "user_prompt": user_prompt,
            "response": response,
            "full_prompt": full_prompt,
            "config": {
                "tool": row['tool'],
                "input_size": row['input_size'],
                "cpu_core": int(row['cpu_core']),
                "cpu_mem_gb": int(row['cpu_mem_gb']),
                "gpu_sm": int(row['gpu_sm']),
                "gpu_mem_gb": int(row['gpu_mem_gb']),
                "latency_ms": int(row['latency_ms'])
            }
        }
        
        training_data.append(entry)
    
    # Save as JSONL (one JSON object per line)
    logger.info(f"Saving {len(training_data)} training examples to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Successfully generated {len(training_data)} training examples")
    
    # Print some statistics
    unique_tools = df['tool'].nunique()
    unique_tokens = len(set(entry['combined_token'] for entry in training_data))
    
    logger.info(f"Statistics:")
    logger.info(f"  - Unique tools: {unique_tools}")
    logger.info(f"  - Unique combined tokens: {unique_tokens}")
    logger.info(f"  - Total training examples: {len(training_data)}")
    
    # Show a sample entry
    logger.info(f"\nSample entry:")
    sample = training_data[0]
    logger.info(f"  Combined token: {sample['combined_token']}")
    logger.info(f"  User prompt: {sample['user_prompt'][:80]}...")
    logger.info(f"  Response: {sample['response']}")
    
    return len(training_data)


def main():
    """Main entry point."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Define paths
    csv_path = project_root / "data" / "profiling" / "profiling.csv"
    output_path = project_root / "data" / "generated" / "naive_pretrain_data.jsonl"
    
    logger.info("=" * 80)
    logger.info("Naive Encoder Training Data Generator")
    logger.info("=" * 80)
    
    # Check if CSV exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please ensure profiling.csv exists in data/profiling/")
        return 1
    
    # Generate training data
    try:
        num_examples = generate_training_data(str(csv_path), str(output_path))
        logger.info(f"\n{'=' * 80}")
        logger.info(f"SUCCESS: Generated {num_examples} training examples")
        logger.info(f"Output: {output_path}")
        logger.info(f"{'=' * 80}")
        return 0
    
    except Exception as e:
        logger.error(f"Error generating training data: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
