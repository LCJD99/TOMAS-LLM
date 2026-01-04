"""
Extract combined tokens from generated training data.

This script reads the JSONL training data file and:
1. Extracts all unique combined tokens
2. Creates a mapping from tokens to configurations
3. Saves the token list to JSON

Output: data/generated/combined_tokens.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_tokens_from_jsonl(jsonl_path: str) -> tuple[List[str], Dict[str, dict]]:
    """
    Extract unique tokens and their configurations from JSONL file.
    
    Args:
        jsonl_path: Path to naive_pretrain_data.jsonl
        
    Returns:
        Tuple of (token_list, token_to_config_mapping)
    """
    logger.info(f"Reading training data from: {jsonl_path}")
    
    tokens_set: Set[str] = set()
    token_to_config: Dict[str, dict] = {}
    
    # Read JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Extract combined token
                token = entry['combined_token']
                tokens_set.add(token)
                
                # Store token-to-config mapping
                if token not in token_to_config:
                    token_to_config[token] = entry['config']
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: Invalid JSON - {e}")
                continue
            except KeyError as e:
                logger.warning(f"Skipping line {line_num}: Missing key - {e}")
                continue
    
    # Convert set to sorted list for consistency
    tokens_list = sorted(tokens_set)
    
    logger.info(f"Extracted {len(tokens_list)} unique tokens")
    
    return tokens_list, token_to_config


def save_token_data(
    tokens: List[str],
    token_to_config: Dict[str, dict],
    output_path: str
) -> None:
    """
    Save token data to JSON file.
    
    Args:
        tokens: List of unique tokens
        token_to_config: Mapping from tokens to configurations
        output_path: Path to output JSON file
    """
    logger.info(f"Saving token data to: {output_path}")
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output_data = {
        "tokens": tokens,
        "token_to_config": token_to_config,
        "num_new_tokens": len(tokens)
    }
    
    # Save to JSON with pretty formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(tokens)} tokens to {output_path}")


def display_statistics(tokens: List[str], token_to_config: Dict[str, dict]) -> None:
    """
    Display statistics about extracted tokens.
    
    Args:
        tokens: List of unique tokens
        token_to_config: Mapping from tokens to configurations
    """
    logger.info("\n" + "=" * 80)
    logger.info("Token Statistics")
    logger.info("=" * 80)
    
    # Count by tool type
    tool_counts = {}
    input_size_counts = {}
    
    for token, config in token_to_config.items():
        tool = config['tool']
        input_size = config['input_size']
        
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        input_size_counts[input_size] = input_size_counts.get(input_size, 0) + 1
    
    logger.info(f"Total unique tokens: {len(tokens)}")
    logger.info(f"\nTokens by tool type:")
    for tool, count in sorted(tool_counts.items()):
        logger.info(f"  - {tool}: {count}")
    
    logger.info(f"\nTokens by input size:")
    for size, count in sorted(input_size_counts.items()):
        logger.info(f"  - {size}: {count}")
    
    # Show sample tokens
    logger.info(f"\nSample tokens (first 5):")
    for i, token in enumerate(tokens[:5], 1):
        config = token_to_config[token]
        logger.info(f"  {i}. {token}")
        logger.info(f"     Tool: {config['tool']}, Size: {config['input_size']}, "
                   f"Latency: {config['latency_ms']}ms")


def main():
    """Main entry point."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Define paths
    jsonl_path = project_root / "data" / "generated" / "naive_pretrain_data.jsonl"
    output_path = project_root / "data" / "generated" / "combined_tokens.json"
    
    logger.info("=" * 80)
    logger.info("Combined Tokens Extractor")
    logger.info("=" * 80)
    
    # Check if JSONL file exists
    if not jsonl_path.exists():
        logger.error(f"Training data file not found: {jsonl_path}")
        logger.error("Please run naive_pretrain_data_generator.py first")
        return 1
    
    try:
        # Extract tokens from JSONL
        tokens, token_to_config = extract_tokens_from_jsonl(str(jsonl_path))
        
        # Save token data
        save_token_data(tokens, token_to_config, str(output_path))
        
        # Display statistics
        display_statistics(tokens, token_to_config)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"SUCCESS: Extracted {len(tokens)} unique tokens")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error extracting tokens: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
