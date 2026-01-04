"""
Extend tokenizer with combined tokens.

This script:
1. Loads the base tokenizer (Qwen2.5-7B)
2. Adds combined tokens from combined_tokens.json
3. Saves the extended tokenizer
4. Validates the extension

Output: data/generated/extended_tokenizer/
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_combined_tokens(tokens_path: str) -> tuple[List[str], int]:
    """
    Load combined tokens from JSON file.
    
    Args:
        tokens_path: Path to combined_tokens.json
        
    Returns:
        Tuple of (token_list, num_tokens)
    """
    logger.info(f"Loading combined tokens from: {tokens_path}")
    
    with open(tokens_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tokens = data['tokens']
    num_tokens = data['num_new_tokens']
    
    logger.info(f"Loaded {num_tokens} combined tokens")
    
    return tokens, num_tokens


def extend_tokenizer(
    base_model_name: str,
    new_tokens: List[str],
    output_path: str,
    cache_dir: str = "hub"
) -> tuple[AutoTokenizer, int, int]:
    """
    Extend tokenizer with new tokens.
    
    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen2.5-7B")
        new_tokens: List of new tokens to add
        output_path: Path to save extended tokenizer
        cache_dir: Cache directory for downloaded models
        
    Returns:
        Tuple of (extended_tokenizer, original_vocab_size, new_vocab_size)
    """
    logger.info(f"Loading base tokenizer: {base_model_name}")
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocabulary size: {original_vocab_size:,}")
    
    # Add new tokens
    logger.info(f"Adding {len(new_tokens)} new tokens to vocabulary...")
    num_added = tokenizer.add_tokens(new_tokens)
    
    new_vocab_size = len(tokenizer)
    logger.info(f"Added {num_added} new tokens")
    logger.info(f"New vocabulary size: {new_vocab_size:,}")
    
    # Verify the extension
    if num_added != len(new_tokens):
        logger.warning(
            f"Expected to add {len(new_tokens)} tokens, "
            f"but only {num_added} were added. "
            "Some tokens may already exist in the vocabulary."
        )
    
    # Save extended tokenizer
    logger.info(f"Saving extended tokenizer to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"✓ Saved extended tokenizer")
    
    return tokenizer, original_vocab_size, new_vocab_size


def validate_tokenizer(
    tokenizer: AutoTokenizer,
    test_tokens: List[str],
    original_vocab_size: int
) -> bool:
    """
    Validate that new tokens can be encoded/decoded correctly.
    
    Args:
        tokenizer: Extended tokenizer
        test_tokens: List of tokens to test
        original_vocab_size: Original vocabulary size before extension
        
    Returns:
        True if validation passes
    """
    logger.info("\n" + "=" * 80)
    logger.info("Validating Extended Tokenizer")
    logger.info("=" * 80)
    
    success = True
    
    # Test a few sample tokens
    test_samples = test_tokens[:5]
    
    for i, token in enumerate(test_samples, 1):
        # Encode
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        
        # Check if it's a single token
        is_single_token = len(token_ids) == 1
        
        # Check if token_id is in the new range
        if len(token_ids) > 0:
            token_id = token_ids[0]
            is_new_token = token_id >= original_vocab_size
        else:
            is_new_token = False
            token_id = None
        
        logger.info(f"\nTest {i}: {token}")
        logger.info(f"  Encoded IDs: {token_ids}")
        logger.info(f"  Decoded: {decoded}")
        logger.info(f"  Single token: {is_single_token}")
        logger.info(f"  Token ID: {token_id}")
        logger.info(f"  In new range: {is_new_token}")
        
        # Validation checks
        if not is_single_token:
            logger.error(f"  ✗ Token was split into multiple IDs!")
            success = False
        elif decoded.strip() != token:
            logger.error(f"  ✗ Decoded token doesn't match original!")
            success = False
        elif not is_new_token:
            logger.warning(f"  ⚠ Token ID is not in the new range")
        else:
            logger.info(f"  ✓ Validation passed")
    
    # Test encoding a sample prompt with new tokens
    logger.info("\n" + "-" * 80)
    logger.info("Testing full prompt encoding:")
    
    sample_token = test_tokens[0]
    test_prompt = (
        f"This is a test prompt with combined token: {sample_token}"
    )
    
    encoded = tokenizer.encode(test_prompt, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)
    
    logger.info(f"  Prompt: {test_prompt}")
    logger.info(f"  Encoded length: {len(encoded)} tokens")
    logger.info(f"  Decoded: {decoded}")
    
    if sample_token in decoded:
        logger.info(f"  ✓ Combined token preserved in decoding")
    else:
        logger.error(f"  ✗ Combined token lost in decoding!")
        success = False
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ All validation checks passed")
    else:
        logger.error("✗ Some validation checks failed")
    logger.info("=" * 80)
    
    return success


def main():
    """Main entry point."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Define paths
    tokens_path = project_root / "data" / "generated" / "combined_tokens.json"
    output_path = project_root / "data" / "generated" / "extended_tokenizer"
    
    # Base model name
    base_model_name = "Qwen/Qwen2.5-7B"
    cache_dir = str(project_root / "hub")
    
    logger.info("=" * 80)
    logger.info("Tokenizer Extension")
    logger.info("=" * 80)
    
    # Check if tokens file exists
    if not tokens_path.exists():
        logger.error(f"Combined tokens file not found: {tokens_path}")
        logger.error("Please run token_generator.py first")
        return 1
    
    try:
        # Load combined tokens
        tokens, num_tokens = load_combined_tokens(str(tokens_path))
        
        # Extend tokenizer
        tokenizer, orig_size, new_size = extend_tokenizer(
            base_model_name,
            tokens,
            str(output_path),
            cache_dir
        )
        
        # Validate extension
        validation_passed = validate_tokenizer(tokenizer, tokens, orig_size)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Summary")
        logger.info("=" * 80)
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Original vocab size: {orig_size:,}")
        logger.info(f"New vocab size: {new_size:,}")
        logger.info(f"Added tokens: {new_size - orig_size:,}")
        logger.info(f"Expected tokens: {num_tokens:,}")
        logger.info(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        return 0 if validation_passed else 1
        
    except Exception as e:
        logger.error(f"Error extending tokenizer: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
