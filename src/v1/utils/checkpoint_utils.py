"""
Checkpoint Management Utilities

Provides utilities for saving and loading lightweight checkpoints
that only contain the newly added parameters (embeddings and LM head extensions).
"""

import os
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_new_parameters(
    model: nn.Module,
    save_path: str,
    original_vocab_size: int,
    num_new_tokens: int,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save only the new parameters (new token embeddings and lm_head).
    This creates a lightweight checkpoint.
    
    Args:
        model: The NaiveEncoderForPretraining model
        save_path: Path to save the new parameters
        original_vocab_size: Original vocabulary size before extension
        num_new_tokens: Number of newly added tokens
        metadata: Optional metadata to save with the checkpoint
    """
    logger.info(f"Saving new parameters to: {save_path}")
    
    # Save new token modules
    checkpoint = {
        "new_token_embeddings": model.new_token_embeddings.state_dict(),
        "new_token_lm_head": model.new_token_lm_head.state_dict(),
        "num_new_tokens": num_new_tokens,
        "original_vocab_size": original_vocab_size,
        "hidden_dim": model.hidden_dim,
    }
    
    # Add metadata
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save
    torch.save(checkpoint, save_path)
    
    # Log file size
    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    logger.info(f"Saved new parameters: {file_size_mb:.2f} MB")


def load_new_parameters(
    model: nn.Module,
    load_path: str,
    device: str = "cpu",
) -> Dict:
    """
    Load previously saved new parameters.
    
    Args:
        model: The NaiveEncoderForPretraining model
        load_path: Path to the saved new parameters
        device: Device to load the parameters to
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    logger.info(f"Loading new parameters from: {load_path}")
    
    # Load checkpoint
    checkpoint = torch.load(load_path, map_location=device)
    
    # Verify compatibility
    original_vocab_size = checkpoint["original_vocab_size"]
    num_new_tokens = checkpoint["num_new_tokens"]
    
    # Load module states
    model.new_token_embeddings.load_state_dict(checkpoint["new_token_embeddings"])
    model.new_token_lm_head.load_state_dict(checkpoint["new_token_lm_head"])
    
    logger.info("Successfully loaded new parameters")
    
    # Return metadata
    metadata = checkpoint.get("metadata", {})
    return metadata


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint-* directories (Trainer format)
    checkpoint_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    
    if not checkpoint_dirs:
        return None
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_dirs[-1])
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint


def get_new_params_path(checkpoint_dir: str, step: Optional[int] = None) -> str:
    """
    Get the path to save/load new parameters.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        step: Training step number (None for final checkpoint)
    
    Returns:
        Path to the new_params.pt file
    """
    if step is None:
        return os.path.join(checkpoint_dir, "new_params_final.pt")
    else:
        return os.path.join(checkpoint_dir, f"checkpoint-{step}", "new_params.pt")
