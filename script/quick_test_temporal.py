"""
Quick Test Script for Temporal Encoder Pretraining.

Tests the training pipeline on a small dataset to verify correctness.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.pretrain_temporal_encoder import (
    set_seed, create_model, create_datasets, create_dataloaders,
    train_epoch, validate
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_quick_test_config(base_config_path: str, num_train: int, num_val: int, epochs: int):
    """Create a quick test configuration."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick testing
    config['data']['num_train_samples'] = num_train
    config['data']['num_val_samples'] = num_val
    config['data']['batch_size'] = min(8, num_train)
    config['data']['num_workers'] = 0  # Avoid multiprocessing issues
    config['training']['num_epochs'] = epochs
    config['training']['log_interval'] = 1
    config['training']['save_interval'] = 100
    config['training']['eval_interval'] = 50
    
    # Use smaller model for testing
    config['model']['llm_name'] = "Qwen2.5-7B-Instruct"
    config['model']['llm_embedding_dim'] = 3584  # Qwen2.5-7B embedding size
    
    # Simpler encoder
    config['model']['temporal_encoder']['hidden_channels'] = 32
    config['model']['temporal_encoder']['output_dim'] = 128
    config['model']['temporal_encoder']['num_layers'] = 2
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Quick Test for Temporal Encoder Training")
    parser.add_argument('--config', type=str, default='configs/pretrain_temporal.yaml',
                       help='Base config file')
    parser.add_argument('--num_train', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=20,
                       help='Number of validation samples')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("QUICK TEST MODE")
    logger.info(f"Train samples: {args.num_train}, Val samples: {args.num_val}, Epochs: {args.epochs}")
    logger.info("=" * 80)
    
    # Create test config
    config = create_quick_test_config(args.config, args.num_train, args.num_val, args.epochs)
    
    # Override device if provided
    if args.device:
        config['training']['device'] = args.device
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Step 1: Create model
    logger.info("\n[1/5] Creating model...")
    try:
        model, tokenizer = create_model(config, device)
        logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        raise
    
    # Step 2: Create datasets
    logger.info("\n[2/5] Creating datasets...")
    try:
        train_dataset, val_dataset = create_datasets(config, tokenizer)
        logger.info(f"✓ Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
    except Exception as e:
        logger.error(f"✗ Dataset creation failed: {e}")
        raise
    
    # Step 3: Create dataloaders
    logger.info("\n[3/5] Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
        logger.info(f"✓ Dataloaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    except Exception as e:
        logger.error(f"✗ Dataloader creation failed: {e}")
        raise
    
    # Step 4: Test forward pass
    logger.info("\n[4/5] Testing forward pass...")
    try:
        model.eval()
        batch = next(iter(train_loader))
        curve = batch['curve'].to(device)
        prompt_ids = batch['prompt_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            loss = model(curve, prompt_ids, target_ids, attention_mask)
        
        logger.info(f"✓ Forward pass successful, loss: {loss.item():.4f}")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        raise
    
    # Step 5: Test training loop
    logger.info("\n[5/5] Testing training loop...")
    try:
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Simple scheduler
        from transformers import get_constant_schedule
        scheduler = get_constant_schedule(optimizer)
        
        # Train for one epoch
        model.train()
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, 1, config, writer=None, global_step=0
        )
        
        logger.info(f"✓ Training epoch successful, avg loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_perplexity = validate(model, val_loader, device)
        logger.info(f"✓ Validation successful, loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")
        
    except Exception as e:
        logger.error(f"✗ Training loop failed: {e}")
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS PASSED! ✓")
    logger.info("=" * 80)
    logger.info("\nYou can now run full training with:")
    logger.info(f"  python script/pretrain_temporal_encoder.py --config {args.config}")


if __name__ == "__main__":
    main()
