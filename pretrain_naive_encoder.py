#!/usr/bin/env python3
"""
Pretrain NaiveEncoderForPretraining

Main entry point for training the NaiveEncoderForPretraining model.
This script extends the LLM vocabulary with tool configuration tokens
and trains only the newly added embeddings and LM head extensions.

Usage:
    # First time training
    python pretrain_naive_encoder.py --config configs/pretrain_naive_encoder.yaml
    
    # Resume from latest checkpoint
    python pretrain_naive_encoder.py --config configs/pretrain_naive_encoder.yaml --resume latest
    
    # Resume from specific checkpoint
    python pretrain_naive_encoder.py --config configs/pretrain_naive_encoder.yaml --resume checkpoints/naive_encoder/checkpoint-1000

Features:
    - Lightweight checkpoints (only new parameters saved)
    - W&B logging integration
    - Mixed precision training
    - Automatic resume from checkpoint
    - HuggingFace Trainer API integration
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.v1.pipeline.train.naive_encoder_pretrainer import NaiveEncoderPretrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NaiveEncoderForPretraining with extended vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time training
  python pretrain_naive_encoder.py
  
  # Resume from latest checkpoint
  python pretrain_naive_encoder.py --resume latest
  
  # Use custom config
  python pretrain_naive_encoder.py --config configs/my_config.yaml
  
  # Disable W&B
  python pretrain_naive_encoder.py --no-wandb
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_naive_encoder.yaml",
        help="Path to configuration file (default: configs/pretrain_naive_encoder.yaml)",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (path or 'latest')",
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("NaiveEncoderForPretraining Training")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Resume from: {args.resume if args.resume else 'None (training from scratch)'}")
    logger.info(f"W&B logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    logger.info("=" * 80)
    
    # Verify config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize trainer
        logger.info("Initializing NaiveEncoderPretrainer...")
        trainer = NaiveEncoderPretrainer(config_path=args.config)
        
        # Override config if needed
        if args.no_wandb:
            trainer.config["wandb"]["enabled"] = False
            trainer.config["training"]["report_to"] = []
        
        if args.output_dir:
            trainer.config["training"]["output_dir"] = args.output_dir
            trainer.config["paths"]["checkpoint_dir"] = args.output_dir
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume)
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
