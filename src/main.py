"""
Main entry point for TOMAS-LLM training and evaluation.

Usage:
    python src/main.py --config configs/stage1_tool_learning.yaml --stage 1
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TOMAS-LLM Training")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training stage: 1 (tool learning) or 2 (planning)"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config: dict, args):
    """Setup training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.get('seed', 42))
    
    # Create output directory
    output_dir = args.output_dir or config.get('output_dir', 'checkpoints/')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup distributed training if needed
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    return output_dir


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    output_dir = setup_environment(config, args)
    
    print(f"Starting TOMAS-LLM Stage {args.stage} Training")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    
    if args.stage == 1:
        # Import and run stage 1 trainer
        # from engine.stage1_trainer import Stage1Trainer
        # trainer = Stage1Trainer(config)
        # trainer.train()
        print("Stage 1 trainer not yet implemented. See TODO.md Phase 3.")
    
    elif args.stage == 2:
        # Import and run stage 2 trainer
        # from engine.stage2_trainer import Stage2Trainer
        # trainer = Stage2Trainer(config)
        # trainer.train()
        print("Stage 2 trainer not yet implemented. See TODO.md Phase 4.")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
