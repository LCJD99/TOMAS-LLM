from src.engine.stage1_trainer import train_stage1
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stage 1 Model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--load_lora_from',
        type=str,
        default=None,
        help='Path to existing LoRA checkpoint directory to load for stage-2 training (not resume, starts new training with loaded LoRA weights)'
    )
    
    args = parser.parse_args()
    train_stage1(args.config, load_lora_from=args.load_lora_from)
