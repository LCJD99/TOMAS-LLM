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
        '--resume_from',
        type=str,
        default=None,
        help='Path to LoRA checkpoint directory to resume training from'
    )
    
    args = parser.parse_args()
    train_stage1(args.config, resume_from=args.resume_from)
