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
    
    args = parser.parse_args()
    train_stage1(args.config)
