from src.engine.stage1_trainer import train_stage1
from src.engine.stage1_trainer_v2 import train_stage1_v2
from src.engine.stage1_trainer_v3 import train_stage1_v3

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
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        choices=['v1', 'v2', 'v3'],
        help='Training version: v1=Standard, v2=Dynamic Embeddings, v3=Virtual Tokens (recommended)'
    )
    
    args = parser.parse_args()
    
    # Select training version
    if args.version == 'v3':
        print("Using Stage 1 Trainer V3 (Virtual Tokens - No Embedding Expansion)")
        train_stage1_v3(args.config, load_lora_from=args.load_lora_from)
    elif args.version == 'v2':
        print("Using Stage 1 Trainer V2 (Dynamic Embeddings)")
        train_stage1_v2(args.config, load_lora_from=args.load_lora_from)
    else:
        print("Using Stage 1 Trainer V1 (Standard)")
        train_stage1(args.config, load_lora_from=args.load_lora_from)
