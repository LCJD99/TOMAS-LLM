from src.engine.stage1_trainer import train_stage1
from src.engine.stage1_trainer_v2 import train_stage1_v2

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
        '--use_v2',
        action='store_true',
        help='Use stage1_trainer_v2 with Dynamic Embeddings architecture (Profile Encoder + Dynamic LM Head)'
    )
    
    args = parser.parse_args()
    
    # 选择训练版本
    if args.use_v2:
        print("Using Stage 1 Trainer V2 (Dynamic Embeddings)")
        train_stage1_v2(args.config, load_lora_from=args.load_lora_from)
    else:
        print("Using Stage 1 Trainer V1 (Standard)")
        train_stage1(args.config, load_lora_from=args.load_lora_from)
