"""
Stage 1 Trainer: Tool Token Learning
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: Dict, load_lora_from: Optional[str] = None):
    """
    Load and configure model with LoRA and trainable embeddings.
    
    Args:
        config: Configuration dictionary
        load_lora_from: Path to existing LoRA checkpoint to load for stage-2 training (optional)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer'],
        trust_remote_code=True
    )
    
    print("Loading initialized model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['initialized_model'],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.get('bf16', True) else torch.float16,
        device_map='auto'
    )
    
    # Configure LoRA
    if config.get('use_lora', True):
        if load_lora_from:
            # Load existing LoRA checkpoint for stage-2 training
            print(f"Loading existing LoRA weights from {load_lora_from} for stage-2 training...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, load_lora_from)
            print("LoRA weights loaded successfully. Starting new training phase with these weights.")
        else:
            # Create new LoRA configuration
            print("Configuring new LoRA...")
            lora_config = LoraConfig(
                r=config.get('lora_r', 64),
                lora_alpha=config.get('lora_alpha', 32),
                target_modules=config.get('lora_target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj']),
                lora_dropout=config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
        
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    if config.get('gradient_checkpointing', False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # Set embedding layers trainable
    if config.get('train_embeddings', True):
        print("Setting embedding layers trainable...")
        
        # For PEFT models, access base model
        if hasattr(model, 'base_model'):
            base_model = model.base_model.model
        else:
            base_model = model
        
        # Set input embeddings trainable
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            for param in base_model.model.embed_tokens.parameters():
                param.requires_grad = True
        
        # Set output embeddings (lm_head) trainable
        if hasattr(base_model, 'lm_head'):
            for param in base_model.lm_head.parameters():
                param.requires_grad = True
    
    return model, tokenizer


def setup_training_args(config: Dict) -> TrainingArguments:
    """
    Setup TrainingArguments from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TrainingArguments instance
    """
    # Determine report_to based on wandb availability
    report_to = config.get('report_to', 'none')
    if report_to == 'wandb' and not WANDB_AVAILABLE:
        print("Warning: wandb not available, setting report_to='none'")
        report_to = 'none'
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        
        # Training hyperparameters
        num_train_epochs=int(config.get('num_epochs', 3)),
        per_device_train_batch_size=int(config.get('batch_size', 4)),
        gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 8)),
        learning_rate=float(config.get('learning_rate', 2e-5)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_ratio=float(config.get('warmup_ratio', 0.03)),
        max_grad_norm=float(config.get('max_grad_norm', 1.0)),
        
        # Optimizer
        optim=config.get('optim', 'adamw_torch'),
        adam_beta1=float(config.get('adam_beta1', 0.9)),
        adam_beta2=float(config.get('adam_beta2', 0.999)),
        adam_epsilon=float(config.get('adam_epsilon', 1e-8)),
        
        # Mixed precision
        fp16=config.get('fp16', False),
        bf16=config.get('bf16', True),
        
        # Logging
        logging_steps=int(config.get('logging_steps', 10)),
        logging_dir=os.path.join(config['output_dir'], 'logs'),
        report_to=report_to,
        
        # Checkpointing
        save_steps=int(config.get('save_steps', 500)),
        save_total_limit=int(config.get('save_total_limit', 3)),
        save_strategy='steps',
        
        # Other
        dataloader_num_workers=int(config.get('dataloader_num_workers', 4)),
        remove_unused_columns=bool(config.get('remove_unused_columns', False)),
        seed=int(config.get('seed', 42)),
        
        # Disable evaluation if no val data
        # evaluation_strategy='no',
        # load_best_model_at_end=False,
    )
    
    return training_args


def train_stage1(config_path: str, load_lora_from: Optional[str] = None):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
        load_lora_from: Path to existing LoRA checkpoint for stage-2 training (optional)
                       Note: This loads LoRA weights but starts a NEW training (not resume)
    """
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Initialize wandb if enabled
    if config.get('report_to') == 'wandb' and WANDB_AVAILABLE:
        wandb.init(
            project=config.get('wandb_project', 'tomas-llm'),
            name=config.get('wandb_run_name', 'stage1-tool-learning'),
            config=config
        )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config, load_lora_from=load_lora_from)
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Load dataset
    from src.datasets.tool_instruction_dataset import ToolInstructionDataset, ToolInstructionDataCollator
    
    print(f"Loading training data from {config['train_data']}...")
    train_dataset = ToolInstructionDataset(
        data_path=config['train_data'],
        tokenizer=tokenizer,
        max_length=config.get('max_length', 512)
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Data collator
    data_collator = ToolInstructionDataCollator(tokenizer=tokenizer)
    
    # Custom callback for evaluation
    from src.engine.evaluator import EvaluationCallback
    
    callbacks = [EvaluationCallback(tokenizer=tokenizer)]
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Train
    print("\n" + "=" * 60)
    if load_lora_from:
        print(f"Starting stage-2 training with LoRA weights from: {load_lora_from}")
        print("Note: This is a NEW training phase, not resuming from checkpoint")
    else:
        print("Starting stage-1 training with new LoRA initialization...")
    print("=" * 60)
    
    # Start new training (even if loading existing LoRA weights)
    # For true checkpoint resume, use Trainer's auto-detection in output_dir
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    output_dir = Path(config['output_dir']) / 'final_model'
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\nTraining complete! Model saved to {output_dir}")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


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
        help='Path to existing LoRA checkpoint for stage-2 training'
    )
    
    args = parser.parse_args()
    
    train_stage1(args.config, load_lora_from=args.load_lora_from)
