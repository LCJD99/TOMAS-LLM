"""
Tool Encoder Pre-training with transformers.Trainer

This module refactors the encoder pre-training to use HuggingFace Trainer API.
"""

import argparse
import os
import yaml
from datetime import datetime
from typing import Dict, Optional, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import wandb

from src.v1.data.tool_encoder_pretrain_dataset import EncoderPretrainDataset
from src.v1.model.tool_encoder import ResourceEncoderForPretraining
from src.v1.utils.config import load_config, merge_config_with_args


class ToolEncoderTrainer(Trainer):
    """
    Custom Trainer for Tool Encoder pre-training.
    
    Extends HuggingFace Trainer to handle:
    1. Encoder embeddings injection between prefix and suffix tokens
    2. Frozen LLM backbone with trainable encoder
    3. Custom loss computation with label masking
    """
    
    def __init__(
        self,
        encoder: ResourceEncoderForPretraining,
        llm_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        """
        Initialize custom trainer.
        
        Args:
            encoder: The trainable resource encoder
            llm_model: Frozen LLM backbone (for generating embeddings)
            tokenizer: Tokenizer for loss computation
            **kwargs: Arguments passed to base Trainer
        """
        # Store encoder and LLM separately
        self.encoder = encoder
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        
        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Move models to device from args
        device = kwargs.get('args').device
        self.encoder = self.encoder.to(device)
        self.llm_model = self.llm_model.to(device)
        
        # Initialize base Trainer with encoder as the main model
        # (This allows Trainer to handle optimizer, scheduler, checkpointing, etc.)
        super().__init__(model=encoder, tokenizer=tokenizer, **kwargs)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def inject_encoder_embedding(
        self, 
        encoder_embeddings: torch.Tensor, 
        prefix_input_ids: torch.Tensor, 
        prefix_attention_mask: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        suffix_attention_mask: torch.Tensor
    ):
        """
        Inject encoder embeddings between prefix and suffix token sequences.
        
        Structure: [prefix_tokens] + [encoder_embedding] + [suffix_tokens]
        
        Args:
            encoder_embeddings: [B, D] - Encoder output embeddings
            prefix_input_ids: [B, L_pre] - Tokens before <tool_feat>
            prefix_attention_mask: [B, L_pre]
            suffix_input_ids: [B, L_suf] - Tokens after <tool_feat>
            suffix_attention_mask: [B, L_suf]
        
        Returns:
            combined_embeddings: [B, L_pre + 1 + L_suf, D]
            combined_attention_mask: [B, L_pre + 1 + L_suf]
        """
        batch_size = encoder_embeddings.size(0)
        
        # Get LLM embedding layer
        try:
            embed_layer = self.llm_model.get_input_embeddings()
        except AttributeError:
            embed_layer = self.llm_model.module.get_input_embeddings()
        
        # Convert token IDs to embeddings
        prefix_token_embeddings = embed_layer(prefix_input_ids)  # [B, L_pre, D]
        suffix_token_embeddings = embed_layer(suffix_input_ids)  # [B, L_suf, D]
        
        # Expand encoder embeddings to [B, 1, D]
        encoder_embeddings_expanded = encoder_embeddings.unsqueeze(1)  # [B, 1, D]
        
        # Concatenate: prefix + encoder + suffix
        combined_embeddings = torch.cat([
            prefix_token_embeddings,
            encoder_embeddings_expanded,
            suffix_token_embeddings
        ], dim=1)  # [B, L_pre + 1 + L_suf, D]
        
        # Create combined attention mask
        encoder_mask = torch.ones(
            batch_size, 1, 
            dtype=prefix_attention_mask.dtype, 
            device=prefix_attention_mask.device
        )
        combined_attention_mask = torch.cat([
            prefix_attention_mask,
            encoder_mask,
            suffix_attention_mask
        ], dim=1)  # [B, L_pre + 1 + L_suf]
        
        return combined_embeddings, combined_attention_mask
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for encoder pre-training.
        
        Args:
            model: The encoder model (automatically passed by Trainer)
            inputs: Batch dictionary from dataset
            return_outputs: Whether to return model outputs
            
        Returns:
            loss or (loss, outputs) tuple
        """
        # Extract inputs
        tool_ids = inputs["tool_id"]
        resource_vectors = inputs["resource_vector"]
        prefix_input_ids = inputs["prefix_input_ids"]
        prefix_attention_mask = inputs["prefix_attention_mask"]
        suffix_input_ids = inputs["suffix_input_ids"]
        suffix_attention_mask = inputs["suffix_attention_mask"]
        prefix_labels = inputs["prefix_labels"]
        suffix_labels = inputs["suffix_labels"]
        
        # Get encoder embeddings (model is self.encoder)
        encoder_embeddings = model(tool_ids, resource_vectors)
        
        # Ensure encoder embeddings match LLM dtype
        llm_dtype = next(self.llm_model.parameters()).dtype
        encoder_embeddings = encoder_embeddings.to(llm_dtype)
        
        # Inject encoder embedding between prefix and suffix
        combined_embeddings, combined_attention_mask = self.inject_encoder_embedding(
            encoder_embeddings,
            prefix_input_ids,
            prefix_attention_mask,
            suffix_input_ids,
            suffix_attention_mask
        )
        
        # Forward through LLM (frozen)
        with torch.no_grad():
            outputs = self.llm_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                return_dict=True
            )
        
        # Combine labels: [prefix_labels] + [-100 for encoder] + [suffix_labels]
        batch_size = tool_ids.size(0)
        encoder_label = torch.full(
            (batch_size, 1), -100, 
            dtype=prefix_labels.dtype, 
            device=prefix_labels.device
        )
        combined_labels = torch.cat([
            prefix_labels,
            encoder_label,
            suffix_labels
        ], dim=1)
        
        # Shift logits and labels for causal LM
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = combined_labels[:, 1:].contiguous()
        
        # Compute loss
        loss = self.criterion(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        
        return (loss, outputs) if return_outputs else loss


class GateAlphaCallback(TrainerCallback):
    """Callback to log gate_alpha value during training."""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log gate_alpha value to wandb/tensorboard."""
        if model is not None and hasattr(model, 'get_gate_value'):
            gate_alpha = model.get_gate_value()
            logs['gate_alpha'] = gate_alpha



def train_tool_encoder(args: argparse.Namespace):
    """
    Main training function using transformers.Trainer.
    """
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # Get configurations
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    output_cfg = config['output']
    logging_cfg = config['logging']
    
    # Setup output directory
    os.makedirs(output_cfg['output_dir'], exist_ok=True)

    # ========================================
    # 1. Load Dataset
    # ========================================
    print("\n[1/4] Loading dataset...")
    
    torch.manual_seed(data_cfg['augmentation']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data_cfg['augmentation']['seed'])

    dataset = EncoderPretrainDataset(
        tool_registry_path=data_cfg['tool_registry'],
        profiling_data_path=data_cfg['profiling_data'],
        tokenizer_name=model_cfg['llm_model'],
        augmentation_mode=data_cfg['augmentation']['mode'],
        num_augmented_copies=data_cfg['augmentation']['num_copies'],
        use_variation=data_cfg['augmentation']['use_variation'],
        seed=data_cfg['augmentation']['seed']
    )
    
    print(f"  ✓ Dataset size: {len(dataset)} samples")
    
    # ========================================
    # 2. Initialize Models
    # ========================================
    print("\n[2/4] Initializing models...")
    
    # Initialize encoder
    encoder = ResourceEncoderForPretraining(
        llm_model_name=model_cfg['llm_model'],
        tool_registry_path=data_cfg['tool_registry'],
        d_resource=model_cfg.get('d_resource'),
        num_attention_heads=model_cfg['num_attention_heads'],
        dropout=model_cfg['dropout'],
        cache_dir=model_cfg.get('cache_dir', 'hub')
    )
    
    print(f"  ✓ Encoder loaded: {encoder.num_tools} tools")
    print(f"  ✓ Hidden dimension: {encoder.llm_hidden_dim}")
    print(f"  ✓ Gate alpha: {encoder.get_gate_value():.6f}")
    print(f"  ✓ Trainable params: {sum(p.numel() for p in encoder.get_trainable_parameters()):,}")
    
    # Load LLM backbone
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_cfg['llm_model'],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    print(f"  ✓ LLM backbone loaded (frozen)")
    
    # ========================================
    # 3. Setup Training Arguments
    # ========================================
    print("\n[3/4] Configuring training arguments...")
    
    # Get device preference
    device = train_cfg.get('device', args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    training_args = TrainingArguments(
        output_dir=output_cfg['output_dir'],
        
        # Training hyperparameters
        num_train_epochs=train_cfg['num_epochs'],
        per_device_train_batch_size=train_cfg['batch_size'],
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        
        # Optimization
        lr_scheduler_type=train_cfg['scheduler']['type'],
        warmup_ratio=train_cfg['scheduler'].get('warmup_ratio', 0.0),
        max_grad_norm=1.0,
        
        # Mixed precision
        bf16=True if device == 'cuda' and torch.cuda.is_available() else False,
        
        # Logging
        logging_dir=os.path.join(output_cfg['output_dir'], 'logs'),
        logging_steps=10,
        logging_strategy='steps',
        
        # Checkpointing
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=False,
        
        # Evaluation (disabled for now)
        evaluation_strategy='no',
        
        # Other settings
        dataloader_num_workers=config['dataloader']['num_workers'],
        remove_unused_columns=False,  # Important: keep all dataset columns
        report_to='wandb' if logging_cfg['wandb']['enabled'] else 'none',
        run_name=f"encoder_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        
        # Device
        no_cuda=(device == 'cpu'),
    )
    
    # ========================================
    # 4. Initialize Trainer
    # ========================================
    print("\n[4/4] Initializing trainer...")
    
    # Initialize W&B if enabled
    if logging_cfg['wandb']['enabled']:
        wandb.init(
            project=logging_cfg['wandb']['project'],
            config={
                "model": "ResourceEncoderForPretraining",
                "architecture": "Deep Semantic + Gated Fusion",
                "llm_backbone": model_cfg['llm_model'],
                "num_tools": encoder.num_tools,
                "hidden_dim": encoder.llm_hidden_dim,
                "gate_alpha_init": encoder.get_gate_value(),
                "trainable_params": sum(p.numel() for p in encoder.get_trainable_parameters()),
                "total_params": sum(p.numel() for p in encoder.parameters()),
                **train_cfg
            }
        )
    
    trainer = ToolEncoderTrainer(
        encoder=encoder,
        llm_model=llm_model,
        tokenizer=dataset.tokenizer,
        args=training_args,
        train_dataset=dataset,
        callbacks=[GateAlphaCallback()],
    )
    
    # ========================================
    # 5. Training
    # ========================================
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    # Resume from checkpoint if specified
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            resume_from_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Warning: Checkpoint {args.resume_from_checkpoint} not found")
    
    # Train!
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # ========================================
    # 6. Save Final Model
    # ========================================
    print("\n" + "="*60)
    print("Training Complete - Saving final model")
    print("="*60)
    
    final_model_path = os.path.join(output_cfg['output_dir'], 'final_model')
    trainer.save_model(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Cleanup W&B
    if logging_cfg['wandb']['enabled']:
        wandb.finish()
    
    print("\n✓ Training completed successfully!")

