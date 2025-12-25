"""
Encoder Pre-training Script for TOMAS-LLM.

Trains the resource encoder to compress 6D resource vectors into LLM-understandable
embeddings using self-supervised language modeling.

Training Flow:
1. Encoder generates embedding from (Tool ID + Resource Vector)
2. Embedding is injected as prefix to Qwen2.5
3. LLM generates natural language description
4. Loss is computed against ground truth text
5. Only Stream B + Fusion parameters are updated (Stream A frozen)

Goal: Overfit the encoder to memorize all 1701 configurations in semantic space.
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb

from src.data.pretrain_dataset import EncoderPretrainDataset
from src.offline.pretrain_encoder import ResourceEncoderForPretraining


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """
    Merge YAML config with command-line arguments.
    Command-line args take precedence over config file.
    """
    # Create a copy of config
    merged = config.copy()
    
    # Override with command-line arguments if provided
    args_dict = vars(args)
    
    # Map command-line args to config structure
    if args_dict.get('batch_size') is not None:
        merged['training']['batch_size'] = args_dict['batch_size']
    if args_dict.get('num_epochs') is not None:
        merged['training']['num_epochs'] = args_dict['num_epochs']
    if args_dict.get('lr') is not None:
        merged['training']['learning_rate'] = args_dict['lr']
    if args_dict.get('device') is not None:
        merged['device']['type'] = args_dict['device']
    if args_dict.get('output_dir') is not None:
        merged['output']['output_dir'] = args_dict['output_dir']
    if args_dict.get('log_wandb') is not None:
        merged['logging']['wandb']['enabled'] = args_dict['log_wandb']
    
    return merged


class EncoderPretrainer:
    """Manages the pre-training process for the resource encoder."""
    
    def __init__(
        self,
        encoder: ResourceEncoderForPretraining,
        llm_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        log_wandb: bool = False,
        project_name: str = "tomas-encoder-pretrain"
    ):
        """
        Initialize the pretrainer.
        
        Args:
            encoder: ResourceEncoderForPretraining instance
            llm_model: Frozen Qwen2.5 model for language modeling
            tokenizer: Tokenizer for the LLM
            optimizer: Optimizer for encoder parameters
            scheduler: Optional learning rate scheduler
            device: Device to train on
            log_wandb: Whether to log to Weights & Biases
            project_name: W&B project name
        """
        self.encoder = encoder.to(device)
        self.llm_model = llm_model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_wandb = log_wandb
        
        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Ensure encoder Stream A is frozen
        for param in self.encoder.semantic_embedding.parameters():
            param.requires_grad = False
        
        # Initialize W&B if requested
        if log_wandb:
            wandb.init(project=project_name, config={
                "model": "ResourceEncoderForPretraining",
                "llm_backbone": "Qwen2.5-7B",
                "trainable_params": sum(p.numel() for p in encoder.get_trainable_parameters())
            })
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Metrics tracking
        self.best_loss = float('inf')
        self.global_step = 0
    
    def inject_prefix_embeddings(
        self,
        encoder_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple:
        """
        Inject encoder embeddings as prefix to LLM input.
        
        Args:
            encoder_embeddings: [batch, hidden_dim] from encoder
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
        
        Returns:
            Tuple of (combined_embeddings, combined_attention_mask)
        """
        batch_size = encoder_embeddings.size(0)
        
        # Get token embeddings from LLM
        token_embeddings = self.llm_model.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden]
        
        # Expand encoder embeddings to [batch, 1, hidden]
        prefix_embeddings = encoder_embeddings.unsqueeze(1)  # [batch, 1, hidden]
        
        # Concatenate: [prefix, tokens]
        combined_embeddings = torch.cat([prefix_embeddings, token_embeddings], dim=1)  # [batch, 1+seq_len, hidden]
        
        # Update attention mask
        prefix_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [batch, 1+seq_len]
        
        return combined_embeddings, combined_attention_mask
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary with keys:
                - tool_id: [batch]
                - resource_vector: [batch, 6]
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - labels: [batch, seq_len]
        
        Returns:
            Dictionary with loss and metrics
        """
        self.encoder.train()
        self.llm_model.eval()
        
        # Move to device
        tool_ids = batch["tool_id"].to(self.device)
        resource_vectors = batch["resource_vector"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 1. Encoder: Generate embeddings
        encoder_embeddings = self.encoder(tool_ids, resource_vectors)  # [batch, hidden]
        
        # 2. Inject prefix: Combine encoder embedding with token embeddings
        combined_embeddings, combined_attention_mask = self.inject_prefix_embeddings(
            encoder_embeddings, input_ids, attention_mask
        )
        
        # 3. LLM forward pass with injected embeddings
        outputs = self.llm_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            return_dict=True
        )
        
        # 4. Calculate loss
        # Note: We need to shift logits/labels for causal LM
        # The prefix embedding is at position 0, tokens are at positions 1+
        logits = outputs.logits[:, :-1, :]  # [batch, prefix+seq_len-1, vocab]
        
        # Shift labels: we predict tokens 1 to end from positions 0 to end-1
        # For prefix position, we predict the first token
        # For token position i, we predict token i+1
        shifted_labels = labels.clone()  # [batch, seq_len]
        
        # Create shifted labels with prefix position predicting first token
        # Positions: [prefix->token0, token0->token1, ..., tokenN-1->tokenN]
        # We take logits[:, :-1] which is [prefix, token0, ..., tokenN-1]
        # And labels are already [token0, token1, ..., tokenN]
        # So we need to align: logits[i] predicts labels[i-1] for i>0, logits[0] predicts labels[0]
        
        # Actually, simpler approach:
        # logits from combined_embeddings will have shape [batch, 1+seq_len, vocab]
        # We want to predict labels (original tokens)
        # Position 0 (prefix) should predict labels[0]
        # Position i (token i-1) should predict labels[i]
        
        # So logits[:, :-1, :] gives predictions for positions 0 to seq_len-1
        # These should match labels[:, :] for positions 0 to seq_len-1
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len, vocab] (0 is prefix predicting first token)
        
        # Reshape for loss computation
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),  # [batch*seq_len, vocab]
            shifted_labels.reshape(-1)  # [batch*seq_len]
        )
        
        # 5. Backward pass (only encoder parameters)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.get_trainable_parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Track metrics
        self.global_step += 1
        
        metrics = {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "global_step": self.global_step
        }
        
        return metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the encoder on a dataset.
        
        Args:
            dataloader: Validation dataloader
        
        Returns:
            Dictionary with validation metrics
        """
        self.encoder.eval()
        self.llm_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                tool_ids = batch["tool_id"].to(self.device)
                resource_vectors = batch["resource_vector"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                encoder_embeddings = self.encoder(tool_ids, resource_vectors)
                combined_embeddings, combined_attention_mask = self.inject_prefix_embeddings(
                    encoder_embeddings, input_ids, attention_mask
                )
                
                outputs = self.llm_model(
                    inputs_embeds=combined_embeddings,
                    attention_mask=combined_attention_mask,
                    return_dict=True
                )
                
                logits = outputs.logits[:, :-1, :]
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {"val_loss": avg_loss}
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
        
        Returns:
            Dictionary with epoch metrics
        """
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            metrics = self.train_step(batch)
            
            epoch_loss += metrics["loss"]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "avg_loss": f"{epoch_loss/num_batches:.4f}",
                "lr": f"{metrics['lr']:.6f}"
            })
            
            # Log to W&B
            if self.log_wandb and self.global_step % 10 == 0:
                wandb.log(metrics, step=self.global_step)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        return {"train_loss": avg_loss, "epoch": epoch}
    
    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict):
        """Save encoder checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train Resource Encoder")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="configs/pretrain_encoder.yaml",
                        help="Path to YAML configuration file")
    
    # Optional overrides (take precedence over config file)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override number of epochs from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--log_wandb", action="store_true", default=None,
                        help="Override W&B logging from config")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Merge with command-line arguments
    config = merge_config_with_args(config, args)
    
    # Extract config sections for convenience
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    dataloader_cfg = config['dataloader']
    output_cfg = config['output']
    logging_cfg = config['logging']
    device_cfg = config['device']
    
    # Create output directory
    os.makedirs(output_cfg['output_dir'], exist_ok=True)
    
    print("="*80)
    print("TOMAS-LLM Encoder Pre-training")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Device: {device_cfg['type']}")
    print(f"LLM Model: {model_cfg['llm_model']}")
    print(f"Batch Size: {train_cfg['batch_size']}")
    print(f"Learning Rate: {train_cfg['learning_rate']}")
    print(f"Max Epochs: {train_cfg['num_epochs']}")
    print(f"Target Loss: {train_cfg['target_loss']}")
    print(f"Augmentation: {data_cfg['augmentation']['mode']} (×{data_cfg['augmentation']['num_copies']})")
    print("="*80)
    
    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = EncoderPretrainDataset(
        tool_registry_path=data_cfg['tool_registry'],
        profiling_data_path=data_cfg['profiling_data'],
        tokenizer_name=model_cfg['llm_model'],
        augmentation_mode=data_cfg['augmentation']['mode'],
        num_augmented_copies=data_cfg['augmentation']['num_copies'],
        use_variation=data_cfg['augmentation']['use_variation'],
        seed=data_cfg['augmentation']['seed']
    )
    
    stats = dataset.get_statistics()
    print(f"  ✓ Loaded {stats['total_samples']} samples from {stats['unique_configs']} configs")
    print(f"  ✓ Augmentation factor: {stats['augmentation_factor']}×")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=dataloader_cfg['shuffle'],
        num_workers=dataloader_cfg['num_workers'],
        pin_memory=dataloader_cfg['pin_memory'] if device_cfg['type'] == "cuda" else False
    )
    
    # 2. Initialize encoder
    print("\n[2/5] Initializing encoder...")
    encoder = ResourceEncoderForPretraining(
        llm_model_name=model_cfg['llm_model'],
        llm_hidden_dim=model_cfg['llm_hidden_dim'],
        d_resource=model_cfg['d_resource'],
        num_attention_heads=model_cfg['num_attention_heads'],
        dropout=model_cfg['dropout'],
        num_tools=model_cfg['num_tools'],
        freeze_semantic=model_cfg['freeze_semantic'],
        cache_dir=model_cfg['cache_dir']
    )
    
    trainable_params = sum(p.numel() for p in encoder.get_trainable_parameters())
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 3. Load LLM (frozen)
    print("\n[3/5] Loading LLM backbone...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_cfg['llm_model'],
        trust_remote_code=True,
        cache_dir=model_cfg['cache_dir'],
        torch_dtype=torch.float32  # Use FP32 for training
    )
    tokenizer = dataset.tokenizer
    print(f"  ✓ LLM loaded: {model_cfg['llm_model']}")
    
    # 4. Setup optimizer and scheduler
    print("\n[4/5] Setting up optimizer...")
    optimizer = AdamW(
        encoder.get_trainable_parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    
    scheduler_cfg = train_cfg['scheduler']
    if scheduler_cfg['type'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg['num_epochs'] * len(dataloader),
            eta_min=train_cfg['learning_rate'] * scheduler_cfg['eta_min_ratio']
        )
    else:
        scheduler = None
    
    print(f"  ✓ Optimizer: AdamW (lr={train_cfg['learning_rate']}, wd={train_cfg['weight_decay']})")
    print(f"  ✓ Scheduler: {scheduler_cfg['type']}")
    
    # 5. Initialize trainer
    print("\n[5/5] Initializing trainer...")
    trainer = EncoderPretrainer(
        encoder=encoder,
        llm_model=llm_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device_cfg['type'],
        log_wandb=logging_cfg['wandb']['enabled'],
        project_name=logging_cfg['wandb']['project']
    )
    print("  ✓ Trainer ready")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_loss = float('inf')
    
    for epoch in range(train_cfg['num_epochs']):
        # Train epoch
        epoch_metrics = trainer.train_epoch(dataloader, epoch)
        
        train_loss = epoch_metrics["train_loss"]
        print(f"\nEpoch {epoch+1}/{train_cfg['num_epochs']} - Train Loss: {train_loss:.4f}")
        
        # Log epoch metrics
        if logging_cfg['wandb']['enabled']:
            wandb.log(epoch_metrics, step=trainer.global_step)
        
        # Save checkpoint
        if (epoch + 1) % output_cfg['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(
                output_cfg['output_dir'],
                f"{output_cfg['checkpoint_prefix']}_epoch{epoch+1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
        
        # Check for best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(
                output_cfg['output_dir'], 
                output_cfg['best_model_name']
            )
            trainer.save_checkpoint(best_model_path, epoch, epoch_metrics)
            print(f"  ✓ New best model saved (loss: {best_loss:.4f})")
        
        # Early stopping
        if train_loss < train_cfg['target_loss']:
            print(f"\n🎉 Target loss {train_cfg['target_loss']} reached! (current: {train_loss:.4f})")
            print("Training complete - encoder has successfully memorized all configurations!")
            break
    
    # Save final model
    final_model_path = os.path.join(
        output_cfg['output_dir'], 
        output_cfg['final_model_name']
    )
    trainer.save_checkpoint(final_model_path, epoch, epoch_metrics)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Model: {final_model_path}")
    print(f"Best Model: {best_model_path}")
    
    if logging_cfg['wandb']['enabled']:
        wandb.finish()


if __name__ == "__main__":
    main()
