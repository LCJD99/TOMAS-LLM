"""
Temporal Encoder Pretraining Script with Distributed Data Parallel (DDP).

Train the Temporal Encoder to align resource timeline curves
with natural language descriptions using a frozen LLM on multiple GPUs.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 script/pretrain_temporal_encoder_ddp.py \
        --config configs/pretrain_temporal.yaml

    # Multi-node, 2 nodes, 4 GPUs each
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr="192.168.1.1" --master_port=29500 \
        script/pretrain_temporal_encoder_ddp.py --config configs/pretrain_temporal.yaml
    
    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr="192.168.1.1" --master_port=29500 \
        script/pretrain_temporal_encoder_ddp.py --config configs/pretrain_temporal.yaml
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import yaml

from src.context.temporal_encoder import TemporalEncoder
from src.context.temporal_llm_wrapper import TemporalLLMWrapper
from src.data.temporal_pretrain_dataset import TemporalPretrainDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_ddp():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.error("DDP environment variables not found. Use torchrun to launch.")
        raise RuntimeError("DDP environment not properly set up")
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    logger.info(f"[Rank {rank}/{world_size}] Initialized on {device}")
    
    return rank, world_size, local_rank, device


def cleanup_ddp():
    """Clean up distributed environment."""
    dist.destroy_process_group()


def set_seed(seed: int, rank: int):
    """Set random seeds for reproducibility (rank-specific)."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    if rank == 0:
        logger.info(f"Random seed set to {seed} (rank-adjusted)")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: torch.device, rank: int):
    """
    Create TemporalLLMWrapper model.
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        rank: Process rank
    
    Returns:
        model: TemporalLLMWrapper instance
        tokenizer: Tokenizer instance
    """
    model_config = config['model']
    
    # Load LLM and tokenizer (only log on rank 0)
    if rank == 0:
        logger.info(f"Loading LLM: {model_config['llm_name']}")
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True,
        torch_dtype=torch.float16 if config['training'].get('fp16', False) else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create TemporalEncoder
    if rank == 0:
        logger.info("Creating TemporalEncoder")
    
    temporal_config = model_config['temporal_encoder']
    cnn_config = {
        'in_channels': 4,
        'hidden_channels': temporal_config['hidden_channels'],
        'output_dim': temporal_config['output_dim'],
        'num_layers': temporal_config['num_layers'],
        'pooling': temporal_config['pooling']
    }
    
    temporal_encoder = TemporalEncoder(
        timeline=None,  # No timeline for synthetic data
        cnn_config=cnn_config,
        min_timesteps=temporal_config['min_timesteps'],
        max_timesteps=temporal_config['max_timesteps'],
        time_granularity_ms=temporal_config['time_granularity_ms'],
        llm_embedding_dim=model_config['llm_embedding_dim']
    )
    
    # Create wrapper
    if rank == 0:
        logger.info("Creating TemporalLLMWrapper")
    
    model = TemporalLLMWrapper(
        temporal_encoder=temporal_encoder,
        llm_model=llm_model,
        llm_embedding_dim=model_config['llm_embedding_dim'],
        freeze_llm=model_config['freeze_llm']
    )
    
    model = model.to(device)
    
    # Log parameter counts (only on rank 0)
    if rank == 0:
        total_params = model.count_total_parameters()
        trainable_params = model.count_trainable_parameters()
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer


def create_datasets(config: dict, tokenizer, rank: int):
    """Create train and validation datasets."""
    data_config = config['data']
    
    if rank == 0:
        logger.info(f"Creating datasets: {data_config['num_train_samples']} train, {data_config['num_val_samples']} val")
    
    train_dataset = TemporalPretrainDataset(
        num_samples=data_config['num_train_samples'],
        tokenizer=tokenizer,
        type_distribution=data_config['type_distribution'],
        max_length=data_config['max_length'],
        noise_level=data_config.get('noise_level', 0.05),
        spike_probability=data_config.get('spike_probability', 0.3),
        seed=config['training']['seed']
    )
    
    val_dataset = TemporalPretrainDataset(
        num_samples=data_config['num_val_samples'],
        tokenizer=tokenizer,
        type_distribution=data_config['type_distribution'],
        max_length=data_config['max_length'],
        noise_level=data_config.get('noise_level', 0.05),
        spike_probability=data_config.get('spike_probability', 0.3),
        seed=config['training']['seed'] + 1  # Different seed for val
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: dict, rank: int, world_size: int):
    """Create data loaders with DistributedSampler."""
    data_config = config['data']
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config['training']['seed']
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config['training']['seed']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        sampler=train_sampler,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn,
        drop_last=True  # Important for DDP
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        sampler=val_sampler,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def train_epoch(
    model,
    dataloader,
    sampler,
    optimizer,
    scheduler,
    device,
    epoch,
    config,
    rank,
    writer=None,
    global_step=0
):
    """Train for one epoch."""
    model.train()
    
    # Set epoch for sampler (important for shuffling)
    sampler.set_epoch(epoch)
    
    total_loss = 0
    num_batches = 0
    
    train_config = config['training']
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        curve = batch['curve'].to(device)
        prompt_ids = batch['prompt_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        loss = model(curve, prompt_ids, target_ids, attention_mask)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if train_config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.module.get_trainable_parameters(),  # Use model.module for DDP
                    train_config['gradient_clip']
                )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Logging (only on rank 0)
            if rank == 0 and global_step % train_config.get('log_interval', 10) == 0:
                current_lr = scheduler.get_last_lr()[0]
                if writer is not None:
                    writer.add_scalar('train/loss', loss.item() * gradient_accumulation_steps, global_step)
                    writer.add_scalar('train/lr', current_lr, global_step)
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar (only on rank 0)
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    
    # Synchronize loss across all processes
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss = avg_loss_tensor.item()
    
    return avg_loss, global_step


def validate(model, dataloader, device, rank):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validation")
        else:
            pbar = dataloader
            
        for batch in pbar:
            curve = batch['curve'].to(device)
            prompt_ids = batch['prompt_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            loss = model(curve, prompt_ids, target_ids, attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Synchronize loss across all processes
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss = avg_loss_tensor.item()
    
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    train_loss,
    val_loss,
    save_path,
    rank
):
    """Save training checkpoint (only on rank 0)."""
    if rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict(),  # Use model.module for DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Temporal Encoder Pretraining with DDP")
    parser.add_argument('--config', type=str, default='configs/pretrain_temporal.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank, device = setup_ddp()
    
    # Load config
    config = load_config(args.config)
    
    # Override output dir if provided
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Create output directories (only on rank 0)
    if rank == 0:
        output_dir = Path(config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir = Path(config['paths']['tensorboard_dir'])
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Barrier to ensure directories are created before other ranks proceed
    dist.barrier()
    
    output_dir = Path(config['paths']['output_dir'])
    log_dir = Path(config['paths']['log_dir'])
    tensorboard_dir = Path(config['paths']['tensorboard_dir'])
    
    # Set seed
    set_seed(config['training']['seed'], rank)
    
    if rank == 0:
        logger.info(f"Training on {world_size} GPUs")
        logger.info(f"Using device: {device}")
    
    # Create model and tokenizer
    model, tokenizer = create_model(config, device, rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer, rank)
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        train_dataset, val_dataset, config, rank, world_size
    )
    
    # Create optimizer
    optimizer_config = config['optimizer']
    optimizer = torch.optim.AdamW(
        model.module.get_trainable_parameters(),  # Use model.module for DDP
        lr=config['training']['learning_rate'],
        betas=optimizer_config['betas'],
        eps=optimizer_config['eps'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = config['training'].get('warmup_steps', 0)
    if num_warmup_steps == 0 and 'warmup_ratio' in config['training']:
        num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
    
    scheduler = get_scheduler(
        name=config['scheduler']['type'],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        
        # Load checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    if rank == 0:
        logger.info("Starting training")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        if rank == 0:
            logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, train_sampler, optimizer, scheduler,
            device, epoch, config, rank, writer, global_step
        )
        
        if rank == 0:
            logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_perplexity = validate(model, val_loader, device, rank)
        
        if rank == 0:
            logger.info(f"Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            
            # Log to tensorboard
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/val_perplexity', val_perplexity, epoch)
        
        # Save checkpoint after each epoch (only on rank 0)
        checkpoint_path = output_dir / f"epoch_{epoch + 1}_step_{global_step}.pt"
        save_checkpoint(
            model, optimizer, scheduler,
            epoch, global_step,
            train_loss, val_loss,
            checkpoint_path,
            rank
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, global_step,
                train_loss, val_loss,
                best_checkpoint_path,
                rank
            )
            if rank == 0:
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
        
        # Barrier to ensure checkpoint is saved before next epoch
        dist.barrier()
    
    if rank == 0:
        writer.close()
        logger.info("Training complete!")
    
    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()
