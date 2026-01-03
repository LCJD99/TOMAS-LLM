"""
Tool Encoder Pre-training 

"""

import argparse
import os
import yaml
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
from torch.profiler import profile, ProfilerActivity, schedule

from src.v1.data.tool_encoder_pretrain_dataset import EncoderPretrainDataset
from src.v1.model.tool_encoder import ResourceEncoderForPretraining
from src.v1.utils.config import load_config, merge_config_with_args

class ToolEncoderPretrainer:
    def __init__(
        self,
        encoder: ResourceEncoderForPretraining,
        llm_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        log_wandb: bool = False,
        project_name: str = "tomas-encoder-pretrain",
        enable_profiler: bool = False,
        profiler_output_dir: str = "./profiler_outputs"
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.log_wandb = log_wandb
        
        # Move models to device
        self.encoder = encoder.to(device)
        self.llm_model = llm_model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Initialize W&B
        if log_wandb:
            wandb.init(project=project_name, config={
                "model": "ResourceEncoderForPretraining",
                "architecture": "Deep Semantic + Gated Fusion",
                "llm_backbone": "Qwen2.5-7B",
                "device": device,
                "num_tools": self.encoder.num_tools,
                "hidden_dim": self.encoder.llm_hidden_dim,
                "gate_alpha_init": self.encoder.get_gate_value(),
                "trainable_params": sum(p.numel() for p in self.encoder.get_trainable_parameters()),
                "total_params": sum(p.numel() for p in self.encoder.parameters())
            })
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Profiler settings
        self.enable_profiler = enable_profiler
        self.profiler_output_dir = profiler_output_dir
        if self.enable_profiler:
            os.makedirs(self.profiler_output_dir, exist_ok=True)
            print(f"\n[Profiler] Enabled - Output directory: {self.profiler_output_dir}")
            print("[Profiler] Will profile first 5 batches with memory tracking")
    
    def inject_encoder_embedding(
        self, 
        encoder_embeddings, 
        prefix_input_ids, 
        prefix_attention_mask,
        suffix_input_ids,
        suffix_attention_mask
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
        
        breakpoint()
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
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.encoder.train()
        self.llm_model.eval()
        
        # Move batch to device and extract data
        tool_ids = batch["tool_id"].to(self.device)
        resource_vectors = batch["resource_vector"].to(self.device)
        prefix_input_ids = batch["prefix_input_ids"].to(self.device)
        prefix_attention_mask = batch["prefix_attention_mask"].to(self.device)
        suffix_input_ids = batch["suffix_input_ids"].to(self.device)
        suffix_attention_mask = batch["suffix_attention_mask"].to(self.device)
        prefix_labels = batch["prefix_labels"].to(self.device)
        suffix_labels = batch["suffix_labels"].to(self.device)
        
        # Get encoder embeddings
        encoder_embeddings = self.encoder(tool_ids, resource_vectors)
        
        # Ensure encoder embeddings match LLM dtype (critical for mixed precision)
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
        
        # Forward through LLM
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
        ], dim=1)  # [B, L_pre + 1 + L_suf]
        
        # Shift logits and labels for causal LM
        # outputs.logits: [B, L_pre + 1 + L_suf, V]
        # We want to predict: [tok_0, tok_1, ..., tok_{L-1}]
        shift_logits = outputs.logits[:, :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = combined_labels[:, 1:].contiguous()  # [B, L-1]
        
        # # Debug: Check for invalid label values
        # vocab_size = shift_logits.size(-1)
        # invalid_labels = shift_labels[(shift_labels >= vocab_size) & (shift_labels != -100)]
        # if len(invalid_labels) > 0:
        #     print(f"ERROR: Found {len(invalid_labels)} labels >= vocab_size ({vocab_size})")
        #     print(f"Invalid labels: {invalid_labels[:10]}")  # Show first 10
        #     print(f"Max label value: {shift_labels[shift_labels != -100].max().item()}")
        #     print(f"Vocab size: {vocab_size}")
        #     # Clamp invalid labels to -100 to prevent crash
        #     shift_labels = torch.where(
        #         (shift_labels >= vocab_size) | (shift_labels < -100),
        #         torch.tensor(-100, dtype=shift_labels.dtype, device=shift_labels.device),
        #         shift_labels
        #     )
        
        # Compute loss (only on non -100 labels, i.e., assistant response)
        loss = self.criterion(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        
        # Get gate_alpha value for monitoring
        gate_alpha = self.encoder.get_gate_value()
        
        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "gate_alpha": gate_alpha,
            "global_step": self.global_step
        }

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        # Setup profiler if enabled (first epoch only)
        profiler_context = None
        if self.enable_profiler and epoch == 0:
            # Profile schedule: wait=1, warmup=1, active=3, repeat=1
            # This will profile batches 2-4 (skip first for warmup)
            profiler_schedule = schedule(
                wait=1,      # Skip first batch
                warmup=1,    # Warmup on second batch
                active=3,    # Profile batches 3-5
                repeat=1     # Only do this once
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_file = os.path.join(
                self.profiler_output_dir, 
                f"profile_epoch{epoch+1}_{timestamp}.json"
            )
            
            profiler_context = profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=profiler_schedule,
                on_trace_ready=lambda prof: prof.export_chrome_trace(trace_file),
                record_shapes=True,      # Record tensor shapes
                profile_memory=True,     # Track memory allocations (critical for OOM diagnosis)
                with_stack=True,         # Record stack traces
                with_flops=False         # Disable FLOPS (can be heavy)
            )
            print(f"[Profiler] Starting profiling, trace will be saved to: {trace_file}")
            print("[Profiler] Profiling batches 2-5 (with warmup)")
        
        # Training loop with optional profiling
        if profiler_context is not None:
            profiler_context.__enter__()
            
        try:
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                
                postfix = {
                    "loss": f"{metrics['loss']:.4f}",
                    "avg_loss": f"{epoch_loss/num_batches:.4f}",
                    "lr": f"{metrics['lr']:.6f}",
                    "α": f"{metrics['gate_alpha']:.4f}"
                }
                
                # Add memory info during profiling
                if profiler_context is not None and batch_idx < 10:
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1024**3
                        mem_reserved = torch.cuda.memory_reserved() / 1024**3
                        postfix["mem_alloc"] = f"{mem_allocated:.2f}GB"
                        postfix["mem_rsv"] = f"{mem_reserved:.2f}GB"
                
                pbar.set_postfix(postfix)
                
                if self.log_wandb and self.global_step % 10 == 0:
                    wandb.log(metrics, step=self.global_step)
                
                # Step profiler if active
                if profiler_context is not None:
                    profiler_context.step()
                    # Stop profiling after 10 batches to avoid huge files
                    if batch_idx >= 9:
                        print("[Profiler] Completed profiling, stopping...")
                        break
        finally:
            if profiler_context is not None:
                profiler_context.__exit__(None, None, None)
                print(f"[Profiler] Profile saved successfully")
                print(f"[Profiler] View with: chrome://tracing or https://ui.perfetto.dev/")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        return {"train_loss": avg_loss, "epoch": epoch}

    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict):
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
            "best_loss": self.best_loss
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str) -> int:
        """Load checkpoint and return the epoch number."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model state
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        epoch = checkpoint.get("epoch", 0)
        
        print(f"Checkpoint loaded from {load_path}")
        print(f"  - Resuming from epoch {epoch + 1}")
        print(f"  - Global step: {self.global_step}")
        print(f"  - Best loss: {self.best_loss:.4f}")
        
        return epoch


def train_tool_encoder(args: argparse.Namespace):
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # 获取配置
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    dataloader_cfg = config['dataloader']
    output_cfg = config['output']
    logging_cfg = config['logging']
    
    # Get device
    device = train_cfg.get('device', args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    os.makedirs(output_cfg['output_dir'], exist_ok=True)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    
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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=dataloader_cfg['shuffle'],
        num_workers=dataloader_cfg['num_workers'],
        pin_memory=True
    )
    
    # 2. Initialize encoder 
    print("\n[2/5] Initializing encoder (with deep semantic encoding)...")
        
    encoder = ResourceEncoderForPretraining(
        llm_model_name=model_cfg['llm_model'],
        tool_registry_path=data_cfg['tool_registry'],
        d_resource=model_cfg.get('d_resource'),
        num_attention_heads=model_cfg['num_attention_heads'],
        dropout=model_cfg['dropout'],
        cache_dir=model_cfg.get('cache_dir', 'hub')
    )
    
    print(f"  ✓ Loaded {encoder.num_tools} tools")
    print(f"  ✓ Hidden dimension: {encoder.llm_hidden_dim}")
    print(f"  ✓ Gate alpha initialized to: {encoder.get_gate_value():.6f}")
    
    # 3. Load LLM
    print("\n[3/5] Loading LLM backbone...")
    
    llm_load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16
    }
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_cfg['llm_model'],
        **llm_load_kwargs
    )
    
    # 4. Optimizer
    print("\n[4/5] Setting up optimizer and scheduler...")
    optimizer = AdamW(
        encoder.get_trainable_parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    
    if train_cfg['scheduler']['type'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg['num_epochs'] * len(dataloader),
            eta_min=train_cfg['learning_rate'] * train_cfg['scheduler']['eta_min_ratio']
        )
    else:
        scheduler = None
        
    # 5. Initialize Trainer
    trainer = ToolEncoderPretrainer(
        encoder=encoder,
        llm_model=llm_model,
        tokenizer=dataset.tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_wandb=logging_cfg['wandb']['enabled'],
        project_name=logging_cfg['wandb']['project'],
        enable_profiler=args.profile,
        profiler_output_dir=args.profile_output_dir
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            print(f"\n[Resume] Loading checkpoint: {args.resume_from_checkpoint}")
            saved_epoch = trainer.load_checkpoint(args.resume_from_checkpoint)
            # Start from the next epoch after the saved one
            start_epoch = saved_epoch + 1
        else:
            print(f"Warning: Checkpoint {args.resume_from_checkpoint} not found. Starting from scratch.")

    # Training Loop
    for epoch in range(start_epoch, train_cfg['num_epochs']):
        epoch_metrics = trainer.train_epoch(epoch, dataloader)
        train_loss = epoch_metrics["train_loss"]
        
        # Get current gate_alpha value
        gate_alpha = trainer.encoder.get_gate_value()
        
        print(f"\nEpoch {epoch+1}/{train_cfg['num_epochs']} - Train Loss: {train_loss:.4f} - Gate α: {gate_alpha:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % output_cfg['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(
                output_cfg['output_dir'],
                f"{output_cfg['checkpoint_prefix']}_epoch{epoch+1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
        
        # Save best model
        if train_loss < trainer.best_loss:
            trainer.best_loss = train_loss
            best_model_path = os.path.join(
                output_cfg['output_dir'], 
                output_cfg['best_model_name']
            )
            trainer.save_checkpoint(best_model_path, epoch, epoch_metrics)
            print(f"  ✓ New best model saved (loss: {train_loss:.4f})")

    if logging_cfg['wandb']['enabled']:
        wandb.finish()
