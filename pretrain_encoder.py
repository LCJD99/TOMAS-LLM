"""
Encoder Pre-training Script for TOMAS-LLM (Multi-GPU Version - Redesigned).

This script trains the Resource Encoder using the new architecture:
- Stream A: Deep semantic encoding via LLM forward pass (frozen)
- Stream B: Trainable resource MLP
- Fusion: Gated cross-attention with learnable gate parameter (α)

Key improvements:
- Uses complete LLM forward pass for tool semantics (not just embedding layer)
- Gated fusion ensures cold-start semantic alignment (α=0 initialization)
- Precomputed semantic embeddings for efficient training
- Monitors gate_alpha evolution during training
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
from torch.profiler import profile, ProfilerActivity, schedule

# ### MODIFIED: 引入 accelerate
from accelerate import Accelerator 
from accelerate.utils import set_seed

from src.data.pretrain_dataset import EncoderPretrainDataset
from src.offline.pretrain_encoder import ResourceEncoderForPretraining


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    merged = config.copy()
    args_dict = vars(args)
    
    if args_dict.get('batch_size') is not None:
        merged['training']['batch_size'] = args_dict['batch_size']
    if args_dict.get('num_epochs') is not None:
        merged['training']['num_epochs'] = args_dict['num_epochs']
    if args_dict.get('lr') is not None:
        merged['training']['learning_rate'] = args_dict['lr']
    # device argument is handled by accelerate now
    if args_dict.get('output_dir') is not None:
        merged['output']['output_dir'] = args_dict['output_dir']
    if args_dict.get('log_wandb') is not None:
        merged['logging']['wandb']['enabled'] = args_dict['log_wandb']
    
    return merged


class EncoderPretrainer:
    def __init__(
        self,
        encoder: ResourceEncoderForPretraining,
        llm_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,  # ### MODIFIED: 传入 dataloader 以便 prepare
        accelerator: Accelerator, # ### MODIFIED: 传入 accelerator
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_wandb: bool = False,
        project_name: str = "tomas-encoder-pretrain",
        enable_profiler: bool = False,
        profiler_output_dir: str = "./profiler_outputs"
    ):
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.log_wandb = log_wandb
        
        # Freeze LLM parameters
        for param in llm_model.parameters():
            param.requires_grad = False
        
        # Note: Encoder Stream A (semantic_encoder) is already frozen by design
        # No need to manually freeze it here

        # ### MODIFIED: 使用 accelerate.prepare 包装所有对象
        # 注意：llm_model 不需要 prepare，因为它是冻结的且可以手动通过 device_map 处理，
        # 或者也可以 prepare 但需要在 forward 时注意。
        # 这里我们将 llm_model 也 prepare，以便 accelerate 处理设备放置。
        (
            self.encoder, 
            self.llm_model,
            self.optimizer, 
            self.dataloader, 
            self.scheduler
        ) = self.accelerator.prepare(
            encoder, llm_model, optimizer, dataloader, scheduler
        )

        # Initialize W&B (Only on main process)
        if log_wandb and self.accelerator.is_main_process:
            unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
            wandb.init(project=project_name, config={
                "model": "ResourceEncoderForPretraining",
                "architecture": "Deep Semantic + Gated Fusion",
                "llm_backbone": "Qwen2.5-7B",
                "num_tools": unwrapped_encoder.num_tools,
                "hidden_dim": unwrapped_encoder.llm_hidden_dim,
                "gate_alpha_init": unwrapped_encoder.get_gate_value(),
                "trainable_params": sum(p.numel() for p in unwrapped_encoder.get_trainable_parameters()),
                "total_params": sum(p.numel() for p in unwrapped_encoder.parameters())
            })
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Profiler settings
        self.enable_profiler = enable_profiler
        self.profiler_output_dir = profiler_output_dir
        if self.enable_profiler and self.accelerator.is_main_process:
            os.makedirs(self.profiler_output_dir, exist_ok=True)
            print(f"\n[Profiler] Enabled - Output directory: {self.profiler_output_dir}")
            print("[Profiler] Will profile first 5 batches with memory tracking")
    
    def inject_prefix_embeddings(self, encoder_embeddings, input_ids, attention_mask):
        # 逻辑保持不变，Tensor 已经在正确的 device 上
        batch_size = encoder_embeddings.size(0)
        
        # LLM 可能是被 DDP 包装的，访问原始方法可能需要 unwrapping，
        # 但通常 __call__ 会透传。如果报错，尝试 self.llm_model.module.get_input_embeddings()
        # 这里通常直接调用是可以的，因为 DDP 包装的是 forward
        # 为了安全，获取 embedding layer 最好通过 unwrap 或者直接属性访问（如果 transformers 支持）
        try:
            embed_layer = self.llm_model.get_input_embeddings()
        except AttributeError:
            embed_layer = self.llm_model.module.get_input_embeddings()

        token_embeddings = embed_layer(input_ids)
        prefix_embeddings = encoder_embeddings.unsqueeze(1)
        combined_embeddings = torch.cat([prefix_embeddings, token_embeddings], dim=1)
        
        prefix_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return combined_embeddings, combined_attention_mask
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.encoder.train()
        self.llm_model.eval()
        
        # ### MODIFIED: 不需要手动 .to(device)，accelerate 也就是 dataloader 会自动处理
        tool_ids = batch["tool_id"]
        resource_vectors = batch["resource_vector"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        encoder_embeddings = self.encoder(tool_ids, resource_vectors)
        
        # Ensure encoder embeddings match LLM dtype (critical for mixed precision)
        llm_dtype = next(self.llm_model.parameters()).dtype
        encoder_embeddings = encoder_embeddings.to(llm_dtype)
        
        combined_embeddings, combined_attention_mask = self.inject_prefix_embeddings(
            encoder_embeddings, input_ids, attention_mask
        )
        
        outputs = self.llm_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            return_dict=True
        )
        
        # Shift logits and labels for causal LM
        # outputs.logits: [B, L+1, V] (prefix + L tokens)
        # We want to predict: [tok_0, tok_1, ..., tok_{L-1}]
        # Using logits from: [prefix, tok_0, ..., tok_{L-2}]
        shift_logits = outputs.logits[:, :-1, :].contiguous()  # [B, L, V]
        shift_labels = labels.contiguous()  # [B, L]
        
        # Compute loss
        loss = self.criterion(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        
        self.optimizer.zero_grad()
        
        # ### MODIFIED: 使用 accelerator.backward 替代 loss.backward
        self.accelerator.backward(loss)
        
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        
        # Get gate_alpha value for monitoring
        gate_alpha = self.accelerator.unwrap_model(self.encoder).get_gate_value()
        
        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "gate_alpha": gate_alpha,
            "global_step": self.global_step
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        # ### MODIFIED: dataloader 已经在 init 中被 prepare 并存为 self.dataloader
        epoch_loss = 0.0
        num_batches = 0
        
        # Only show progress bar on main process
        if self.accelerator.is_main_process:
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = self.dataloader
        
        # Setup profiler if enabled (only on main process and first epoch)
        profiler_context = None
        if self.enable_profiler and self.accelerator.is_main_process and epoch == 0:
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
                
                if self.accelerator.is_main_process:
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
                        if self.accelerator.is_main_process:
                            print("[Profiler] Completed profiling, stopping...")
                        break
        finally:
            if profiler_context is not None:
                profiler_context.__exit__(None, None, None)
                if self.accelerator.is_main_process:
                    print(f"[Profiler] Profile saved successfully")
                    print(f"[Profiler] View with: chrome://tracing or https://ui.perfetto.dev/")
        
        # Gather metrics across GPUs (optional, mainly for accurate validation loss)
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        return {"train_loss": avg_loss, "epoch": epoch}

    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict):
        # ### MODIFIED: 所有的保存操作只在主进程进行，并且需要 unwrap 模型
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
            checkpoint = {
                "epoch": epoch,
                "encoder_state_dict": unwrapped_encoder.state_dict(),
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
        # Load checkpoint on main process
        checkpoint = torch.load(load_path, map_location="cpu")
        
        # Load model state
        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
        unwrapped_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        
        # Load optimizer state (prepare later will handle device placement)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        epoch = checkpoint.get("epoch", 0)
        
        # Ensure all processes are synchronized
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            print(f"Checkpoint loaded from {load_path}")
            print(f"  - Resuming from epoch {epoch + 1}")
            print(f"  - Global step: {self.global_step}")
            print(f"  - Best loss: {self.best_loss:.4f}")
        
        return epoch


def main():
    parser = argparse.ArgumentParser(description="Pre-train Resource Encoder")
    parser.add_argument("--config", type=str, default="configs/pretrain_encoder.yaml")
    # ... 其他原有参数保留 ...
    # device 参数可以移除或忽略，由 accelerate 控制
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None) 
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_wandb", action="store_true", default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to the .pt checkpoint file to resume from")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable torch profiler for performance and memory analysis")
    parser.add_argument("--profile_output_dir", type=str, default="./profiler_outputs",
                        help="Directory to save profiler output files")

    
    args = parser.parse_args()
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # ### MODIFIED: 初始化 Accelerator
    # mixed_precision='bf16' 是 4090 的关键
    accelerator = Accelerator(mixed_precision='bf16') 
    
    # 获取配置
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    dataloader_cfg = config['dataloader']
    output_cfg = config['output']
    logging_cfg = config['logging']
    
    if accelerator.is_main_process:
        os.makedirs(output_cfg['output_dir'], exist_ok=True)
        print("="*80)
        print(f"TOMAS-LLM Encoder Pre-training (Distributed: {accelerator.num_processes} GPUs)")
        print("="*80)

    # 1. Load dataset
    if accelerator.is_main_process:
        print("\n[1/5] Loading dataset...")
    
    # 设置随机种子
    set_seed(data_cfg['augmentation']['seed'])

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
        batch_size=train_cfg['batch_size'], # 这里的 batch_size 是单卡的
        shuffle=dataloader_cfg['shuffle'],
        num_workers=dataloader_cfg['num_workers'],
        pin_memory=True
    )
    
    # 2. Initialize encoder (REDESIGNED API)
    if accelerator.is_main_process:
        print("\n[2/5] Initializing encoder (with deep semantic encoding)...")
        
    encoder = ResourceEncoderForPretraining(
        llm_model_name=model_cfg['llm_model'],
        tool_registry_path=data_cfg['tool_registry'],  # NEW: Required for semantic encoding
        d_resource=model_cfg.get('d_resource'),  # Optional: auto-matches LLM hidden dim
        num_attention_heads=model_cfg['num_attention_heads'],
        dropout=model_cfg['dropout'],
        cache_dir=model_cfg.get('cache_dir', 'hub')
    )
    
    if accelerator.is_main_process:
        print(f"  ✓ Loaded {encoder.num_tools} tools")
        print(f"  ✓ Hidden dimension: {encoder.llm_hidden_dim}")
        print(f"  ✓ Gate alpha initialized to: {encoder.get_gate_value():.6f}")
    
    # 3. Load LLM
    if accelerator.is_main_process:
        print("\n[3/5] Loading LLM backbone...")
    
    llm_load_kwargs = {
        "trust_remote_code": True,
        # ### MODIFIED: 关键修改！一定要用 bfloat16，否则 4090 存不下 FP32
        "torch_dtype": torch.bfloat16 
    }
    
    # 自动加载到内存，稍后由 prepare 移到 GPU
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_cfg['llm_model'],
        **llm_load_kwargs
    )
    
    # 4. Optimizer
    if accelerator.is_main_process:
        print("\n[4/5] Setting up optimizer and scheduler...")
        trainable_params = sum(p.numel() for p in encoder.get_trainable_parameters())
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"  ✓ Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
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
    trainer = EncoderPretrainer(
        encoder=encoder,
        llm_model=llm_model,
        tokenizer=dataset.tokenizer,
        optimizer=optimizer,
        dataloader=dataloader,
        accelerator=accelerator, # 传入 accelerator
        scheduler=scheduler,
        log_wandb=logging_cfg['wandb']['enabled'],
        project_name=logging_cfg['wandb']['project'],
        enable_profiler=args.profile,
        profiler_output_dir=args.profile_output_dir
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            if accelerator.is_main_process:
                print(f"\n[6/5] Resuming from checkpoint: {args.resume_from_checkpoint}")
            saved_epoch = trainer.load_checkpoint(args.resume_from_checkpoint)
            # Start from the next epoch after the saved one
            start_epoch = saved_epoch + 1
        else:
            if accelerator.is_main_process:
                print(f"Warning: Checkpoint {args.resume_from_checkpoint} not found. Starting from scratch.")

    # Training Loop
    for epoch in range(start_epoch, train_cfg['num_epochs']):
        epoch_metrics = trainer.train_epoch(epoch)
        train_loss = epoch_metrics["train_loss"]
        
        if accelerator.is_main_process:
            # Get current gate_alpha value
            gate_alpha = trainer.accelerator.unwrap_model(trainer.encoder).get_gate_value()
            
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

    if accelerator.is_main_process and logging_cfg['wandb']['enabled']:
        wandb.finish()

if __name__ == "__main__":
    main()