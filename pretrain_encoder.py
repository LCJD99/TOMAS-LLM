"""
Encoder Pre-training Script for TOMAS-LLM (Multi-GPU Version).
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
        project_name: str = "tomas-encoder-pretrain"
    ):
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.log_wandb = log_wandb
        
        # Freeze LLM parameters
        for param in llm_model.parameters():
            param.requires_grad = False
        
        # Ensure encoder Stream A is frozen
        for param in encoder.semantic_embedding.parameters():
            param.requires_grad = False

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
            wandb.init(project=project_name, config={
                "model": "ResourceEncoderForPretraining",
                "llm_backbone": "Qwen2.5-7B",
                # ### MODIFIED: 获取解包后的参数量
                "trainable_params": sum(p.numel() for p in self.accelerator.unwrap_model(self.encoder).get_trainable_parameters())
            })
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.best_loss = float('inf')
        self.global_step = 0
    
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
        
        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
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
            
        for batch in pbar:
            metrics = self.train_step(batch)
            epoch_loss += metrics["loss"]
            num_batches += 1
            
            if self.accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "avg_loss": f"{epoch_loss/num_batches:.4f}",
                    "lr": f"{metrics['lr']:.6f}"
                })
                
                if self.log_wandb and self.global_step % 10 == 0:
                    wandb.log(metrics, step=self.global_step)
        
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
    
    # 2. Initialize encoder
    if accelerator.is_main_process:
        print("\n[2/5] Initializing encoder...")
        
    encoder = ResourceEncoderForPretraining(
        llm_model_name=model_cfg['llm_model'],
        llm_hidden_dim=model_cfg['llm_hidden_dim'],
        d_resource=model_cfg['d_resource'],
        num_attention_heads=model_cfg['num_attention_heads'],
        dropout=model_cfg['dropout'],
        num_tools=model_cfg['num_tools'],
        freeze_semantic=model_cfg['freeze_semantic'],
        cache_dir=model_cfg.get('cache_dir', 'hub')
    )
    
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
    
    breakpoint()
    # 4. Optimizer
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
        project_name=logging_cfg['wandb']['project']
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
            print(f"\nEpoch {epoch+1}/{train_cfg['num_epochs']} - Train Loss: {train_loss:.4f}")
            
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