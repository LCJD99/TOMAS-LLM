"""
Stage 1 Trainer V2: Tool Token Learning with Dynamic Embeddings
使用 Dynamic Tool Embedding + Profile Encoder (Hypernetwork) 架构
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

from src.models.profile_encoder import ProfileHyperNet, create_profile_encoder, load_system_config
from src.models.dynamic_tool_embedding import DynamicToolEmbedding
from src.models.dynamic_lm_head import DynamicLMHead

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


def load_tool_registry(registry_path: str) -> Dict:
    """Load tool registry JSON"""
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    return registry


def prepare_tool_data(
    registry: Dict,
    tokenizer,
    base_model,
    device: str = 'cuda',
    use_real_values: bool = True
) -> tuple:
    """
    准备工具语义向量和配置向量
    
    Args:
        registry: 工具注册表
        tokenizer: Tokenizer
        base_model: 基础模型（用于计算语义向量）
        device: 设备
        use_real_values: 是否使用真实配置值（True）还是离散级别（False）
    
    Returns:
        (tool_semantics, profiles, token_list)
        - tool_semantics: [num_tokens, hidden_dim]
        - profiles: [num_tokens, 5] 真实配置值或离散级别
        - token_list: token 名称列表
    """
    print("Preparing tool semantic embeddings and profile vectors...")
    
    tokens = registry['tokens']
    token_list = sorted(tokens.keys())
    num_tokens = len(token_list)
    
    # 获取 hidden_size
    hidden_size = base_model.config.hidden_size
    
    # 初始化存储
    tool_semantics = torch.zeros(num_tokens, hidden_size, device=device)
    profiles = torch.zeros(num_tokens, 5, device=device)
    
    # 获取原始 embedding 层
    input_embeddings = base_model.get_input_embeddings()
    
    with torch.no_grad():
        for idx, token_name in enumerate(token_list):
            token_info = tokens[token_name]
            
            # 1. 计算语义向量（使用 semantic_description）
            desc_text = token_info.get('semantic_description', token_info['description'])
            
            # Tokenize description
            desc_input_ids = tokenizer(
                desc_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )['input_ids'].to(device)
            
            # 获取 embeddings 并平均
            desc_embeddings = input_embeddings(desc_input_ids)
            semantic_vec = desc_embeddings.mean(dim=1).squeeze(0)  # [hidden_size]
            
            tool_semantics[idx] = semantic_vec
            
            # 2. 提取资源配置向量
            resources = token_info.get('resources', {})
            resource_levels = token_info.get('resource_levels', {})
            
            if use_real_values:
                # 使用真实配置值
                # 映射 input_size 到数值：small=1, medium=2, large=3
                size_map = {'small': 1, 'medium': 2, 'large': 3}
                tool_size = size_map.get(token_info.get('input_size', 'medium'), 2)
                
                cpu_core = resources.get('cpu_core', 2)
                cpu_mem = resources.get('cpu_mem_gb', 4.0)
                gpu_sm = resources.get('gpu_sm', 20)
                gpu_mem = resources.get('gpu_mem_gb', 2.0)
            else:
                # 使用离散级别：low=1, medium=2, high=3
                level_map = {'low': 1, 'medium': 2, 'med': 2, 'high': 3}
                
                tool_size = level_map.get(resource_levels.get('tool_size', 'medium'), 2)
                cpu_core = level_map.get(resource_levels.get('cpu_core', 'low'), 1)
                cpu_mem = level_map.get(resource_levels.get('cpu_mem', 'low'), 1)
                gpu_sm = level_map.get(resource_levels.get('gpu_sm', 'low'), 1)
                gpu_mem = level_map.get(resource_levels.get('gpu_mem', 'low'), 1)
            
            profiles[idx] = torch.tensor([tool_size, cpu_core, cpu_mem, gpu_sm, gpu_mem])
    
    print(f"Prepared {num_tokens} tool embeddings")
    print(f"Tool semantics shape: {tool_semantics.shape}")
    print(f"Profiles shape: {profiles.shape}")
    print(f"Profile value range: [{profiles.min():.1f}, {profiles.max():.1f}]")
    if use_real_values:
        print("Using real configuration values")
    else:
        print("Using discrete levels (1-3)")
    
    return tool_semantics, profiles, token_list


def setup_model_with_dynamic_embeddings(
    config: Dict,
    load_lora_from: Optional[str] = None
):
    """
    设置带有动态 Embedding 的模型
    
    Args:
        config: 配置字典
        load_lora_from: 已有 LoRA checkpoint 路径（可选）
    
    Returns:
        (model, tokenizer, new_token_start_idx)
    """
    print("=" * 60)
    print("Setting up model with Dynamic Tool Embeddings")
    print("=" * 60)
    
    # 1. 加载工具注册表
    registry = load_tool_registry(config['tool_registry'])
    
    # 2. 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer'],
        trust_remote_code=True
    )
    
    original_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {original_vocab_size}")
    
    # 计算新 token 起始索引（假设扩充的 tokenizer 已包含虚拟 token）
    # 如果是基础 tokenizer，需要添加新 token
    token_list = sorted(registry['tokens'].keys())
    new_tokens = [f"<{token_name}>" for token_name in token_list]
    
    # 检查是否需要添加新 token
    first_new_token = new_tokens[0]
    if first_new_token not in tokenizer.get_vocab():
        print(f"Adding {len(new_tokens)} new tokens to tokenizer...")
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} tokens")
    
    new_token_start_idx = tokenizer.convert_tokens_to_ids(first_new_token)
    print(f"New token start index: {new_token_start_idx}")
    
    # 3. 加载基础模型
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.get('base_model', config.get('initialized_model')),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.get('bf16', True) else torch.float16,
        device_map='auto'
    )
    
    # 调整 embedding 大小
    model.resize_token_embeddings(len(tokenizer))
    
    # 4. 准备工具数据
    device = next(model.parameters()).device
    use_real_values = config.get('use_real_profile_values', True)
    tool_semantics, profiles, _ = prepare_tool_data(
        registry, tokenizer, model, device=device, use_real_values=use_real_values
    )
    
    # 5. 加载系统配置（用于归一化）
    system_config_path = config.get('system_config_path', None)
    max_values = load_system_config(system_config_path) if use_real_values else None
    if max_values:
        print(f"\nSystem max configuration:")
        for k, v in max_values.items():
            print(f"  {k}: {v}")
    
    # 6. 创建 Profile Encoder (Hypernetwork)
    print("\nCreating Profile Encoder (Hypernetwork)...")
    profile_encoder = create_profile_encoder(
        encoder_type=config.get('profile_encoder_type', 'simple'),
        input_dim=5,
        output_dim=model.config.hidden_size,
        hidden_dims=config.get('profile_encoder_hidden_dims', [128, 512]),
        activation=config.get('profile_encoder_activation', 'gelu'),
        dropout=config.get('profile_encoder_dropout', 0.0),
        zero_init=True,  # 关键：零初始化
        normalize=use_real_values,  # 是否归一化
        max_values=max_values,  # 最大配置值
        system_config_path=system_config_path
    )
    
    profile_encoder = profile_encoder.to(
        device=device,
        dtype=torch.bfloat16 if config.get('bf16', True) else torch.float16
    )
    
    print(f"Profile Encoder parameters: {sum(p.numel() for p in profile_encoder.parameters()):,}")
    
    # 6a. 替换 Embedding 层为 Dynamic Embedding
    print("\nReplacing embedding layer with DynamicToolEmbedding...")
    original_embed_layer = model.model.embed_tokens
    
    dynamic_embed_layer = DynamicToolEmbedding(
        original_embedding=original_embed_layer,
        profile_encoder=profile_encoder,
        tool_semantics=tool_semantics,
        profiles=profiles,
        new_token_start_idx=new_token_start_idx
    )
    
    # 强制替换
    model.model.embed_tokens = dynamic_embed_layer
    
    print("\u2713 Dynamic embedding layer installed")
    
    # 6b. 替换 LM Head 为 Dynamic LM Head
    print("\nReplacing lm_head with DynamicLMHead...")
    original_lm_head = model.lm_head
    
    dynamic_lm_head = DynamicLMHead(
        original_lm_head=original_lm_head,
        profile_encoder=profile_encoder,
        tool_semantics=tool_semantics,
        profiles=profiles,
        new_token_start_idx=new_token_start_idx
    )
    
    # 强制替换
    model.lm_head = dynamic_lm_head
    
    print("✓ Dynamic LM Head installed")
    
    # 7. 配置 LoRA
    if config.get('use_lora', True):
        if load_lora_from:
            print(f"\nLoading existing LoRA from {load_lora_from}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, load_lora_from)
            print("✓ LoRA loaded")
        else:
            print("\nConfiguring new LoRA...")
            lora_config = LoraConfig(
                r=config.get('lora_r', 64),
                lora_alpha=config.get('lora_alpha', 32),
                target_modules=config.get('lora_target_modules', [
                    'q_proj', 'v_proj', 'k_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ]),
                lora_dropout=config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
            print("✓ LoRA configured")
        
        model.print_trainable_parameters()
    
    # 8. Enable gradient checkpointing
    if config.get('gradient_checkpointing', False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # 9. 设置 Profile Encoder 可训练
    print("\nEnabling Profile Encoder gradients...")
    # 获取实际的 profile_encoder（可能被 PEFT 包裹）
    if hasattr(model, 'base_model'):
        # Embedding layer的 profile_encoder
        embed_encoder = model.base_model.model.model.embed_tokens.profile_encoder
        # LM Head的 profile_encoder
        lm_head_encoder = model.base_model.model.lm_head.profile_encoder
    else:
        embed_encoder = model.model.embed_tokens.profile_encoder
        lm_head_encoder = model.lm_head.profile_encoder
    
    for param in embed_encoder.parameters():
        param.requires_grad = True
    for param in lm_head_encoder.parameters():
        param.requires_grad = True
    
    encoder_params = sum(p.numel() for p in embed_encoder.parameters() if p.requires_grad)
    lm_head_encoder_params = sum(p.numel() for p in lm_head_encoder.parameters() if p.requires_grad)
    print(f"✓ Embedding Profile Encoder trainable params: {encoder_params:,}")
    print(f"✓ LM Head Profile Encoder trainable params: {lm_head_encoder_params:,}")
    
    # 注意：两个 encoder 共享同一个实例，参数不重复计算
    print(f"Note: Embed and LM Head share the same Profile Encoder instance")
    
    # 10. 设置 DynamicLMHead 的 original_lm_head 可训练（可选）
    if config.get('train_lm_head', True):
        print("\nSetting original_lm_head trainable...")
        if hasattr(model, 'base_model'):
            original_lm = model.base_model.model.lm_head.original_lm_head
        else:
            original_lm = model.lm_head.original_lm_head
        
        for param in original_lm.parameters():
            param.requires_grad = True
        
        lm_params = sum(p.numel() for p in original_lm.parameters() if p.requires_grad)
        print(f"✓ original_lm_head trainable params: {lm_params:,}")
    
    print("\n" + "=" * 60)
    print("Model setup complete!")
    print("=" * 60)
    
    return model, tokenizer, new_token_start_idx


class DynamicEmbeddingTrainer(Trainer):
    """
    自定义 Trainer：为 Profile Encoder 设置不同的学习率
    """
    
    def __init__(self, *args, hypernet_lr: float = 1e-3, **kwargs):
        self.hypernet_lr = hypernet_lr
        super().__init__(*args, **kwargs)
    
    def create_optimizer(self):
        """
        创建优化器，为不同组件设置不同学习率
        - LoRA parameters: 使用配置中的学习率
        - Profile Encoder (Hypernetwork): 使用更大的学习率
        - lm_head: 使用配置中的学习率
        """
        if self.optimizer is None:
            # 分组参数
            hypernet_params = []
            lora_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if "profile_encoder" in name or "hypernet" in name:
                    hypernet_params.append(param)
                elif "lora" in name:
                    lora_params.append(param)
                else:
                    other_params.append(param)
            
            # 构建优化器组
            optimizer_grouped_parameters = []
            
            if lora_params:
                optimizer_grouped_parameters.append({
                    "params": lora_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                })
                print(f"LoRA params: {sum(p.numel() for p in lora_params):,}, lr={self.args.learning_rate}")
            
            if hypernet_params:
                optimizer_grouped_parameters.append({
                    "params": hypernet_params,
                    "weight_decay": 0.0,  # Profile Encoder 不使用 weight decay
                    "lr": self.hypernet_lr,
                })
                print(f"Hypernet params: {sum(p.numel() for p in hypernet_params):,}, lr={self.hypernet_lr}")
            
            if other_params:
                optimizer_grouped_parameters.append({
                    "params": other_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                })
                print(f"Other params: {sum(p.numel() for p in other_params):,}, lr={self.args.learning_rate}")
            
            # 创建优化器
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        
        return self.optimizer


def setup_training_args(config: Dict) -> TrainingArguments:
    """Setup TrainingArguments from config"""
    # Determine report_to
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
    )
    
    return training_args


def train_stage1_v2(config_path: str, load_lora_from: Optional[str] = None):
    """
    Main training function with Dynamic Embeddings
    
    Args:
        config_path: Path to configuration YAML file
        load_lora_from: Path to existing LoRA checkpoint (optional)
    """
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Initialize wandb if enabled
    if config.get('report_to') == 'wandb' and WANDB_AVAILABLE:
        wandb.init(
            project=config.get('wandb_project', 'tomas-llm'),
            name=config.get('wandb_run_name', 'stage1-v2-dynamic-embeddings'),
            config=config
        )
    
    # Setup model with dynamic embeddings
    model, tokenizer, new_token_start_idx = setup_model_with_dynamic_embeddings(
        config, load_lora_from=load_lora_from
    )
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Load dataset
    from src.datasets.tool_instruction_dataset import ToolInstructionDataset, ToolInstructionDataCollator
    
    print(f"\nLoading training data from {config['train_data']}...")
    train_dataset = ToolInstructionDataset(
        data_path=config['train_data'],
        tokenizer=tokenizer,
        max_length=config.get('max_length', 512)
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Data collator
    data_collator = ToolInstructionDataCollator(tokenizer=tokenizer)
    
    # Custom callback
    from src.engine.evaluator import EvaluationCallback
    callbacks = [EvaluationCallback(tokenizer=tokenizer)]
    
    # Initialize Custom Trainer with different learning rates
    hypernet_lr = float(config.get('hypernet_lr', 1e-3))
    
    trainer = DynamicEmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        hypernet_lr=hypernet_lr
    )
    
    # Train
    print("\n" + "=" * 60)
    if load_lora_from:
        print(f"Starting stage-2 training with LoRA from: {load_lora_from}")
    else:
        print("Starting stage-1 training with Dynamic Embeddings...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    output_dir = Path(config['output_dir']) / 'final_model'
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save additional info
    model_info = {
        'new_token_start_idx': new_token_start_idx,
        'num_new_tokens': len(train_dataset.tokenizer) - new_token_start_idx,
        'hypernet_lr': hypernet_lr,
        'architecture': 'dynamic_tool_embedding_v2'
    }
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {output_dir}")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stage 1 Model V2 (Dynamic Embeddings)')
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
    
    train_stage1_v2(args.config, load_lora_from=args.load_lora_from)
