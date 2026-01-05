#!/usr/bin/env python3
"""
Embedding 初始化
为虚拟 Token 计算有意义的初始 Embedding (而非随机初始化)
"""

import json
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_registry(registry_path: str) -> dict:
    """加载工具注册表"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    return registry


def initialize_embeddings(
    base_model_name: str,
    expanded_tokenizer_path: str,
    registry: dict,
    output_dir: str,
    device: str = 'cuda'
):
    """初始化虚拟 Token 的 Embedding"""
    
    print(f"加载基础模型: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    print(f"加载扩充后的 tokenizer: {expanded_tokenizer_path}")
    expanded_tokenizer = AutoTokenizer.from_pretrained(
        expanded_tokenizer_path,
        trust_remote_code=True
    )
    
    original_vocab_size = len(base_tokenizer)
    new_vocab_size = len(expanded_tokenizer)
    num_new_tokens = new_vocab_size - original_vocab_size
    
    print(f"原始词表大小: {original_vocab_size}")
    print(f"新词表大小: {new_vocab_size}")
    print(f"新增 Token 数: {num_new_tokens}")
    
    # 扩充模型 Embedding 层
    print("扩充模型 Embedding 层...")
    base_model.resize_token_embeddings(new_vocab_size)
    
    # 获取 Embedding 层
    input_embeddings = base_model.get_input_embeddings()
    output_embeddings = base_model.get_output_embeddings()
    
    print(f"初始化虚拟 Token Embeddings (共 {len(registry['tokens'])} 个)...")
    
    # 为每个虚拟 Token 初始化 Embedding
    with torch.no_grad():
        for token_name, token_info in tqdm(registry['tokens'].items(), desc="初始化 Embeddings"):
            # 构造语义描述文本
            semantic_text = f"{token_info['description']} with {token_info['semantic_description']}"
            
            # Tokenize 并获取原始 Embedding
            input_ids = base_tokenizer(
                semantic_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )['input_ids'].to(device)
            
            # 获取文本的 embeddings
            text_embeddings = input_embeddings(input_ids)
            
            # 计算均值作为新 Token 的初始 Embedding
            mean_embedding = text_embeddings.mean(dim=1).squeeze(0)  # [hidden_size]
            
            # 获取新 Token 的 ID
            virtual_token = f"<{token_name}>"
            new_token_id = expanded_tokenizer.convert_tokens_to_ids(virtual_token)
            
            if new_token_id == expanded_tokenizer.unk_token_id:
                print(f"警告: Token {virtual_token} 未找到")
                continue
            
            # 赋值给新 Token 的 Embedding
            input_embeddings.weight.data[new_token_id] = mean_embedding
            if output_embeddings is not None:
                output_embeddings.weight.data[new_token_id] = mean_embedding
    
    # 保存初始化后的模型
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存初始化后的模型到: {output_dir}")
    base_model.save_pretrained(output_dir)
    expanded_tokenizer.save_pretrained(output_dir)
    
    # 保存初始化信息
    init_info = {
        "base_model": base_model_name,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "num_new_tokens": num_new_tokens,
        "num_initialized": len(registry['tokens']),
        "embedding_dim": input_embeddings.weight.shape[1]
    }
    
    with open(output_path / 'embedding_init_info.json', 'w', encoding='utf-8') as f:
        json.dump(init_info, f, indent=2, ensure_ascii=False)
    
    print(f"保存初始化信息到: {output_path / 'embedding_init_info.json'}")
    
    return base_model, expanded_tokenizer, init_info


def verify_embeddings(model, tokenizer, registry: dict, num_samples: int = 5):
    """验证 Embedding 初始化"""
    print("\n验证 Embedding 初始化...")
    
    input_embeddings = model.get_input_embeddings()
    
    # 随机选择几个 Token 验证
    import random
    token_names = random.sample(list(registry['tokens'].keys()), min(num_samples, len(registry['tokens'])))
    
    for token_name in token_names:
        virtual_token = f"<{token_name}>"
        token_id = tokenizer.convert_tokens_to_ids(virtual_token)
        
        if token_id == tokenizer.unk_token_id:
            print(f"  警告: {virtual_token} 未找到")
            continue
        
        embedding = input_embeddings.weight.data[token_id]
        
        # 检查是否为零向量或全为相同值
        is_zero = torch.all(embedding == 0).item()
        is_constant = torch.all(embedding == embedding[0]).item()
        norm = torch.norm(embedding).item()
        
        status = "✓" if not (is_zero or is_constant) else "×"
        print(f"  {status} {virtual_token}: norm={norm:.4f}, zero={is_zero}, constant={is_constant}")
    
    print("验证完成")


def main():
    parser = argparse.ArgumentParser(description='初始化虚拟 Token Embeddings')
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='基础模型名称或路径 (如 Qwen/Qwen2.5-7B)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='扩充后的 tokenizer 路径'
    )
    parser.add_argument(
        '--registry',
        type=str,
        required=True,
        help='工具注册表 JSON 文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证 Embedding 初始化'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Embedding 初始化")
    print("=" * 60)
    
    # 加载注册表
    registry = load_registry(args.registry)
    print(f"\n加载注册表: {len(registry['tokens'])} 个虚拟 Token\n")
    
    # 初始化 Embeddings
    model, tokenizer, init_info = initialize_embeddings(
        args.base_model,
        args.tokenizer,
        registry,
        args.output,
        args.device
    )
    
    # 验证
    if args.verify:
        verify_embeddings(model, tokenizer, registry)
    
    print("\n" + "=" * 60)
    print("Embedding 初始化完成!")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print(f"初始化 Token 数: {init_info['num_initialized']}")
    print(f"Embedding 维度: {init_info['embedding_dim']}")


if __name__ == '__main__':
    main()
