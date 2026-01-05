#!/usr/bin/env python3
"""
Tokenizer 扩充脚本
将虚拟 Token 添加到 Qwen2.5-7B tokenizer 中
"""

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def load_registry(registry_path: str) -> dict:
    """加载工具注册表"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    return registry


def extract_tokens(registry: dict) -> list:
    """从注册表提取所有虚拟 Token"""
    tokens = list(registry['tokens'].keys())
    # 添加角括号包裹
    tokens = [f"<{token}>" for token in tokens]
    return tokens


def expand_tokenizer(base_model: str, virtual_tokens: list, output_dir: str):
    """扩充 tokenizer"""
    print(f"加载基础 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    
    original_vocab_size = len(tokenizer)
    print(f"原始词表大小: {original_vocab_size}")
    
    # 添加虚拟 Token
    print(f"添加 {len(virtual_tokens)} 个虚拟 Token...")
    num_added = tokenizer.add_tokens(virtual_tokens, special_tokens=True)
    
    new_vocab_size = len(tokenizer)
    print(f"新词表大小: {new_vocab_size}")
    print(f"实际添加: {num_added} 个 Token")
    
    # 验证
    assert new_vocab_size == original_vocab_size + num_added, "词表大小不匹配"
    
    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"保存扩充后的 tokenizer 到: {output_dir}")
    tokenizer.save_pretrained(output_dir)
    
    return tokenizer, num_added


def verify_tokens(tokenizer, virtual_tokens: list):
    """验证 Token 添加"""
    print("\n验证 Token 添加...")
    
    # 测试编码/解码
    test_tokens = virtual_tokens[:5]
    for token in test_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded = tokenizer.decode([token_id])
        print(f"  {token} -> ID: {token_id} -> 解码: {decoded}")
        assert decoded == token, f"解码不匹配: {token} != {decoded}"
    
    print("✓ Token 编码/解码验证通过")


def main():
    parser = argparse.ArgumentParser(description='扩充 Tokenizer')
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='基础模型名称或路径 (如 Qwen/Qwen2.5-7B)'
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
        '--verify',
        action='store_true',
        help='验证 Token 添加'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tokenizer 扩充")
    print("=" * 60)
    
    # 加载注册表
    registry = load_registry(args.registry)
    print(f"\n加载注册表: {registry['metadata']['total_tokens']} tokens")
    
    # 提取虚拟 Token
    virtual_tokens = extract_tokens(registry)
    print(f"提取虚拟 Token: {len(virtual_tokens)} 个")
    
    # 扩充 tokenizer
    tokenizer, num_added = expand_tokenizer(
        args.base_model,
        virtual_tokens,
        args.output
    )
    
    # 验证
    if args.verify:
        verify_tokens(tokenizer, virtual_tokens)
    
    print("\n" + "=" * 60)
    print("Tokenizer 扩充完成!")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print(f"添加 Token 数: {num_added}")


if __name__ == '__main__':
    main()
