#!/usr/bin/env python3
"""
构造训练数据 (Instruction Data)
生成阶段一训练数据，训练模型将自然语言映射到虚拟 Token
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from .templates import AUGMENTATION_GENERATORS
    from .token_schema import format_token_for_model
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from templates import AUGMENTATION_GENERATORS
    from token_schema import format_token_for_model


def load_registry(registry_path: str) -> Dict:
    """加载工具注册表"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    return registry


def generate_instruction_sample(
    token_name: str,
    token_info: Dict,
    augmentation_type: str
) -> Dict:
    """
    生成单个训练样本
    
    Args:
        token_name: token 名称 (如 IMG_CLS_SMALL_LOW_MED_HIGH_LOW)
        token_info: token 的详细信息
        augmentation_type: 数据增强类型
    
    Returns:
        训练样本字典
    """
    # 获取对应的生成器
    generator = AUGMENTATION_GENERATORS[augmentation_type]
    
    # 生成输入文本
    input_text = generator(token_info)
    
    # 输出是格式化的 token (带尖括号)
    output_token = format_token_for_model(token_name)
    
    # 构造样本
    sample = {
        "input": input_text,
        "output": output_token,
        "metadata": {
            "augmentation_type": augmentation_type,
            "expected_latency_ms": token_info['latency_ms'],
            "tool": token_info['tool_name'],
            "token_name": token_name,
            "input_size": token_info['input_size'],
            "resources": token_info['resources'],
            "resource_levels": token_info['resource_levels']
        }
    }
    
    return sample


def generate_all_samples(registry: Dict, augmentation_types: List[str]) -> List[Dict]:
    """
    生成所有训练样本
    
    Args:
        registry: 工具注册表
        augmentation_types: 数据增强类型列表
    
    Returns:
        所有样本列表
    """
    all_samples = []
    tokens = registry['tokens']
    
    print(f"总共 {len(tokens)} 个 tokens")
    print(f"每个 token 生成 {len(augmentation_types)} 种增强样本")
    print(f"预计生成 {len(tokens) * len(augmentation_types)} 个样本\n")
    
    for token_name, token_info in tokens.items():
        for aug_type in augmentation_types:
            sample = generate_instruction_sample(token_name, token_info, aug_type)
            all_samples.append(sample)
    
    print(f"实际生成了 {len(all_samples)} 个样本")
    return all_samples


def save_jsonl(samples: List[Dict], output_path: str):
    """保存为 JSONL 格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"已保存 {len(samples)} 个样本到 {output_path}")


def print_statistics(samples: List[Dict], split_name: str):
    """打印数据统计信息"""
    print(f"\n=== {split_name} 数据统计 ===")
    
    # 按增强类型统计
    aug_counts = defaultdict(int)
    tool_counts = defaultdict(int)
    input_size_counts = defaultdict(int)
    
    for sample in samples:
        aug_type = sample['metadata']['augmentation_type']
        tool = sample['metadata']['tool']
        input_size = sample['metadata']['input_size']
        
        aug_counts[aug_type] += 1
        tool_counts[tool] += 1
        input_size_counts[input_size] += 1
    
    print(f"总样本数: {len(samples)}")
    
    print("\n增强类型分布:")
    for aug_type, count in sorted(aug_counts.items()):
        print(f"  {aug_type}: {count}")
    
    print("\n工具分布:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")
    
    print("\n输入规模分布:")
    for size, count in sorted(input_size_counts.items()):
        print(f"  {size}: {count}")


def print_sample_examples(samples: List[Dict], num_examples: int = 3):
    """打印样本示例"""
    print(f"\n=== 样本示例 (共 {num_examples} 个) ===\n")
    
    for i, sample in enumerate(samples[:num_examples], 1):
        print(f"--- 样本 {i} ---")
        print(f"增强类型: {sample['metadata']['augmentation_type']}")
        print(f"工具: {sample['metadata']['tool']}")
        print(f"\n输入:")
        print(sample['input'])
        print(f"\n输出:")
        print(sample['output'])
        print(f"\n延迟: {sample['metadata']['expected_latency_ms']}ms")
        print()


def main():
    parser = argparse.ArgumentParser(description='构造训练数据')
    parser.add_argument(
        '--registry',
        type=str,
        required=True,
        help='工具注册表 JSON 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--show_examples',
        type=int,
        default=3,
        help='显示样本示例数量 (默认: 3)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TOMAS-LLM 训练数据生成")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载注册表
    print(f"正在加载注册表: {args.registry}")
    registry = load_registry(args.registry)
    print(f"加载完成: {registry['metadata']['total_tokens']} tokens, "
          f"{registry['metadata']['tool_count']} tools\n")
    
    # 定义三种数据增强类型
    augmentation_types = ['full_description', 'resource_focused', 'performance_focused']
    
    # 生成所有样本
    print("正在生成训练样本...")
    all_samples = generate_all_samples(registry, augmentation_types)
    
    # 打印样本示例
    if args.show_examples > 0:
        print_sample_examples(all_samples, args.show_examples)
    
    # 打印统计信息
    print_statistics(all_samples, "训练集")
    
    # 保存数据 (全部作为训练集)
    output_dir = Path(args.output_dir)
    print(f"\n正在保存数据到 {output_dir}...")
    
    save_jsonl(all_samples, output_dir / 'train.jsonl')
    
    # 保存元数据
    metadata = {
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "registry_path": args.registry,
        "total_samples": len(all_samples),
        "train_samples": len(all_samples),
        "augmentation_types": augmentation_types,
        "num_tokens": registry['metadata']['total_tokens'],
        "num_tools": registry['metadata']['tool_count']
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"已保存元数据到 {metadata_path}")
    
    print("\n" + "=" * 60)
    print("数据生成完成!")
    print("=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
