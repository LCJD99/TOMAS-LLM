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
    from .token_schema import format_token_for_model
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
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


def generate_all_samples(
    registry: Dict,
    samples_per_token: int = 15,
    tool_filter: str = None
) -> List[Dict]:
    """
    生成所有训练样本
    
    Args:
        registry: 工具注册表
        samples_per_token: 每个token生成的样本数量
        tool_filter: 过滤工具名称（只生成该工具的数据，可选）
    
    Returns:
        所有样本列表
    """
    from .templates import (
        FULL_DESCRIPTION_TEMPLATES,
        RESOURCE_FOCUSED_TEMPLATES,
        PERFORMANCE_FOCUSED_TEMPLATES
    )
    
    # 收集所有模板
    all_templates = []
    for template in FULL_DESCRIPTION_TEMPLATES:
        all_templates.append((template, 'full_description'))
    for template in RESOURCE_FOCUSED_TEMPLATES:
        all_templates.append((template, 'resource_focused'))
    for template in PERFORMANCE_FOCUSED_TEMPLATES:
        all_templates.append((template, 'performance_focused'))
    
    print(f"可用模板总数: {len(all_templates)}")
    print(f"  - Full Description: {len(FULL_DESCRIPTION_TEMPLATES)}")
    print(f"  - Resource Focused: {len(RESOURCE_FOCUSED_TEMPLATES)}")
    print(f"  - Performance Focused: {len(PERFORMANCE_FOCUSED_TEMPLATES)}")
    
    all_samples = []
    tokens = registry['tokens']
    
    # 过滤token（如果指定了工具）
    if tool_filter:
        filtered_tokens = {k: v for k, v in tokens.items() 
                          if v['tool_name'] == tool_filter}
        print(f"\n过滤工具: {tool_filter}")
        print(f"过滤后的tokens: {len(filtered_tokens)} 个")
        tokens = filtered_tokens
        
        if not tokens:
            print(f"\n错误: 找不到工具 '{tool_filter}' 的tokens")
            print(f"可用工具列表:")
            unique_tools = set(v['tool_name'] for v in registry['tokens'].values())
            for tool in sorted(unique_tools):
                print(f"  - {tool}")
            return []
    
    print(f"\n总共 {len(tokens)} 个 tokens")
    print(f"每个 token 生成 {samples_per_token} 个样本")
    print(f"预计生成 {len(tokens) * samples_per_token} 个样本\n")
    
    for token_name, token_info in tokens.items():
        # 为每个token随机选择samples_per_token个模板
        if samples_per_token >= len(all_templates):
            # 如果需要的样本数大于模板数，允许重复
            selected_templates = random.choices(all_templates, k=samples_per_token)
        else:
            # 否则不重复选择
            selected_templates = random.sample(all_templates, k=samples_per_token)
        
        for template, template_type in selected_templates:
            sample = generate_instruction_sample(
                token_name, 
                token_info, 
                template, 
                template_type
            )
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
    
    # 按模板类型统计
    template_counts = defaultdict(int)
    tool_counts = defaultdict(int)
    input_size_counts = defaultdict(int)
    
    for sample in samples:
        template_type = sample['metadata']['template_type']
        tool = sample['metadata']['tool']
        input_size = sample['metadata']['input_size']
        
        template_counts[template_type] += 1
        tool_counts[tool] += 1
        input_size_counts[input_size] += 1
    
    print(f"总样本数: {len(samples)}")
    
    print("\n模板类型分布:")
    for template_type, count in sorted(template_counts.items()):
        print(f"  {template_type}: {count}")
    
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
        print(f"模板类型: {sample['metadata']['template_type']}")
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
        '--samples_per_token',
        type=int,
        default=15,
        help='每个token生成的样本数量 (默认: 15)'
    )
    parser.add_argument(
        '--tool',
        type=str,
        default=None,
        help='指定工具名称，只生成该工具的数据 (可选)'
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
    
    # 生成所有样本
    print("正在生成训练样本...")
    all_samples = generate_all_samples(
        registry, 
        samples_per_token=args.samples_per_token,
        tool_filter=args.tool
    )
    
    if not all_samples:
        print("\n错误: 没有生成任何样本")
        return
    
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
        "samples_per_token": args.samples_per_token,
        "tool_filter": args.tool,
        "num_tokens": len(registry['tokens']) if not args.tool else len([k for k, v in registry['tokens'].items() if v['tool_name'] == args.tool]),
        "num_tools": 1 if args.tool else registry['metadata']['tool_count']
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
