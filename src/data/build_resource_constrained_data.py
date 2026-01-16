#!/usr/bin/env python3
"""
构造资源约束训练数据
生成基于系统资源约束和任务描述的训练数据，目标是在给定资源下选择 latency 最小的配置
"""

import json
import csv
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from .token_schema import format_token_for_model
    from .resource_binning import RESOURCE_BINS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from token_schema import format_token_for_model
    from resource_binning import RESOURCE_BINS


def load_registry(registry_path: str) -> Dict:
    """加载工具注册表"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    return registry


def load_profiling_data(profiling_path: str) -> List[Dict]:
    """加载 profiling 数据"""
    data = []
    with open(profiling_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'tool': row['tool'],
                'input_size': row['input_size'],
                'cpu_core': int(float(row['cpu_core'])),
                'cpu_mem_gb': float(row['cpu_mem_gb']),
                'gpu_sm': int(float(row['gpu_sm'])),
                'gpu_mem_gb': float(row['gpu_mem_gb']),
                'latency_ms': float(row['latency_ms'])
            })
    return data


def get_tool_semantic_description(tool_name: str, registry: Dict) -> str:
    """
    获取工具的语义描述
    
    Args:
        tool_name: 工具名称（如 'image_classification'）
        registry: 工具注册表
    
    Returns:
        工具的语义描述
    """
    # 从任意一个该工具的 token 中提取基础描述
    for token_name, token_info in registry['tokens'].items():
        if token_info['tool_name'] == tool_name:
            # 提取工具的基础描述（不包含资源配置部分）
            desc = token_info.get('semantic_description', token_info['description'])
            # 简化描述，只保留工具功能部分
            # 例如："Image Classification for small inputs with..." -> "Image Classification"
            base_desc = desc.split(' for ')[0] if ' for ' in desc else desc
            return base_desc
    
    # 如果找不到，返回格式化的工具名称
    return tool_name.replace('_', ' ').title()


def format_input_size_description(input_size: str) -> str:
    """格式化输入大小描述"""
    size_descriptions = {
        'small': 'small-scale input data',
        'medium': 'medium-scale input data',
        'large': 'large-scale input data'
    }
    return size_descriptions.get(input_size, f'{input_size} input data')


def value_to_level(value: float, resource_type: str) -> str:
    """将资源数值转换为级别描述"""
    bins = RESOURCE_BINS.get(resource_type, {})
    for level, bin_value in bins.items():
        if value <= bin_value:
            return level
    return 'high'


def generate_resource_constraint_templates() -> List[str]:
    """生成资源约束描述的模板"""
    templates = [
        # 完整资源约束描述
        "I need to run {task_desc} on {input_size}. My system has up to {cpu_cores} CPU cores, "
        "{cpu_mem}GB RAM, {gpu_sm} GPU SM units, and {gpu_mem}GB GPU memory available. "
        "Which configuration should I use for minimum latency?",
        
        "For {task_desc} task with {input_size}, my available resources are: "
        "{cpu_cores} CPU cores, {cpu_mem}GB memory, {gpu_sm} GPU streaming multiprocessors, "
        "and {gpu_mem}GB GPU memory. What's the optimal choice?",
        
        "I want to perform {task_desc} on {input_size}. "
        "System constraints: {cpu_cores} cores, {cpu_mem}GB RAM, {gpu_sm} SMs, {gpu_mem}GB VRAM. "
        "Recommend the fastest configuration.",
        
        "Task: {task_desc} | Input: {input_size} | "
        "Resources: CPU={cpu_cores} cores, RAM={cpu_mem}GB, GPU={gpu_sm} SMs, VRAM={gpu_mem}GB. "
        "Select optimal config for lowest latency.",
        
        "Running {task_desc} with {input_size}. Hardware limits: "
        "{cpu_cores}-core CPU, {cpu_mem}GB system memory, {gpu_sm} GPU compute units, "
        "{gpu_mem}GB video memory. Best configuration?",
        
        # 简化版本
        "Execute {task_desc} on {input_size} with max {cpu_cores} CPU cores and {gpu_mem}GB GPU memory.",
        
        "Perform {task_desc} ({input_size}). Available: {cpu_cores} cores, {gpu_sm} SMs, {gpu_mem}GB VRAM.",
        
        "{task_desc} task for {input_size}. System: {cpu_cores}C/{cpu_mem}G RAM, {gpu_sm}SM/{gpu_mem}G VRAM.",
        
        # 性能导向描述
        "Need fastest {task_desc} for {input_size} within these limits: "
        "CPU ≤ {cpu_cores} cores, Memory ≤ {cpu_mem}GB, GPU ≤ {gpu_sm} SMs, VRAM ≤ {gpu_mem}GB.",
        
        "Minimize latency for {task_desc} on {input_size}. "
        "Constraints: {cpu_cores} CPU cores, {cpu_mem}GB RAM, {gpu_sm} GPU SMs, {gpu_mem}GB VRAM.",
        
        # 资源级别描述
        "I have a system with {cpu_level} CPU ({cpu_cores} cores), {mem_level} RAM ({cpu_mem}GB), "
        "{gpu_level} GPU ({gpu_sm} SMs), and {vram_level} VRAM ({gpu_mem}GB). "
        "Optimize {task_desc} for {input_size}.",
        
        "For {task_desc} with {input_size}: "
        "CPU capacity: {cpu_level} ({cpu_cores} cores), "
        "Memory: {mem_level} ({cpu_mem}GB), "
        "GPU: {gpu_level} ({gpu_sm} SMs), "
        "VRAM: {vram_level} ({gpu_mem}GB). "
        "Choose optimal configuration.",
    ]
    return templates


def generate_input_description(
    profile_row: Dict,
    registry: Dict,
    template: str = None
) -> str:
    """
    生成输入描述（任务 + 资源约束）
    
    Args:
        profile_row: profiling 数据行
        registry: 工具注册表
        template: 使用的模板（可选，随机选择）
    
    Returns:
        输入描述字符串
    """
    # 获取任务描述
    task_desc = get_tool_semantic_description(profile_row['tool'], registry)
    input_size_desc = format_input_size_description(profile_row['input_size'])
    
    # 资源约束（使用当前配置的值）
    cpu_cores = profile_row['cpu_core']
    cpu_mem = profile_row['cpu_mem_gb']
    gpu_sm = profile_row['gpu_sm']
    gpu_mem = profile_row['gpu_mem_gb']
    
    # 获取资源级别描述
    cpu_level = value_to_level(cpu_cores, 'cpu_core')
    mem_level = value_to_level(cpu_mem, 'cpu_mem_gb')
    gpu_level = value_to_level(gpu_sm, 'gpu_sm')
    vram_level = value_to_level(gpu_mem, 'gpu_mem_gb')
    
    # 随机选择模板
    if template is None:
        templates = generate_resource_constraint_templates()
        template = random.choice(templates)
    
    # 填充模板
    input_text = template.format(
        task_desc=task_desc,
        input_size=input_size_desc,
        cpu_cores=cpu_cores,
        cpu_mem=cpu_mem,
        gpu_sm=gpu_sm,
        gpu_mem=gpu_mem,
        cpu_level=cpu_level,
        mem_level=mem_level,
        gpu_level=gpu_level,
        vram_level=vram_level
    )
    
    return input_text


def find_matching_token(
    profile_row: Dict,
    registry: Dict
) -> Tuple[str, Dict]:
    """
    找到与 profiling 数据匹配的 token
    
    Args:
        profile_row: profiling 数据行
        registry: 工具注册表
    
    Returns:
        (token_name, token_info)
    """
    # 构造匹配条件
    tool = profile_row['tool']
    input_size = profile_row['input_size']
    cpu = profile_row['cpu_core']
    cpu_mem = profile_row['cpu_mem_gb']
    gpu_sm = profile_row['gpu_sm']
    gpu_mem = profile_row['gpu_mem_gb']
    
    # 映射到资源级别
    cpu_level = value_to_level(cpu, 'cpu_core')
    mem_level = value_to_level(cpu_mem, 'cpu_mem_gb')
    gpu_level = value_to_level(gpu_sm, 'gpu_sm')
    vram_level = value_to_level(gpu_mem, 'gpu_mem_gb')
    
    # 映射输入大小
    size_map = {'small': 'SMALL', 'medium': 'MEDIUM', 'large': 'LARGE'}
    size_abbrev = size_map.get(input_size, input_size.upper())
    
    # 在 registry 中查找匹配的 token
    for token_name, token_info in registry['tokens'].items():
        if token_info['tool_name'] != tool:
            continue
        if token_info['input_size'] != input_size:
            continue
        
        # 检查资源配置是否匹配
        res = token_info['resources']
        res_levels = token_info['resource_levels']
        
        if (res['cpu_core'] == cpu and
            res['cpu_mem_gb'] == cpu_mem and
            res['gpu_sm'] == gpu_sm and
            res['gpu_mem_gb'] == gpu_mem):
            return token_name, token_info
    
    # 如果找不到精确匹配，返回 None
    return None, None


def generate_sample_from_profile(
    profile_row: Dict,
    registry: Dict,
    template: str = None
) -> Dict:
    """
    从 profiling 数据行生成训练样本
    
    Args:
        profile_row: profiling 数据行
        registry: 工具注册表
        template: 模板（可选）
    
    Returns:
        训练样本字典，如果找不到匹配的 token 则返回 None
    """
    # 找到匹配的 token
    token_name, token_info = find_matching_token(profile_row, registry)
    
    if token_name is None:
        return None
    
    # 生成输入描述
    input_text = generate_input_description(profile_row, registry, template)
    
    # 输出是格式化的 token
    output_token = format_token_for_model(token_name)
    
    # 构造样本
    sample = {
        "input": input_text,
        "output": output_token,
        "metadata": {
            "data_type": "resource_constrained",
            "expected_latency_ms": profile_row['latency_ms'],
            "tool": profile_row['tool'],
            "token_name": token_name,
            "input_size": profile_row['input_size'],
            "resources": {
                "cpu_core": profile_row['cpu_core'],
                "cpu_mem_gb": profile_row['cpu_mem_gb'],
                "gpu_sm": profile_row['gpu_sm'],
                "gpu_mem_gb": profile_row['gpu_mem_gb']
            },
            "resource_levels": token_info['resource_levels']
        }
    }
    
    return sample


def generate_all_samples(
    profiling_data: List[Dict],
    registry: Dict,
    samples_per_config: int = 1,
    tool_filter: str = None
) -> List[Dict]:
    """
    生成所有训练样本
    
    Args:
        profiling_data: profiling 数据列表
        registry: 工具注册表
        samples_per_config: 每个配置生成的样本数量
        tool_filter: 过滤工具名称（可选）
    
    Returns:
        所有样本列表
    """
    templates = generate_resource_constraint_templates()
    all_samples = []
    skipped = 0
    
    # 过滤数据
    if tool_filter:
        filtered_data = [row for row in profiling_data if row['tool'] == tool_filter]
        print(f"\n过滤工具: {tool_filter}")
        print(f"过滤后的配置: {len(filtered_data)} 个")
        profiling_data = filtered_data
        
        if not profiling_data:
            print(f"\n错误: 找不到工具 '{tool_filter}' 的数据")
            print(f"可用工具列表:")
            unique_tools = set(row['tool'] for row in profiling_data)
            for tool in sorted(unique_tools):
                print(f"  - {tool}")
            return []
    
    print(f"\n总共 {len(profiling_data)} 个配置")
    print(f"每个配置生成 {samples_per_config} 个样本")
    print(f"预计生成 {len(profiling_data) * samples_per_config} 个样本\n")
    
    for profile_row in profiling_data:
        # 为每个配置生成多个样本（使用不同模板）
        selected_templates = random.choices(templates, k=samples_per_config)
        
        for template in selected_templates:
            sample = generate_sample_from_profile(profile_row, registry, template)
            
            if sample is not None:
                all_samples.append(sample)
            else:
                skipped += 1
    
    if skipped > 0:
        print(f"\n警告: 跳过了 {skipped} 个无法匹配 token 的配置")
    
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
    
    tool_counts = defaultdict(int)
    input_size_counts = defaultdict(int)
    latency_stats = []
    
    for sample in samples:
        tool = sample['metadata']['tool']
        input_size = sample['metadata']['input_size']
        latency = sample['metadata']['expected_latency_ms']
        
        tool_counts[tool] += 1
        input_size_counts[input_size] += 1
        latency_stats.append(latency)
    
    print(f"总样本数: {len(samples)}")
    
    print("\n工具分布:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")
    
    print("\n输入规模分布:")
    for size, count in sorted(input_size_counts.items()):
        print(f"  {size}: {count}")
    
    if latency_stats:
        print(f"\n延迟统计:")
        print(f"  最小: {min(latency_stats):.1f}ms")
        print(f"  最大: {max(latency_stats):.1f}ms")
        print(f"  平均: {sum(latency_stats)/len(latency_stats):.1f}ms")


def print_sample_examples(samples: List[Dict], num_examples: int = 3):
    """打印样本示例"""
    print(f"\n=== 样本示例 (共 {num_examples} 个) ===\n")
    
    for i, sample in enumerate(samples[:num_examples], 1):
        print(f"--- 样本 {i} ---")
        print(f"工具: {sample['metadata']['tool']}")
        print(f"输入大小: {sample['metadata']['input_size']}")
        print(f"\n输入:")
        print(sample['input'])
        print(f"\n输出:")
        print(sample['output'])
        print(f"\n资源配置:")
        res = sample['metadata']['resources']
        print(f"  CPU: {res['cpu_core']} cores, {res['cpu_mem_gb']}GB RAM")
        print(f"  GPU: {res['gpu_sm']} SMs, {res['gpu_mem_gb']}GB VRAM")
        print(f"预期延迟: {sample['metadata']['expected_latency_ms']}ms")
        print()


def main():
    parser = argparse.ArgumentParser(description='构造资源约束训练数据')
    parser.add_argument(
        '--registry',
        type=str,
        required=True,
        help='工具注册表 JSON 文件路径'
    )
    parser.add_argument(
        '--profiling',
        type=str,
        required=True,
        help='Profiling CSV 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--samples_per_config',
        type=int,
        default=1,
        help='每个配置生成的样本数量 (默认: 1)'
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
    print("TOMAS-LLM 资源约束训练数据生成")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载注册表
    print(f"正在加载注册表: {args.registry}")
    registry = load_registry(args.registry)
    print(f"加载完成: {registry['metadata']['total_tokens']} tokens\n")
    
    # 加载 profiling 数据
    print(f"正在加载 profiling 数据: {args.profiling}")
    profiling_data = load_profiling_data(args.profiling)
    print(f"加载完成: {len(profiling_data)} 个配置\n")
    
    # 生成所有样本
    print("正在生成训练样本...")
    all_samples = generate_all_samples(
        profiling_data,
        registry,
        samples_per_config=args.samples_per_config,
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
    
    # 保存数据
    output_dir = Path(args.output_dir)
    print(f"\n正在保存数据到 {output_dir}...")
    
    save_jsonl(all_samples, output_dir / 'train_resource_constrained.jsonl')
    
    # 保存元数据
    metadata = {
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data_type": "resource_constrained",
        "registry_path": args.registry,
        "profiling_path": args.profiling,
        "total_samples": len(all_samples),
        "train_samples": len(all_samples),
        "samples_per_config": args.samples_per_config,
        "tool_filter": args.tool,
        "num_configs": len(profiling_data),
        "description": "训练数据基于系统资源约束和任务描述，目标是选择在给定资源下latency最小的配置"
    }
    
    metadata_path = output_dir / 'metadata_resource_constrained.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"已保存元数据到 {metadata_path}")
    
    print("\n" + "=" * 60)
    print("数据生成完成!")
    print("=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
