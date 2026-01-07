"""
可视化同一工具不同资源配置的Embedding相似度

分析虚拟token的embedding，按工具类型分组，计算同一工具内不同资源配置的embedding相似度，
并用热力图展示。
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.token_schema import TOOL_ABBREV, parse_token_name, strip_token_brackets


def load_virtual_tokens_by_tool(tokenizer) -> Dict[str, List[Tuple[str, int]]]:
    """
    加载虚拟token并按工具类型分组
    
    Args:
        tokenizer: Tokenizer实例
    
    Returns:
        字典，key为工具缩写，value为[(token_name, token_id), ...]列表
    """
    virtual_tokens_by_tool = defaultdict(list)
    
    print("识别虚拟token...")
    vocab = tokenizer.get_vocab()
    
    for token, token_id in vocab.items():
        # 检查是否是虚拟token（以<开头，>结尾）
        if token.startswith('<') and token.endswith('>') and '_' in token:
            # 去掉尖括号
            token_name = strip_token_brackets(token)
            
            # 检查是否匹配任何工具缩写
            for tool_key, tool_abbrev in TOOL_ABBREV.items():
                if token_name.startswith(tool_abbrev):
                    virtual_tokens_by_tool[tool_abbrev].append((token_name, token_id))
                    break
    
    # 统计
    print("\n工具token统计:")
    for tool_abbrev, tokens in sorted(virtual_tokens_by_tool.items()):
        print(f"  {tool_abbrev}: {len(tokens)} tokens")
    
    return dict(virtual_tokens_by_tool)


def extract_embeddings(model, token_ids: List[int], device: str = 'cuda') -> torch.Tensor:
    """
    从模型中提取指定token的embedding
    
    Args:
        model: 模型实例
        token_ids: Token ID列表
        device: 计算设备
    
    Returns:
        Embedding矩阵 [num_tokens, embedding_dim]
    """
    # 获取embedding层
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Qwen2等模型
        embedding_layer = model.model.embed_tokens
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT-2等模型
        embedding_layer = model.transformer.wte
    elif hasattr(model, 'embeddings'):
        embedding_layer = model.embeddings
    else:
        raise ValueError("无法找到embedding层")
    
    # 提取embeddings
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    with torch.no_grad():
        embeddings = embedding_layer(token_ids_tensor)
    
    return embeddings.cpu().numpy()


def compute_similarity_matrix(embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """
    计算embedding相似度矩阵
    
    Args:
        embeddings: Embedding矩阵 [num_tokens, embedding_dim]
        metric: 相似度度量方式 ('cosine', 'euclidean')
    
    Returns:
        相似度矩阵 [num_tokens, num_tokens]
    """
    if metric == 'cosine':
        # 余弦相似度
        similarity = cosine_similarity(embeddings)
    elif metric == 'euclidean':
        # 欧氏距离（转换为相似度）
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
        # 归一化为相似度 (0-1范围)
        similarity = 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    token_names: List[str],
    tool_name: str,
    output_path: str,
    figsize: Tuple[int, int] = None
):
    """
    绘制相似度热力图
    
    Args:
        similarity_matrix: 相似度矩阵
        token_names: Token名称列表
        tool_name: 工具名称
        output_path: 输出文件路径
        figsize: 图像大小
    """
    # 动态调整图像大小
    n_tokens = len(token_names)
    if figsize is None:
        # 根据token数量自动调整大小
        size = max(10, min(30, n_tokens * 0.3))
        figsize = (size, size)
    
    plt.figure(figsize=figsize)
    
    # 创建热力图
    ax = sns.heatmap(
        similarity_matrix,
        xticklabels=token_names,
        yticklabels=token_names,
        cmap='RdYlBu_r',  # 红-黄-蓝反向（红表示高相似度）
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Cosine Similarity'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # 设置标题
    plt.title(f'Embedding Similarity Heatmap: {tool_name}\n({n_tokens} tokens)', 
              fontsize=16, pad=20)
    
    # 旋转x轴标签
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  保存热力图: {output_path}")
    
    plt.close()


def analyze_tool_embeddings(
    model,
    tokenizer,
    tool_abbrev: str,
    tokens_info: List[Tuple[str, int]],
    output_dir: Path,
    device: str = 'cuda'
) -> Dict:
    """
    分析特定工具的embedding相似度
    
    Args:
        model: 模型实例
        tokenizer: Tokenizer实例
        tool_abbrev: 工具缩写
        tokens_info: [(token_name, token_id), ...] 列表
        output_dir: 输出目录
        device: 计算设备
    
    Returns:
        分析结果字典
    """
    print(f"\n分析工具: {tool_abbrev}")
    print(f"Token数量: {len(tokens_info)}")
    
    # 提取token IDs和名称
    token_names = [name for name, _ in tokens_info]
    token_ids = [tid for _, tid in tokens_info]
    
    # 提取embeddings
    print("  提取embeddings...")
    embeddings = extract_embeddings(model, token_ids, device)
    print(f"  Embedding shape: {embeddings.shape}")
    
    # 计算相似度矩阵
    print("  计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(embeddings, metric='cosine')
    
    # 统计信息
    # 排除对角线（自身相似度为1）
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[mask]
    
    stats = {
        'tool': tool_abbrev,
        'num_tokens': len(tokens_info),
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities))
    }
    
    print(f"  平均相似度: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}")
    print(f"  相似度范围: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
    
    # 找到最相似和最不相似的token对
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    upper_triangle_values = similarity_matrix[upper_triangle_indices]
    
    max_idx = np.argmax(upper_triangle_values)
    min_idx = np.argmin(upper_triangle_values)
    
    max_pair = (
        token_names[upper_triangle_indices[0][max_idx]],
        token_names[upper_triangle_indices[1][max_idx]],
        float(upper_triangle_values[max_idx])
    )
    
    min_pair = (
        token_names[upper_triangle_indices[0][min_idx]],
        token_names[upper_triangle_indices[1][min_idx]],
        float(upper_triangle_values[min_idx])
    )
    
    stats['most_similar_pair'] = {
        'token1': max_pair[0],
        'token2': max_pair[1],
        'similarity': max_pair[2]
    }
    
    stats['least_similar_pair'] = {
        'token1': min_pair[0],
        'token2': min_pair[1],
        'similarity': min_pair[2]
    }
    
    print(f"  最相似的token对: {max_pair[0]} <-> {max_pair[1]} ({max_pair[2]:.4f})")
    print(f"  最不相似的token对: {min_pair[0]} <-> {min_pair[1]} ({min_pair[2]:.4f})")
    
    # 绘制热力图
    print("  生成热力图...")
    heatmap_path = output_dir / f"{tool_abbrev}_similarity_heatmap.png"
    plot_similarity_heatmap(
        similarity_matrix,
        token_names,
        tool_abbrev,
        str(heatmap_path)
    )
    
    # 保存相似度矩阵
    matrix_path = output_dir / f"{tool_abbrev}_similarity_matrix.npy"
    np.save(matrix_path, similarity_matrix)
    print(f"  保存相似度矩阵: {matrix_path}")
    
    # 保存token映射
    mapping_path = output_dir / f"{tool_abbrev}_token_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tokens': [{'name': name, 'id': tid} for name, tid in tokens_info]
        }, f, indent=2)
    
    return stats


def plot_overall_statistics(all_stats: List[Dict], output_path: str):
    """
    绘制所有工具的相似度统计对比图
    
    Args:
        all_stats: 所有工具的统计信息列表
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 提取数据
    tools = [s['tool'] for s in all_stats]
    means = [s['mean_similarity'] for s in all_stats]
    stds = [s['std_similarity'] for s in all_stats]
    mins = [s['min_similarity'] for s in all_stats]
    maxs = [s['max_similarity'] for s in all_stats]
    
    # 子图1: 平均相似度（带误差棒）
    ax1 = axes[0]
    x = np.arange(len(tools))
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Tool', fontsize=12)
    ax1.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax1.set_title('Average Embedding Similarity by Tool', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tools, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 子图2: 相似度范围（箱线图风格）
    ax2 = axes[1]
    for i, (tool, min_val, max_val, mean_val) in enumerate(zip(tools, mins, maxs, means)):
        ax2.plot([i, i], [min_val, max_val], 'k-', linewidth=2)
        ax2.plot(i, min_val, 'rv', markersize=8, label='Min' if i == 0 else '')
        ax2.plot(i, max_val, 'g^', markersize=8, label='Max' if i == 0 else '')
        ax2.plot(i, mean_val, 'bo', markersize=8, label='Mean' if i == 0 else '')
    
    ax2.set_xlabel('Tool', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Similarity Range by Tool', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tools, rotation=45, ha='right')
    ax2.legend(loc='best')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n保存统计对比图: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化虚拟token embedding相似度')
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/model_initialized',
        help='模型路径'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='checkpoints/tokenizer_expanded',
        help='Tokenizer路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/embedding_similarity',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='计算设备'
    )
    parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        help='指定要分析的工具（默认全部）'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("虚拟Token Embedding相似度分析")
    print("=" * 60)
    
    # 加载tokenizer
    print(f"\n加载Tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"词汇表大小: {len(tokenizer)}")
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto' if args.device == 'cuda' else None
    )
    
    if args.device == 'cpu':
        model = model.to(args.device)
    
    model.eval()
    print(f"模型加载完成，使用设备: {args.device}")
    
    # 加载虚拟tokens并按工具分组
    virtual_tokens_by_tool = load_virtual_tokens_by_tool(tokenizer)
    
    # 筛选要分析的工具
    if args.tools:
        virtual_tokens_by_tool = {
            k: v for k, v in virtual_tokens_by_tool.items()
            if k in args.tools
        }
    
    if not virtual_tokens_by_tool:
        print("\n错误: 没有找到要分析的工具")
        return
    
    # 分析每个工具
    all_stats = []
    
    for tool_abbrev, tokens_info in sorted(virtual_tokens_by_tool.items()):
        stats = analyze_tool_embeddings(
            model=model,
            tokenizer=tokenizer,
            tool_abbrev=tool_abbrev,
            tokens_info=tokens_info,
            output_dir=output_dir,
            device=args.device
        )
        all_stats.append(stats)
    
    # 保存所有统计信息
    stats_path = output_dir / 'all_tools_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n保存统计信息: {stats_path}")
    
    # 绘制对比图
    if len(all_stats) > 1:
        plot_overall_statistics(
            all_stats,
            str(output_dir / 'tools_comparison.png')
        )
    
    # 打印总结
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  - 每个工具的热力图: *_similarity_heatmap.png")
    print("  - 相似度矩阵: *_similarity_matrix.npy")
    print("  - Token映射: *_token_mapping.json")
    print("  - 统计信息: all_tools_statistics.json")
    if len(all_stats) > 1:
        print("  - 工具对比图: tools_comparison.png")
    
    print("\n各工具平均相似度:")
    for stats in sorted(all_stats, key=lambda x: x['mean_similarity'], reverse=True):
        print(f"  {stats['tool']}: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}")


if __name__ == '__main__':
    main()
