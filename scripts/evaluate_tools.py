#!/usr/bin/env python3
"""
验证训练结果脚本
加载训练数据，使用训练好的模型预测，对比结果并计算准确率
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def predict_token(model, tokenizer, input_text: str, max_new_tokens: int = 50, device: str = 'cuda') -> str:
    """
    使用模型预测输出 token
    
    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
        input_text: 输入文本
        max_new_tokens: 最大生成 token 数
        device: 设备
    
    Returns:
        预测的输出文本
    """
    # 构造完整的 prompt
    prompt = f"Input: {input_text}\nOutput: "
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪婪解码
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取输出部分（去掉 prompt）
    output_text = generated_text[len(prompt):].strip()
    
    # 只保留第一个 token（通常是虚拟 token）
    # 按空格或换行分割
    output_tokens = output_text.split()
    if output_tokens:
        return output_tokens[0]
    else:
        return output_text


def evaluate_model(
    model_path: str,
    data_path: str,
    output_path: str,
    num_samples: int = 100,
    device: str = 'cuda'
):
    """
    评估模型性能
    
    Args:
        model_path: 训练好的模型路径
        data_path: 数据文件路径 (JSONL)
        output_path: 输出结果路径 (JSON)
        num_samples: 评估样本数量
        device: 设备
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载模型和 tokenizer
    print(f"\n加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    # 加载数据
    print(f"加载数据: {data_path}")
    data = load_jsonl(data_path)
    
    # 限制样本数量
    if num_samples > 0 and num_samples < len(data):
        print(f"随机选择 {num_samples} 个样本进行评估")
        import random
        random.seed(42)
        data = random.sample(data, num_samples)
    else:
        num_samples = len(data)
    
    print(f"评估样本数: {num_samples}")
    
    # 评估
    results = []
    correct_count = 0
    
    print("\n开始预测...")
    for i, item in enumerate(tqdm(data, desc="预测进度")):
        input_text = item['input']
        ground_truth = item['output']
        
        # 预测
        try:
            prediction = predict_token(model, tokenizer, input_text, device=device)
        except Exception as e:
            print(f"\n预测失败 (样本 {i}): {e}")
            prediction = "[ERROR]"
        
        # 判断是否正确
        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_count += 1
        
        # 记录结果
        result = {
            "index": i,
            "input": input_text,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
            "metadata": item.get('metadata', {})
        }
        results.append(result)
    
    # 计算准确率
    accuracy = correct_count / num_samples if num_samples > 0 else 0.0
    
    # 按工具类型统计准确率
    tool_stats = {}
    for result in results:
        tool = result['metadata'].get('tool', 'unknown')
        if tool not in tool_stats:
            tool_stats[tool] = {'total': 0, 'correct': 0}
        tool_stats[tool]['total'] += 1
        if result['correct']:
            tool_stats[tool]['correct'] += 1
    
    # 计算每个工具的准确率
    for tool, stats in tool_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    # 按增强类型统计准确率
    aug_stats = {}
    for result in results:
        aug_type = result['metadata'].get('augmentation_type', 'unknown')
        if aug_type not in aug_stats:
            aug_stats[aug_type] = {'total': 0, 'correct': 0}
        aug_stats[aug_type]['total'] += 1
        if result['correct']:
            aug_stats[aug_type]['correct'] += 1
    
    # 计算每个增强类型的准确率
    for aug_type, stats in aug_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    # 汇总结果
    summary = {
        "model_path": model_path,
        "data_path": data_path,
        "total_samples": num_samples,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "tool_accuracy": tool_stats,
        "augmentation_accuracy": aug_stats,
        "predictions": results
    }
    
    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总样本数: {num_samples}")
    print(f"正确预测: {correct_count}")
    print(f"总体准确率: {accuracy:.2%}")
    
    print("\n按工具分类准确率:")
    for tool, stats in sorted(tool_stats.items()):
        print(f"  {tool}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\n按增强类型分类准确率:")
    for aug_type, stats in sorted(aug_stats.items()):
        print(f"  {aug_type}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # 打印一些错误案例
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\n错误案例 (前 5 个):")
        for i, error in enumerate(errors[:5], 1):
            print(f"\n  [{i}] 输入: {error['input'][:100]}...")
            print(f"      预期: {error['ground_truth']}")
            print(f"      预测: {error['prediction']}")
    
    print("\n" + "=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='训练好的模型路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/train.jsonl',
        help='数据文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='输出结果文件路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='评估样本数量 (0 表示全部)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == '__main__':
    main()
