"""
推理工具：使用训练好的模型进行预测
限制只从扩充的虚拟token中采样
"""

import json
import torch
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from src.data.token_schema import TOOL_ABBREV


class VirtualTokenOnlyLogitsProcessor(LogitsProcessor):
    """
    限制模型只能从虚拟token中采样的Logits处理器
    """
    
    def __init__(self, virtual_token_ids: List[int]):
        """
        Args:
            virtual_token_ids: 允许采样的虚拟token ID列表
        """
        self.virtual_token_ids = set(virtual_token_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        将非虚拟token的logits设为负无穷，强制只从虚拟token中采样
        
        Args:
            input_ids: 输入token IDs
            scores: 模型输出的logits [batch_size, vocab_size]
        
        Returns:
            处理后的logits，只保留虚拟token的分数
        """
        # 创建mask，只保留虚拟token的logits
        mask = torch.ones_like(scores, dtype=torch.bool)
        for token_id in self.virtual_token_ids:
            mask[:, token_id] = False
        
        # 将非虚拟token的logits设为负无穷
        scores[mask] = -float('inf')
        
        return scores


def get_virtual_token_ids(tokenizer) -> List[int]:
    """
    获取所有虚拟token的ID
    假设虚拟token格式为 <TOOL_...>，例如 <IMG_CLS_SMALL_LOW_MED_HIGH_LOW>
    
    Args:
        tokenizer: Tokenizer实例
    
    Returns:
        虚拟token ID列表
    """
    virtual_token_ids = []
    
    for token, token_id in tokenizer.get_vocab().items():
        # 检查是否是虚拟token（以 < 开头，以 > 结尾，包含下划线）
        if token.startswith('<') and token.endswith('>') and '_' in token:
            # 进一步检查是否包含工具名称关键字
            tool_keywords = TOOL_ABBREV.values()
            if any(keyword in token for keyword in tool_keywords):
                virtual_token_ids.append(token_id)
    
    print(f"找到 {len(virtual_token_ids)} 个虚拟token")
    return virtual_token_ids


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载JSONL文件
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


class ToolPredictor:
    """
    工具预测器：加载模型并执行推理
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        torch_dtype=torch.bfloat16,
        constrained: bool = True
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            device: 计算设备
            torch_dtype: 数据类型
            constrained: 是否使用约束生成（只从虚拟token采样）
        """
        print(f"加载模型: {model_path}")
        self.device = device
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map='auto' if device == 'cuda' else None
        )
        
        if device == 'cpu':
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # 保存约束配置
        self.constrained = constrained
        
        # 获取虚拟token ID列表
        if constrained:
            print("识别虚拟token...")
            self.virtual_token_ids = get_virtual_token_ids(self.tokenizer)
            
            # 创建logits processor
            self.logits_processor = LogitsProcessorList([
                VirtualTokenOnlyLogitsProcessor(self.virtual_token_ids)
            ])
            print(f"模型加载完成，使用设备: {device}（约束生成模式）")
        else:
            self.virtual_token_ids = []
            self.logits_processor = None
            print(f"模型加载完成，使用设备: {device}（无约束生成模式）")
    
    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 50,
        constrained: Optional[bool] = None
    ) -> str:
        """
        预测工具token
        
        Args:
            input_text: 输入文本（任务描述）
            max_new_tokens: 最大生成token数（默认1）
            temperature: 采样温度
            do_sample: 是否使用采样（False为贪婪解码）
            top_p: Nucleus采样参数
            top_k: Top-K采样参数
            constrained: 是否使用约束生成（None则使用初始化时的设置）
        
        Returns:
            预测的token字符串
        """
        # 构造prompt
        prompt = f"Input: {input_text}\nOutput: "
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # 决定是否使用约束
        use_constrained = constrained if constrained is not None else self.constrained
        
        # Generate
        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'temperature': temperature if do_sample else 1.0,
            'top_p': top_p if do_sample else 1.0,
            'top_k': top_k if do_sample else 50,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        # 如果使用约束，添加logits_processor
        if use_constrained and self.logits_processor is not None:
            generate_kwargs['logits_processor'] = self.logits_processor
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取输出部分（去掉prompt）
        output_text = generated_text[len(prompt):].strip()
        
        # 提取第一个token（通常是虚拟token）
        output_tokens = output_text.split()
        if output_tokens:
            predicted_token = output_tokens[0]
        else:
            predicted_token = output_text
        
        return predicted_token
    
    def predict_batch(
        self,
        input_texts: List[str],
        **kwargs
    ) -> List[str]:
        """
        批量预测
        
        Args:
            input_texts: 输入文本列表
            **kwargs: predict方法的其他参数
        
        Returns:
            预测的虚拟token列表
        """
        predictions = []
        for text in input_texts:
            pred = self.predict(text, **kwargs)
            predictions.append(pred)
        return predictions
    
    def predict_with_scores(
        self,
        input_text: str,
        top_k: int = 5,
        constrained: Optional[bool] = None
    ) -> List[tuple]:
        """
        预测并返回top-k个候选token及其分数
        
        Args:
            input_text: 输入文本
            top_k: 返回前k个候选
            constrained: 是否使用约束生成（None则使用初始化时的设置）
        
        Returns:
            [(token, score), ...] 列表
        """
        # 构造prompt
        prompt = f"Input: {input_text}\nOutput: "
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # 取最后一个位置的logits
        
        # 决定是否使用约束
        use_constrained = constrained if constrained is not None else self.constrained
        
        # 应用logits processor（如果使用约束）
        if use_constrained and self.logits_processor is not None:
            processed_logits = self.logits_processor[0](inputs.input_ids, logits)
            max_k = min(top_k, len(self.virtual_token_ids))
        else:
            processed_logits = logits
            max_k = top_k
        
        # 计算概率
        probs = torch.softmax(processed_logits, dim=-1)
        
        # 获取top-k
        top_probs, top_indices = torch.topk(probs[0], k=max_k)
        
        # 解码token
        results = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()])
            results.append((token, prob.item()))
        
        return results
    
    def evaluate_on_data(
        self,
        data_path: str,
        output_path: str,
        num_samples: Optional[int] = None,
        **predict_kwargs
    ) -> Dict:
        """
        在训练数据上评估模型并输出结果
        
        Args:
            data_path: 训练数据路径（JSONL格式）
            output_path: 输出JSON文件路径
            num_samples: 评估样本数量（None表示全部）
            **predict_kwargs: predict方法的其他参数
        
        Returns:
            评估结果字典
        """
        print(f"\n加载数据: {data_path}")
        data = load_jsonl(data_path)
        
        if num_samples is not None:
            import random
            if num_samples < len(data):
                data = random.sample(data, num_samples)
        
        print(f"评估样本数: {len(data)}")
        
        # 评估
        results = []
        correct = 0
        total = len(data)
        
        # 统计指标
        tool_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        augmentation_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        print("\n开始预测...")
        for item in tqdm(data, desc="评估进度"):
            input_text = item['input']
            ground_truth = item['output']
            
            # 预测
            try:
                prediction = self.predict(input_text, **predict_kwargs)
            except Exception as e:
                print(f"\n预测失败: {e}")
                prediction = "[ERROR]"
            
            # 判断正确性
            is_correct = (prediction == ground_truth)
            if is_correct:
                correct += 1
            
            # 提取工具名称和增强类型（如果存在）
            tool_name = item.get('tool', 'unknown')
            augmentation_type = item.get('augmentation_type', 'unknown')
            
            # 更新统计
            tool_stats[tool_name]['total'] += 1
            if is_correct:
                tool_stats[tool_name]['correct'] += 1
            
            augmentation_stats[augmentation_type]['total'] += 1
            if is_correct:
                augmentation_stats[augmentation_type]['correct'] += 1
            
            # 记录结果
            results.append({
                'input': input_text,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': is_correct,
                'tool': tool_name,
                'augmentation_type': augmentation_type
            })
        
        # 计算准确率
        overall_accuracy = correct / total if total > 0 else 0.0
        
        # 计算各工具准确率
        tool_accuracy = {}
        for tool, stats in tool_stats.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            tool_accuracy[tool] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        # 计算各增强类型准确率
        augmentation_accuracy = {}
        for aug_type, stats in augmentation_stats.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            augmentation_accuracy[aug_type] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        # 构造输出
        output_data = {
            'summary': {
                'total_samples': total,
                'correct_predictions': correct,
                'overall_accuracy': overall_accuracy
            },
            'tool_accuracy': tool_accuracy,
            'augmentation_accuracy': augmentation_accuracy,
            'predictions': results
        }
        
        # 保存到文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估完成！")
        print(f"总样本数: {total}")
        print(f"正确预测: {correct}")
        print(f"总体准确率: {overall_accuracy:.2%}")
        print(f"\n结果已保存到: {output_path}")
        
        # 显示错误案例（前5个）
        error_cases = [r for r in results if not r['correct']]
        if error_cases:
            print(f"\n错误案例数: {len(error_cases)}")
            print("\n前5个错误案例:")
            for i, case in enumerate(error_cases[:5], 1):
                print(f"\n{i}. 输入: {case['input'][:100]}...")
                print(f"   Ground Truth: {case['ground_truth']}")
                print(f"   Prediction: {case['prediction']}")
        
        return output_data


def main():
    """
    使用示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='工具预测推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'evaluate'], 
                       help='运行模式: single=单条预测, evaluate=批量评估')
    parser.add_argument('--constrained', action='store_true', default=True, 
                       help='使用约束生成（只从虚拟token采样）')
    parser.add_argument('--no-constrained', dest='constrained', action='store_false',
                       help='使用无约束生成')
    
    # 单条预测模式参数
    parser.add_argument('--input', type=str, help='输入任务描述（单条预测模式）')
    parser.add_argument('--top_k', type=int, default=5, help='显示top-k候选')
    parser.add_argument('--sample', action='store_true', help='使用采样而非贪婪解码')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--max_tokens', type=int, default=1, help='最大生成token数')
    
    # 批量评估模式参数
    parser.add_argument('--data', type=str, help='训练数据路径（评估模式）')
    parser.add_argument('--output', type=str, help='输出JSON文件路径（评估模式）')
    parser.add_argument('--num_samples', type=int, help='评估样本数（None=全部）')
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = ToolPredictor(args.model, constrained=args.constrained)
    
    if args.mode == 'single':
        # 单条预测模式
        if not args.input:
            parser.error("单条预测模式需要 --input 参数")
        
        print(f"\n输入: {args.input}")
        print("-" * 60)
        
        # 预测
        prediction = predictor.predict(
            args.input,
            max_new_tokens=args.max_tokens,
            do_sample=args.sample,
            temperature=args.temperature
        )
        print(f"预测结果: {prediction}")
        
        # 显示top-k候选
        print(f"\nTop-{args.top_k} 候选:")
        candidates = predictor.predict_with_scores(args.input, top_k=args.top_k)
        for i, (token, score) in enumerate(candidates, 1):
            print(f"{i}. {token:30s} (score: {score:.4f})")
    
    elif args.mode == 'evaluate':
        # 批量评估模式
        if not args.data or not args.output:
            parser.error("评估模式需要 --data 和 --output 参数")
        
        predictor.evaluate_on_data(
            data_path=args.data,
            output_path=args.output,
            num_samples=args.num_samples,
            max_new_tokens=args.max_tokens,
            do_sample=args.sample,
            temperature=args.temperature
        )


if __name__ == '__main__':
    main()
