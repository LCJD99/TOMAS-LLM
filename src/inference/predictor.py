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


def parse_virtual_token(token: str) -> Dict[str, str]:
    """
    解析虚拟token的各个字段
    格式: <TOOL_SIZE_CPU_CPUMEM_GPU_GPUMEM>
    例如: <IMG_CLS_SMALL_LOW_MED_HIGH_LOW>
    
    Args:
        token: 虚拟token字符串
    
    Returns:
        字段字典，包含 tool, size, cpu_core, cpu_mem, gpu_sm, gpu_mem
        如果解析失败，返回空字典
    """
    if not token.startswith('<') or not token.endswith('>'):
        return {}
    
    # 去掉尖括号
    token_content = token[1:-1]
    
    # 按下划线分割
    parts = token_content.split('_')
    
    # 虚拟token应该有6个部分：TOOL, SIZE, CPU, CPUMEM, GPU, GPUMEM
    if len(parts) != 6:
        return {}
    
    return {
        'tool': parts[0],
        'size': parts[1],
        'cpu_core': parts[2],
        'cpu_mem': parts[3],
        'gpu_sm': parts[4],
        'gpu_mem': parts[5]
    }


def compare_token_fields(predicted: str, ground_truth: str) -> Dict[str, any]:
    """
    比较预测token和ground truth token的各个字段
    
    Args:
        predicted: 预测的token
        ground_truth: ground truth token
    
    Returns:
        比较结果字典，包含各字段的匹配情况和总体匹配率
    """
    pred_fields = parse_virtual_token(predicted)
    gt_fields = parse_virtual_token(ground_truth)
    
    # 如果任一解析失败，返回错误信息
    if not pred_fields or not gt_fields:
        return {
            'exact_match': predicted == ground_truth,
            'field_matches': {},
            'match_rate': 1.0 if predicted == ground_truth else 0.0,
            'matched_fields': 0,
            'total_fields': 0,
            'parse_error': True
        }
    
    # 比较各个字段
    field_matches = {}
    matched_count = 0
    
    for field_name in ['tool', 'size', 'cpu_core', 'cpu_mem', 'gpu_sm', 'gpu_mem']:
        is_match = pred_fields[field_name] == gt_fields[field_name]
        field_matches[field_name] = is_match
        if is_match:
            matched_count += 1
    
    total_fields = len(field_matches)
    match_rate = matched_count / total_fields if total_fields > 0 else 0.0
    
    return {
        'exact_match': predicted == ground_truth,
        'field_matches': field_matches,
        'match_rate': match_rate,
        'matched_fields': matched_count,
        'total_fields': total_fields,
        'parse_error': False,
        'predicted_fields': pred_fields,
        'ground_truth_fields': gt_fields
    }


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
        exact_match_count = 0
        total = len(data)
        
        # 字段匹配统计
        field_names = ['tool', 'size', 'cpu_core', 'cpu_mem', 'gpu_sm', 'gpu_mem']
        overall_field_matches = {field: 0 for field in field_names}
        overall_match_rate_sum = 0.0
        
        # 统计指标
        tool_stats = defaultdict(lambda: {'exact_match': 0, 'total': 0, 'match_rate_sum': 0.0})
        augmentation_stats = defaultdict(lambda: {'exact_match': 0, 'total': 0, 'match_rate_sum': 0.0})
        
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
            
            # 比较字段匹配情况
            comparison = compare_token_fields(prediction, ground_truth)
            
            # 统计exact match
            if comparison['exact_match']:
                exact_match_count += 1
            
            # 累计字段匹配数
            if not comparison['parse_error']:
                for field in field_names:
                    if comparison['field_matches'].get(field, False):
                        overall_field_matches[field] += 1
                overall_match_rate_sum += comparison['match_rate']
            
            # 提取工具名称和增强类型（如果存在）
            tool_name = item.get('tool', 'unknown')
            augmentation_type = item.get('augmentation_type', 'unknown')
            
            # 更新统计
            tool_stats[tool_name]['total'] += 1
            if comparison['exact_match']:
                tool_stats[tool_name]['exact_match'] += 1
            if not comparison['parse_error']:
                tool_stats[tool_name]['match_rate_sum'] += comparison['match_rate']
            
            augmentation_stats[augmentation_type]['total'] += 1
            if comparison['exact_match']:
                augmentation_stats[augmentation_type]['exact_match'] += 1
            if not comparison['parse_error']:
                augmentation_stats[augmentation_type]['match_rate_sum'] += comparison['match_rate']
            
            # 记录结果
            results.append({
                'input': input_text,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'exact_match': comparison['exact_match'],
                'match_rate': comparison['match_rate'],
                'matched_fields': comparison['matched_fields'],
                'total_fields': comparison['total_fields'],
                'field_matches': comparison['field_matches'],
                'tool': tool_name,
                'augmentation_type': augmentation_type
            })
        
        # 计算总体指标
        exact_match_rate = exact_match_count / total if total > 0 else 0.0
        overall_avg_match_rate = overall_match_rate_sum / total if total > 0 else 0.0
        
        # 计算各字段匹配率
        field_match_rates = {}
        for field in field_names:
            field_match_rates[field] = overall_field_matches[field] / total if total > 0 else 0.0
        
        # 计算各工具准确率
        tool_accuracy = {}
        for tool, stats in tool_stats.items():
            exact_match_acc = stats['exact_match'] / stats['total'] if stats['total'] > 0 else 0.0
            avg_match_rate = stats['match_rate_sum'] / stats['total'] if stats['total'] > 0 else 0.0
            tool_accuracy[tool] = {
                'exact_match_rate': exact_match_acc,
                'avg_match_rate': avg_match_rate,
                'exact_matches': stats['exact_match'],
                'total': stats['total']
            }
        
        # 计算各增强类型准确率
        augmentation_accuracy = {}
        for aug_type, stats in augmentation_stats.items():
            exact_match_acc = stats['exact_match'] / stats['total'] if stats['total'] > 0 else 0.0
            avg_match_rate = stats['match_rate_sum'] / stats['total'] if stats['total'] > 0 else 0.0
            augmentation_accuracy[aug_type] = {
                'exact_match_rate': exact_match_acc,
                'avg_match_rate': avg_match_rate,
                'exact_matches': stats['exact_match'],
                'total': stats['total']
            }
        
        # 构造输出
        output_data = {
            'summary': {
                'total_samples': total,
                'exact_matches': exact_match_count,
                'exact_match_rate': exact_match_rate,
                'overall_avg_match_rate': overall_avg_match_rate,
                'field_match_rates': field_match_rates
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
        print(f"完全匹配数: {exact_match_count}")
        print(f"完全匹配率: {exact_match_rate:.2%}")
        print(f"平均字段匹配率: {overall_avg_match_rate:.2%}")
        print(f"\n各字段匹配率:")
        for field, rate in field_match_rates.items():
            print(f"  {field:12s}: {rate:.2%}")
        print(f"\n结果已保存到: {output_path}")
        
        # 显示错误案例（前5个）
        error_cases = [r for r in results if not r['exact_match']]
        if error_cases:
            print(f"\n非完全匹配案例数: {len(error_cases)}")
            print("\n前5个非完全匹配案例:")
            for i, case in enumerate(error_cases[:5], 1):
                print(f"\n{i}. 输入: {case['input'][:100]}...")
                print(f"   Ground Truth: {case['ground_truth']}")
                print(f"   Prediction:   {case['prediction']}")
                print(f"   字段匹配率: {case['match_rate']:.1%} ({case['matched_fields']}/{case['total_fields']})")
                if case.get('field_matches'):
                    mismatched = [f for f, matched in case['field_matches'].items() if not matched]
                    if mismatched:
                        print(f"   不匹配字段: {', '.join(mismatched)}")
        
        return output_data
    
    def compare_tokenization(
        self,
        data_path: str,
        dataset_tokenizer_path: Optional[str] = None,
        num_samples: int = 10,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        比较predictor的tokenizer和dataset使用的tokenizer的token ID差异
        
        Args:
            data_path: 训练数据路径（JSONL格式）
            dataset_tokenizer_path: dataset使用的tokenizer路径（None则使用相同的）
            num_samples: 比较的样本数量
            output_path: 输出结果的JSON文件路径（可选）
        
        Returns:
            比较结果字典
        """
        print(f"\n=== Tokenizer比较分析 ===")
        print(f"Predictor tokenizer: {self.tokenizer.name_or_path}")
        
        # 加载dataset tokenizer（如果指定了不同的路径）
        if dataset_tokenizer_path and dataset_tokenizer_path != self.tokenizer.name_or_path:
            print(f"Dataset tokenizer: {dataset_tokenizer_path}")
            from transformers import AutoTokenizer
            dataset_tokenizer = AutoTokenizer.from_pretrained(
                dataset_tokenizer_path,
                trust_remote_code=True
            )
        else:
            print("Dataset tokenizer: 使用相同的tokenizer")
            dataset_tokenizer = self.tokenizer
        
        # 加载数据
        print(f"\n加载数据: {data_path}")
        data = load_jsonl(data_path)
        
        # 随机采样
        import random
        if num_samples < len(data):
            samples = random.sample(data, num_samples)
        else:
            samples = data[:num_samples]
        
        print(f"分析样本数: {len(samples)}\n")
        
        # 比较结果
        comparison_results = []
        total_differences = 0
        
        for idx, item in enumerate(samples, 1):
            input_text = item['input']
            output_text = item['output']
            
            # 构造完整文本（按照dataset的格式）
            input_part = f"Input: {input_text}\nOutput: "
            full_text = input_part + output_text
            
            # Predictor tokenizer编码
            predictor_input_encoding = self.tokenizer(
                input_part,
                add_special_tokens=True,
                return_tensors='pt'
            )
            predictor_full_encoding = self.tokenizer(
                full_text,
                add_special_tokens=True,
                return_tensors='pt'
            )
            
            # Dataset tokenizer编码
            dataset_input_encoding = dataset_tokenizer(
                input_part,
                add_special_tokens=True,
                return_tensors='pt'
            )
            dataset_full_encoding = dataset_tokenizer(
                full_text,
                add_special_tokens=True,
                return_tensors='pt'
            )
            
            # 提取token IDs
            predictor_input_ids = predictor_input_encoding['input_ids'][0].tolist()
            predictor_full_ids = predictor_full_encoding['input_ids'][0].tolist()
            dataset_input_ids = dataset_input_encoding['input_ids'][0].tolist()
            dataset_full_ids = dataset_full_encoding['input_ids'][0].tolist()
            
            # 解码查看差异
            predictor_input_tokens = [self.tokenizer.decode([tid]) for tid in predictor_input_ids]
            predictor_full_tokens = [self.tokenizer.decode([tid]) for tid in predictor_full_ids]
            dataset_input_tokens = [dataset_tokenizer.decode([tid]) for tid in dataset_input_ids]
            dataset_full_tokens = [dataset_tokenizer.decode([tid]) for tid in dataset_full_ids]
            
            # 检查差异
            input_ids_match = (predictor_input_ids == dataset_input_ids)
            full_ids_match = (predictor_full_ids == dataset_full_ids)
            
            if not input_ids_match or not full_ids_match:
                total_differences += 1
            
            # 详细比较
            input_length_predictor = len(predictor_input_ids)
            input_length_dataset = len(dataset_input_ids)
            full_length_predictor = len(predictor_full_ids)
            full_length_dataset = len(dataset_full_ids)
            
            # 找出output部分的token IDs
            predictor_output_ids = predictor_full_ids[input_length_predictor:]
            dataset_output_ids = dataset_full_ids[input_length_dataset:]
            
            predictor_output_tokens = [self.tokenizer.decode([tid]) for tid in predictor_output_ids]
            dataset_output_tokens = [dataset_tokenizer.decode([tid]) for tid in dataset_output_ids]
            
            output_ids_match = (predictor_output_ids == dataset_output_ids)
            
            result = {
                'sample_index': idx,
                'input_text_preview': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                'output_text': output_text,
                'predictor': {
                    'input_length': input_length_predictor,
                    'full_length': full_length_predictor,
                    'output_length': len(predictor_output_ids),
                    'input_ids': predictor_input_ids,
                    'output_ids': predictor_output_ids,
                    'input_tokens': predictor_input_tokens,
                    'output_tokens': predictor_output_tokens
                },
                'dataset': {
                    'input_length': input_length_dataset,
                    'full_length': full_length_dataset,
                    'output_length': len(dataset_output_ids),
                    'input_ids': dataset_input_ids,
                    'output_ids': dataset_output_ids,
                    'input_tokens': dataset_input_tokens,
                    'output_tokens': dataset_output_tokens
                },
                'match': {
                    'input_ids_match': input_ids_match,
                    'output_ids_match': output_ids_match,
                    'full_ids_match': full_ids_match
                }
            }
            
            comparison_results.append(result)
            
            # 打印差异（如果存在）
            if not full_ids_match:
                print(f"\n样本 {idx} - 发现差异:")
                print(f"  输入: {input_text[:80]}...")
                print(f"  输出: {output_text}")
                
                if not input_ids_match:
                    print(f"  ❌ Input部分Token IDs不匹配")
                    print(f"     Predictor长度: {input_length_predictor}")
                    print(f"     Dataset长度: {input_length_dataset}")
                else:
                    print(f"  ✓ Input部分Token IDs匹配")
                
                if not output_ids_match:
                    print(f"  ❌ Output部分Token IDs不匹配")
                    print(f"     Predictor: {predictor_output_ids} -> {predictor_output_tokens}")
                    print(f"     Dataset:   {dataset_output_ids} -> {dataset_output_tokens}")
                else:
                    print(f"  ✓ Output部分Token IDs匹配")
        
        # 生成摘要
        summary = {
            'total_samples': len(samples),
            'samples_with_differences': total_differences,
            'match_rate': (len(samples) - total_differences) / len(samples) if samples else 0,
            'predictor_vocab_size': len(self.tokenizer),
            'dataset_vocab_size': len(dataset_tokenizer),
            'vocab_size_match': (len(self.tokenizer) == len(dataset_tokenizer))
        }
        
        print(f"\n=== 比较摘要 ===")
        print(f"总样本数: {summary['total_samples']}")
        print(f"存在差异的样本数: {summary['samples_with_differences']}")
        print(f"匹配率: {summary['match_rate']:.2%}")
        print(f"Predictor词汇表大小: {summary['predictor_vocab_size']}")
        print(f"Dataset词汇表大小: {summary['dataset_vocab_size']}")
        print(f"词汇表大小匹配: {'✓' if summary['vocab_size_match'] else '❌'}")
        
        # 构造输出
        output_data = {
            'summary': summary,
            'comparisons': comparison_results
        }
        
        # 保存到文件（如果指定）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n详细比较结果已保存到: {output_path}")
        
        return output_data


def main():
    """
    使用示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='工具预测推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'evaluate', 'compare'], 
                       help='运行模式: single=单条预测, evaluate=批量评估, compare=比较tokenization')
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
    
    # 比较tokenization模式参数
    parser.add_argument('--dataset_tokenizer', type=str, help='Dataset使用的tokenizer路径（比较模式，可选）')
    parser.add_argument('--compare_samples', type=int, default=10, help='比较的样本数量（比较模式）')
    
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
    
    elif args.mode == 'compare':
        # 比较tokenization模式
        if not args.data:
            parser.error("比较模式需要 --data 参数")
        
        predictor.compare_tokenization(
            data_path=args.data,
            dataset_tokenizer_path=args.dataset_tokenizer,
            num_samples=args.compare_samples,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
