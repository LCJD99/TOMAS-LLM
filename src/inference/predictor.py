"""
推理工具：使用训练好的模型进行预测
限制只从扩充的虚拟token中采样
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from typing import List, Optional
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


class ToolPredictor:
    """
    工具预测器：加载模型并执行推理
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        torch_dtype=torch.bfloat16
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            device: 计算设备
            torch_dtype: 数据类型
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
        
        # 获取虚拟token ID列表
        print("识别虚拟token...")
        self.virtual_token_ids = get_virtual_token_ids(self.tokenizer)
        
        # 创建logits processor
        self.logits_processor = LogitsProcessorList([
            VirtualTokenOnlyLogitsProcessor(self.virtual_token_ids)
        ])
        
        print(f"模型加载完成，使用设备: {device}")
    
    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> str:
        """
        预测工具token（仅从虚拟token中采样）
        
        Args:
            input_text: 输入文本（任务描述）
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否使用采样（False为贪婪解码）
            top_p: Nucleus采样参数
            top_k: Top-K采样参数
        
        Returns:
            预测的虚拟token字符串
        """
        # 构造prompt
        prompt = f"Input: {input_text}\nOutput: "
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate（限制只从虚拟token采样）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 50,
                logits_processor=self.logits_processor,  # 限制采样范围
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
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
        top_k: int = 5
    ) -> List[tuple]:
        """
        预测并返回top-k个候选token及其分数
        
        Args:
            input_text: 输入文本
            top_k: 返回前k个候选
        
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
        
        # 应用logits processor（限制虚拟token）
        processed_logits = self.logits_processor[0](inputs.input_ids, logits)
        
        # 计算概率
        probs = torch.softmax(processed_logits, dim=-1)
        
        # 获取top-k
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(self.virtual_token_ids)))
        
        # 解码token
        results = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()])
            results.append((token, prob.item()))
        
        return results


def main():
    """
    使用示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='工具预测推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--input', type=str, required=True, help='输入任务描述')
    parser.add_argument('--top_k', type=int, default=5, help='显示top-k候选')
    parser.add_argument('--sample', action='store_true', help='使用采样而非贪婪解码')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = ToolPredictor(args.model)
    
    # 预测
    print(f"\n输入: {args.input}")
    print("-" * 60)
    
    # 贪婪解码预测
    prediction = predictor.predict(
        args.input,
        do_sample=args.sample,
        temperature=args.temperature
    )
    print(f"预测结果: {prediction}")
    
    # 显示top-k候选
    print(f"\nTop-{args.top_k} 候选:")
    candidates = predictor.predict_with_scores(args.input, top_k=args.top_k)
    for i, (token, score) in enumerate(candidates, 1):
        print(f"{i}. {token:30s} (score: {score:.4f})")


if __name__ == '__main__':
    main()
