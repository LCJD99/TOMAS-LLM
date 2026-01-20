"""
Inference predictor V3: For Virtual Token Architecture
Compatible with ToolPlannerModel (no embedding expansion).
"""

import json
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from src.models.tool_planner import ToolPlannerModel
from src.models.profile_encoder import create_profile_encoder, load_system_config
from src.data.token_schema import TOOL_ABBREV


def parse_virtual_token(token: str) -> Dict[str, str]:
    """
    Parse fields from virtual token.
    Format: <TOOL_SIZE_CPU_CPUMEM_GPU_GPUMEM>
    Example: <IMG_CLS_SMALL_LOW_MED_HIGH_LOW>
    
    Args:
        token: Virtual token string
    
    Returns:
        Field dict containing tool, size, cpu_core, cpu_mem, gpu_sm, gpu_mem
        Returns empty dict if parsing fails
    """
    if not token.startswith('<') or not token.endswith('>'):
        return {}
    
    # Remove brackets
    token_content = token[1:-1]
    
    # Split by underscore
    parts = token_content.split('_')
    
    if len(parts) < 6:
        return {}
    
    tool_name = parts[0:-5]
    
    return {
        'tool': '_'.join(tool_name),
        'size': parts[-5],
        'cpu_core': parts[-4],
        'cpu_mem': parts[-3],
        'gpu_sm': parts[-2],
        'gpu_mem': parts[-1]
    }


def compare_token_fields(predicted: str, ground_truth: str) -> Dict[str, Any]:
    """
    Compare predicted token and ground truth token fields.
    
    Args:
        predicted: Predicted token
        ground_truth: Ground truth token
    
    Returns:
        Comparison result dict with field matches and overall match rate
    """
    pred_fields = parse_virtual_token(predicted)
    gt_fields = parse_virtual_token(ground_truth)
    
    # If either parse fails, return error info
    if not pred_fields or not gt_fields:
        return {
            'exact_match': predicted == ground_truth,
            'field_matches': {},
            'match_rate': 1.0 if predicted == ground_truth else 0.0,
            'matched_fields': 0,
            'total_fields': 0,
            'parse_error': True
        }
    
    # Compare each field
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
    Logits processor that restricts sampling to virtual tokens only.
    """
    
    def __init__(self, virtual_token_ids: List[int]):
        """
        Args:
            virtual_token_ids: List of allowed virtual token IDs
        """
        self.virtual_token_ids = set(virtual_token_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Set non-virtual token logits to -inf, forcing sampling from virtual tokens only.
        
        Args:
            input_ids: Input token IDs
            scores: Model output logits [batch_size, vocab_size]
        
        Returns:
            Processed logits with only virtual token scores preserved
        """
        # Create mask to preserve only virtual token logits
        mask = torch.ones_like(scores, dtype=torch.bool)
        for token_id in self.virtual_token_ids:
            mask[:, token_id] = False
        
        # Set non-virtual token logits to -inf
        scores[mask] = -float('inf')
        
        return scores


def get_virtual_token_ids(tokenizer) -> List[int]:
    """
    Get all virtual token IDs.
    Assumes virtual tokens have format <TOOL_...>, e.g., <IMG_CLS_SMALL_LOW_MED_HIGH_LOW>
    
    Args:
        tokenizer: Tokenizer instance
    
    Returns:
        List of virtual token IDs
    """
    virtual_token_ids = []
    
    for token, token_id in tokenizer.get_vocab().items():
        # Check if it's a virtual token (starts with <, ends with >, contains _)
        if token.startswith('<') and token.endswith('>') and '_' in token:
            # Further check if contains tool name keyword
            tool_keywords = TOOL_ABBREV.values()
            if any(keyword in token for keyword in tool_keywords):
                virtual_token_ids.append(token_id)
    
    print(f"Found {len(virtual_token_ids)} virtual tokens")
    return virtual_token_ids


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load JSONL file.
    
    Args:
        file_path: JSONL file path
    
    Returns:
        List of data samples
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


class ToolPredictorV3:
    """
    Tool predictor V3: For Virtual Token Architecture.
    Accepts loaded ToolPlannerModel and tokenizer.
    """
    
    def __init__(
        self,
        model: ToolPlannerModel,
        tokenizer,
        device: Optional[str] = None,
        constrained: bool = True
    ):
        """
        Initialize predictor.
        
        Args:
            model: Loaded ToolPlannerModel
            tokenizer: Loaded tokenizer
            device: Computation device (None for auto-detection)
            constrained: Whether to use constrained generation (sample from virtual tokens only)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Determine device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        
        print(f"Predictor device: {self.device}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Save constraint config
        self.constrained = constrained
        
        # Get virtual token ID list
        if constrained:
            print("Identifying virtual tokens...")
            self.virtual_token_ids = get_virtual_token_ids(self.tokenizer)
            
            # Create logits processor
            self.logits_processor = LogitsProcessorList([
                VirtualTokenOnlyLogitsProcessor(self.virtual_token_ids)
            ])
            print(f"Predictor initialized (constrained generation mode)")
        else:
            self.virtual_token_ids = []
            self.logits_processor = None
            print(f"Predictor initialized (unconstrained generation mode)")
    
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
        Predict tool token.
        
        Args:
            input_text: Input text (task description)
            max_new_tokens: Maximum number of new tokens to generate (default 1)
            temperature: Sampling temperature
            do_sample: Whether to use sampling (False for greedy decoding)
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            constrained: Whether to use constrained generation (None uses initialization setting)
        
        Returns:
            Predicted token string
        """
        # Construct prompt
        prompt = f"Input: {input_text}\nOutput: "
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Decide whether to use constraint
        use_constrained = constrained if constrained is not None else self.constrained
        
        # Generate
        with torch.no_grad():
            if use_constrained and self.logits_processor is not None:
                # Use custom generation with logits processor
                outputs = self._generate_with_processor(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample
                )
            else:
                # Use standard generation
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens
                )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract output part (remove prompt)
        output_text = generated_text[len(prompt):].strip()
        
        # Extract first token (usually the virtual token)
        output_tokens = output_text.split()
        if output_tokens:
            predicted_token = output_tokens[0]
        else:
            predicted_token = output_text
        
        return predicted_token
    
    def _generate_with_processor(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        """
        Custom generation with logits processor (for constrained generation).
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Generated token IDs
        """
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get embeddings
            inputs_embeds = self.model.embed_tokens(generated)
            
            # Forward through LLM
            outputs = self.model.llm(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            logits = self.model.compute_logits(hidden_states)
            
            # Get last position logits
            next_token_logits = logits[:, -1, :]
            
            # Apply logits processor
            next_token_logits = self.logits_processor[0](generated, next_token_logits)
            
            # Sample or select greedily
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def predict_batch(
        self,
        input_texts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Batch prediction.
        
        Args:
            input_texts: List of input texts
            **kwargs: Other arguments for predict method
        
        Returns:
            List of predicted virtual tokens
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
        Predict and return top-k candidate tokens with scores.
        
        Args:
            input_text: Input text
            top_k: Return top k candidates
            constrained: Whether to use constrained generation (None uses initialization setting)
        
        Returns:
            List of [(token, score), ...]
        """
        # Construct prompt
        prompt = f"Input: {input_text}\nOutput: "
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Forward pass
        with torch.no_grad():
            inputs_embeds = self.model.embed_tokens(inputs.input_ids)
            outputs = self.model.llm(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            logits = self.model.compute_logits(hidden_states)
            logits = logits[:, -1, :]  # Get last position logits
        
        # Decide whether to use constraint
        use_constrained = constrained if constrained is not None else self.constrained
        
        # Apply logits processor (if using constraint)
        if use_constrained and self.logits_processor is not None:
            processed_logits = self.logits_processor[0](inputs.input_ids, logits)
            max_k = min(top_k, len(self.virtual_token_ids))
        else:
            processed_logits = logits
            max_k = top_k
        
        # Compute probabilities
        probs = torch.softmax(processed_logits, dim=-1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs[0], k=max_k)
        
        # Decode tokens
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
        Evaluate model on training data and output results.
        
        Args:
            data_path: Training data path (JSONL format)
            output_path: Output JSON file path
            num_samples: Number of evaluation samples (None for all)
            **predict_kwargs: Other arguments for predict method
        
        Returns:
            Evaluation result dict
        """
        print(f"\nLoading data: {data_path}")
        data = load_jsonl(data_path)
        
        if num_samples is not None:
            import random
            if num_samples < len(data):
                data = random.sample(data, num_samples)
        
        print(f"Evaluation samples: {len(data)}")
        
        # Evaluate
        results = []
        exact_match_count = 0
        total = len(data)
        
        # Field match statistics
        field_names = ['tool', 'size', 'cpu_core', 'cpu_mem', 'gpu_sm', 'gpu_mem']
        overall_field_matches = {field: 0 for field in field_names}
        overall_match_rate_sum = 0.0
        
        # Statistics by tool and augmentation type
        tool_stats = defaultdict(lambda: {'exact_match': 0, 'total': 0, 'match_rate_sum': 0.0})
        augmentation_stats = defaultdict(lambda: {'exact_match': 0, 'total': 0, 'match_rate_sum': 0.0})
        
        print("\nStarting prediction...")
        for item in tqdm(data, desc="Evaluation progress"):
            input_text = item['input']
            ground_truth = item['output']
            
            # Predict
            try:
                prediction = self.predict(input_text, **predict_kwargs)
            except Exception as e:
                print(f"\nPrediction failed: {e}")
                prediction = "[ERROR]"
            
            # Compare field matches
            comparison = compare_token_fields(prediction, ground_truth)
            
            # Count exact match
            if comparison['exact_match']:
                exact_match_count += 1
            
            # Accumulate field matches
            if not comparison['parse_error']:
                for field in field_names:
                    if comparison['field_matches'].get(field, False):
                        overall_field_matches[field] += 1
                overall_match_rate_sum += comparison['match_rate']
            
            # Extract tool name and augmentation type
            tool_name = item.get('tool', 'unknown')
            augmentation_type = item.get('augmentation_type', 'unknown')
            
            # Update statistics
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
            
            # Record result
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
        
        # Compute overall metrics
        exact_match_rate = exact_match_count / total if total > 0 else 0.0
        overall_avg_match_rate = overall_match_rate_sum / total if total > 0 else 0.0
        
        # Compute per-field match rates
        field_match_rates = {}
        for field in field_names:
            field_match_rates[field] = overall_field_matches[field] / total if total > 0 else 0.0
        
        # Compute per-tool accuracy
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
        
        # Compute per-augmentation-type accuracy
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
        
        # Construct output
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
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nEvaluation complete!")
        print(f"Total samples: {total}")
        print(f"Exact matches: {exact_match_count}")
        print(f"Exact match rate: {exact_match_rate:.2%}")
        print(f"Average field match rate: {overall_avg_match_rate:.2%}")
        print(f"\nPer-field match rates:")
        for field, rate in field_match_rates.items():
            print(f"  {field:12s}: {rate:.2%}")
        print(f"\nResults saved to: {output_path}")
        
        # Show error cases (first 5)
        error_cases = [r for r in results if not r['exact_match']]
        if error_cases:
            print(f"\nNon-exact-match cases: {len(error_cases)}")
            print("\nFirst 5 non-exact-match cases:")
            for i, case in enumerate(error_cases[:5], 1):
                print(f"\n{i}. Input: {case['input'][:100]}...")
                print(f"   Ground Truth: {case['ground_truth']}")
                print(f"   Prediction:   {case['prediction']}")
                print(f"   Field match rate: {case['match_rate']:.1%} ({case['matched_fields']}/{case['total_fields']})")
                if case.get('field_matches'):
                    mismatched = [f for f, matched in case['field_matches'].items() if not matched]
                    if mismatched:
                        print(f"   Mismatched fields: {', '.join(mismatched)}")
        
        return output_data


def load_model_for_inference(
    model_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    torch_dtype=torch.bfloat16
):
    """
    Load ToolPlannerModel for inference.
    
    Args:
        model_path: Model path
        device: Computation device
        torch_dtype: Data type
    
    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load metadata
    import os
    metadata = torch.load(os.path.join(model_path, 'metadata.bin'), map_location=device)
    
    # Load base LLM
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map='auto' if device == 'cuda' else None
    )
    
    if device == 'cpu':
        base_llm = base_llm.to(device)
    
    # Create profile encoder if needed
    profile_encoder = None
    if metadata['use_profile_encoding']:
        profile_encoder = create_profile_encoder(
            encoder_type='simple',
            input_dim=5,
            output_dim=metadata['hidden_size'],
            hidden_dims=[128, 512],
            normalize=True
        ).to(device=device, dtype=torch_dtype)
    
    # Load ToolPlannerModel
    model = ToolPlannerModel.load_pretrained(
        load_dir=model_path,
        llm=base_llm,
        tokenizer=tokenizer,
        profile_encoder=profile_encoder,
        device=device
    )
    
    model.eval()
    
    print(f"✓ Model loaded, device: {next(model.parameters()).device}")
    print(f"✓ Virtual Token Architecture (V3)")
    print(f"  - Num virtual tokens: {model.num_virtual_tokens}")
    print(f"  - Virtual token start idx: {model.virtual_token_start_idx}")
    
    return model, tokenizer


def main():
    """
    Usage example
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Tool Prediction Inference V3 (Virtual Tokens)')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'evaluate'], 
                       help='Run mode: single=single prediction, evaluate=batch evaluation')
    parser.add_argument('--constrained', action='store_true', default=True, 
                       help='Use constrained generation (sample from virtual tokens only)')
    parser.add_argument('--no-constrained', dest='constrained', action='store_false',
                       help='Use unconstrained generation')
    parser.add_argument('--device', type=str, default=None, 
                       help='Computation device (cuda/cpu, None for auto-detection)')
    
    # Single prediction mode parameters
    parser.add_argument('--input', type=str, help='Input task description (single prediction mode)')
    parser.add_argument('--top_k', type=int, default=5, help='Show top-k candidates')
    parser.add_argument('--sample', action='store_true', help='Use sampling instead of greedy decoding')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=1, help='Maximum new tokens to generate')
    
    # Batch evaluation mode parameters
    parser.add_argument('--data', type=str, help='Training data path (evaluation mode)')
    parser.add_argument('--output', type=str, help='Output JSON file path (evaluation mode)')
    parser.add_argument('--num_samples', type=int, help='Evaluation sample count (None=all)')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(
        args.model,
        device=args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Initialize predictor
    predictor = ToolPredictorV3(
        model=model,
        tokenizer=tokenizer,
        constrained=args.constrained
    )
    
    if args.mode == 'single':
        # Single prediction mode
        if not args.input:
            parser.error("Single prediction mode requires --input parameter")
        
        print(f"\nInput: {args.input}")
        print("-" * 60)
        
        # Predict
        prediction = predictor.predict(
            args.input,
            max_new_tokens=args.max_tokens,
            do_sample=args.sample,
            temperature=args.temperature
        )
        print(f"Prediction result: {prediction}")
        
        # Show top-k candidates
        print(f"\nTop-{args.top_k} candidates:")
        candidates = predictor.predict_with_scores(args.input, top_k=args.top_k)
        for i, (token, score) in enumerate(candidates, 1):
            print(f"{i}. {token:30s} (score: {score:.4f})")
    
    elif args.mode == 'evaluate':
        # Batch evaluation mode
        if not args.data or not args.output:
            parser.error("Evaluation mode requires --data and --output parameters")
        
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
