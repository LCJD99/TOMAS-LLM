"""
Temporal Encoder Evaluation Script.

Evaluate trained Temporal Encoder on various metrics:
1. Perplexity (overall and per task type)
2. Token Accuracy
3. Numerical Accuracy

Usage:
    python script/evaluate_temporal_encoder.py \
        --checkpoint checkpoints/temporal_pretrain/best_model.pt \
        --num_samples 5000 \
        --output_dir logs/temporal_pretrain/evaluation
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.context.temporal_encoder import TemporalEncoder
from src.context.temporal_llm_wrapper import TemporalLLMWrapper
from src.data.temporal_pretrain_dataset import TemporalPretrainDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_memory_usage(device: str, prefix: str = ""):
    """Log current GPU memory usage."""
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        tokenizer: Tokenizer
        checkpoint: Checkpoint dict (for metadata)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    # Load checkpoint to CPU first to avoid OOM
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    log_memory_usage(device, "Before loading model: ")
    
    # Infer config from checkpoint (or load from default)
    # For now, use default config
    config_path = "configs/pretrain_temporal.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Load LLM and tokenizer
    logger.info(f"Loading LLM: {model_config['llm_name']}")
    # Force fp16 for evaluation to save memory
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Always use fp16 for evaluation
        device_map=device  # Load directly to device
    )
    log_memory_usage(device, "After loading LLM: ")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create TemporalEncoder
    temporal_config = model_config['temporal_encoder']
    cnn_config = {
        'in_channels': 4,
        'hidden_channels': temporal_config['hidden_channels'],
        'output_dim': temporal_config['output_dim'],
        'num_layers': temporal_config['num_layers'],
        'pooling': temporal_config['pooling']
    }
    
    temporal_encoder = TemporalEncoder(
        timeline=None,
        cnn_config=cnn_config,
        min_timesteps=temporal_config['min_timesteps'],
        max_timesteps=temporal_config['max_timesteps'],
        time_granularity_ms=temporal_config['time_granularity_ms'],
        llm_embedding_dim=model_config['llm_embedding_dim']
    )
    
    # Create wrapper
    model = TemporalLLMWrapper(
        temporal_encoder=temporal_encoder,
        llm_model=llm_model,
        llm_embedding_dim=model_config['llm_embedding_dim'],
        freeze_llm=model_config['freeze_llm']
    )
    
    # Move temporal encoder to device and convert to fp16
    temporal_encoder = temporal_encoder.to(device).half()
    
    # Create wrapper (LLM already on device from device_map)
    model = TemporalLLMWrapper(
        temporal_encoder=temporal_encoder,
        llm_model=llm_model,
        llm_embedding_dim=model_config['llm_embedding_dim'],
        freeze_llm=model_config['freeze_llm']
    )
    
    # Load weights from checkpoint
    logger.info("Loading checkpoint weights...")
    state_dict = checkpoint['model_state_dict']
    # Convert state dict to fp16 if needed
    state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    log_memory_usage(device, "After loading full model: ")
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        log_memory_usage(device, "After clearing cache: ")
    
    logger.info(f"Model loaded from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    logger.info(f"Checkpoint train loss: {checkpoint['train_loss']:.4f}")
    logger.info(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    
    return model, tokenizer, checkpoint


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
    
    Returns:
        List of extracted numbers
    """
    # Pattern for numbers (including decimals)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def calculate_numerical_accuracy(
    generated_texts: List[str],
    ground_truth_texts: List[str],
    tolerance: float = 0.05
) -> Tuple[float, int, int]:
    """
    Calculate numerical accuracy between generated and ground truth texts.
    
    Args:
        generated_texts: List of generated texts
        ground_truth_texts: List of ground truth texts
        tolerance: Relative error tolerance (default 5%)
    
    Returns:
        accuracy: Numerical accuracy
        num_correct: Number of correct numbers
        num_total: Total number of numbers
    """
    num_correct = 0
    num_total = 0
    
    for gen_text, gt_text in zip(generated_texts, ground_truth_texts):
        gen_numbers = extract_numbers(gen_text)
        gt_numbers = extract_numbers(gt_text)
        
        # Match numbers by position
        for i in range(min(len(gen_numbers), len(gt_numbers))):
            num_total += 1
            
            gt_num = gt_numbers[i]
            gen_num = gen_numbers[i]
            
            # Check if within tolerance
            if gt_num == 0:
                # Absolute error for zero
                if abs(gen_num - gt_num) < 0.01:
                    num_correct += 1
            else:
                # Relative error
                relative_error = abs(gen_num - gt_num) / abs(gt_num)
                if relative_error <= tolerance:
                    num_correct += 1
    
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return accuracy, num_correct, num_total


def evaluate_perplexity(
    model,
    dataloader,
    device,
    task_type: str = None
) -> Tuple[float, float]:
    """
    Calculate perplexity on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
        task_type: Filter by task type (A, B, C, or None for all)
    
    Returns:
        avg_loss: Average loss
        perplexity: Perplexity
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Perplexity (Task {task_type or 'All'})"):
            # Filter by task type if specified
            if task_type is not None:
                mask = torch.tensor([t == task_type for t in batch['task_type']])
                if not mask.any():
                    continue
                
                curve = batch['curve'][mask].to(device)
                prompt_ids = batch['prompt_ids'][mask].to(device)
                target_ids = batch['target_ids'][mask].to(device)
                attention_mask = batch['attention_mask'][mask].to(device)
            else:
                curve = batch['curve'].to(device)
                prompt_ids = batch['prompt_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            
            if curve.size(0) == 0:
                continue
            
            loss = model(curve, prompt_ids, target_ids, attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def evaluate_token_accuracy(
    model,
    dataloader,
    device,
    tokenizer,
    top_k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate token accuracy metrics.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
        tokenizer: Tokenizer
        top_k_list: List of K values for top-K accuracy
    
    Returns:
        Dictionary of accuracy metrics
    """
    model.eval()
    
    total_tokens = 0
    correct_tokens = 0
    top_k_correct = {k: 0 for k in top_k_list}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Token Accuracy"):
            batch_size = len(batch['curve'])
            
            # Process samples one by one to save memory
            for i in range(batch_size):
                curve = batch['curve'][i:i+1].to(device)
                prompt_ids = batch['prompt_ids'][i:i+1].to(device)
                target_ids = batch['target_ids'][i:i+1]
                
                # Generate for single sample
                generated_ids = model.generate(
                    curve=curve,
                    prompt_ids=prompt_ids,
                    max_length=target_ids[i].size(0) + prompt_ids[i].size(0),
                    num_beams=1,
                    do_sample=False
                )
                
                # Remove prompt tokens
                prompt_len = prompt_ids[0].size(0)
                generated_tokens = generated_ids[0, prompt_len:]
                target_tokens = target_ids[0]
                
                # Calculate accuracy
                min_len = min(len(generated_tokens), len(target_tokens))
                if min_len > 0:
                    total_tokens += min_len
                    matches = (generated_tokens[:min_len] == target_tokens[:min_len])
                    correct_tokens += matches.sum().item()
                
                # Clear cache periodically
                if device == 'cuda' and i % 10 == 0:
                    torch.cuda.empty_cache()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'token_accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }


def evaluate_generation_quality(
    model,
    dataloader,
    device,
    tokenizer,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate generation quality (numerical accuracy).
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
        tokenizer: Tokenizer
        num_samples: Number of samples to evaluate
    
    Returns:
        Dictionary of quality metrics
    """
    model.eval()
    
    generated_texts = []
    ground_truth_texts = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generation Quality"):
            if samples_collected >= num_samples:
                break
            
            batch_size = len(batch['curve'])
            
            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break
                
                # Load one sample at a time
                curve = batch['curve'][i:i+1].to(device)
                prompt_ids = batch['prompt_ids'][i:i+1].to(device)
                target_ids = batch['target_ids'][i:i+1]
                
                # Generate
                generated_ids = model.generate(
                    curve=curve,
                    prompt_ids=prompt_ids,
                    max_length=256,
                    num_beams=1,
                    do_sample=False
                )
                
                # Decode
                prompt_len = prompt_ids[0].size(0)
                generated_text = tokenizer.decode(
                    generated_ids[0, prompt_len:],
                    skip_special_tokens=True
                )
                ground_truth_text = tokenizer.decode(
                    target_ids[0],
                    skip_special_tokens=True
                )
                
                generated_texts.append(generated_text)
                ground_truth_texts.append(ground_truth_text)
                
                samples_collected += 1
                
                # Clear cache periodically
                if device == 'cuda' and samples_collected % 10 == 0:
                    torch.cuda.empty_cache()
    
    # Calculate numerical accuracy
    num_accuracy, num_correct, num_total = calculate_numerical_accuracy(
        generated_texts, ground_truth_texts
    )
    
    return {
        'numerical_accuracy': num_accuracy,
        'num_correct_numbers': num_correct,
        'num_total_numbers': num_total,
        'samples_evaluated': samples_collected
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal Encoder")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/pretrain_temporal.yaml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation (use smaller batch for large models)')
    parser.add_argument('--output_dir', type=str, default='logs/temporal_pretrain/evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    model, tokenizer, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluation dataset
    logger.info(f"Creating evaluation dataset with {args.num_samples} samples")
    eval_dataset = TemporalPretrainDataset(
        num_samples=args.num_samples,
        tokenizer=tokenizer,
        type_distribution=config['data']['type_distribution'],
        max_length=config['data']['max_length'],
        seed=config['training']['seed'] + 999  # Different seed for eval
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Evaluation results
    results = {
        'checkpoint_path': args.checkpoint,
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_step': checkpoint['global_step'],
        'checkpoint_train_loss': checkpoint['train_loss'],
        'checkpoint_val_loss': checkpoint['val_loss'],
        'num_samples': args.num_samples
    }
    
    # 1. Perplexity Evaluation
    logger.info("\n" + "="*80)
    logger.info("PERPLEXITY EVALUATION")
    logger.info("="*80)
    
    # Overall perplexity
    avg_loss, perplexity = evaluate_perplexity(model, eval_loader, device)
    logger.info(f"Overall Loss: {avg_loss:.4f}")
    logger.info(f"Overall Perplexity: {perplexity:.2f}")
    
    results['overall_loss'] = avg_loss
    results['overall_perplexity'] = perplexity
    
    # Per-task-type perplexity
    for task_type in ['A', 'B', 'C']:
        avg_loss, perplexity = evaluate_perplexity(model, eval_loader, device, task_type)
        logger.info(f"Task {task_type} Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        results[f'task_{task_type}_loss'] = avg_loss
        results[f'task_{task_type}_perplexity'] = perplexity
    
    # 2. Token Accuracy Evaluation
    logger.info("\n" + "="*80)
    logger.info("TOKEN ACCURACY EVALUATION")
    logger.info("="*80)
    
    token_metrics = evaluate_token_accuracy(
        model, eval_loader, device, tokenizer
    )
    
    logger.info(f"Token Accuracy: {token_metrics['token_accuracy']*100:.2f}%")
    logger.info(f"Correct Tokens: {token_metrics['correct_tokens']:,} / {token_metrics['total_tokens']:,}")
    
    results.update(token_metrics)
    
    # 3. Numerical Accuracy Evaluation
    logger.info("\n" + "="*80)
    logger.info("NUMERICAL ACCURACY EVALUATION")
    logger.info("="*80)
    
    quality_metrics = evaluate_generation_quality(
        model, eval_loader, device, tokenizer, num_samples=500
    )
    
    logger.info(f"Numerical Accuracy: {quality_metrics['numerical_accuracy']*100:.2f}%")
    logger.info(f"Correct Numbers: {quality_metrics['num_correct_numbers']:,} / {quality_metrics['num_total_numbers']:,}")
    logger.info(f"Samples Evaluated: {quality_metrics['samples_evaluated']}")
    
    results.update(quality_metrics)
    
    # 4. Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Overall Perplexity: {results['overall_perplexity']:.2f}")
    logger.info(f"  Task A Perplexity: {results['task_A_perplexity']:.2f}")
    logger.info(f"  Task B Perplexity: {results['task_B_perplexity']:.2f}")
    logger.info(f"  Task C Perplexity: {results['task_C_perplexity']:.2f}")
    logger.info(f"  Token Accuracy: {results['token_accuracy']*100:.2f}%")
    logger.info(f"  Numerical Accuracy: {results['numerical_accuracy']*100:.2f}%")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Quality assessment
    logger.info("\n" + "="*80)
    logger.info("QUALITY ASSESSMENT")
    logger.info("="*80)
    
    if results['overall_perplexity'] < 3.0:
        logger.info("✓ Perplexity: EXCELLENT (< 3.0)")
    elif results['overall_perplexity'] < 5.0:
        logger.info("✓ Perplexity: GOOD (< 5.0)")
    else:
        logger.info("✗ Perplexity: NEEDS IMPROVEMENT (>= 5.0)")
    
    if results['token_accuracy'] > 0.6:
        logger.info("✓ Token Accuracy: EXCELLENT (> 60%)")
    elif results['token_accuracy'] > 0.4:
        logger.info("✓ Token Accuracy: GOOD (> 40%)")
    else:
        logger.info("✗ Token Accuracy: NEEDS IMPROVEMENT (< 40%)")
    
    if results['numerical_accuracy'] > 0.7:
        logger.info("✓ Numerical Accuracy: EXCELLENT (> 70%)")
    elif results['numerical_accuracy'] > 0.5:
        logger.info("✓ Numerical Accuracy: GOOD (> 50%)")
    else:
        logger.info("✗ Numerical Accuracy: NEEDS IMPROVEMENT (< 50%)")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
