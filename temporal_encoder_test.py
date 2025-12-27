"""
Temporal Encoder Testing Script for TOMAS-LLM.
Loads a trained temporal encoder and generates predictions to compare with ground truth.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import yaml

from src.context.temporal_encoder import TemporalEncoder
from src.context.temporal_llm_wrapper import TemporalLLMWrapper
from src.data.temporal_pretrain_dataset import TemporalPretrainDataset, collate_fn


class TemporalEncoderTester:
    """Test trained temporal encoder with text comparison."""
    
    def __init__(
        self,
        model: TemporalLLMWrapper,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Set to evaluation mode
        self.model.eval()
        
    @torch.no_grad()
    def generate_from_curve(
        self, 
        curve: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.1,  # Lower temperature for more deterministic output
        top_p: float = 0.9
    ) -> List[str]:
        """Generate text from resource curve using temporal encoder + LLM."""
        # Generate using the model's generate method
        generated_ids = self.model.generate(
            curve=curve,
            prompt_ids=prompt_ids,
            max_length=prompt_ids.size(1) + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_beams=1
        )
        
        # Decode generated tokens (remove prompt)
        generated_texts = []
        for i, gen_ids in enumerate(generated_ids):
            prompt_len = prompt_ids[i].size(0)
            gen_tokens = gen_ids[prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def test_samples(
        self, 
        dataloader: DataLoader,
        num_samples: int = None,
        max_new_tokens: int = 256,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Test temporal encoder on samples and collect results."""
        results = []
        
        pbar = tqdm(dataloader, desc="Testing temporal encoder", total=num_samples, disable=not verbose)
        sample_count = 0
        
        for batch in pbar:
            if num_samples and sample_count >= num_samples:
                break
                
            # Move batch to device
            curve = batch["curve"].to(self.device)
            prompt_ids = batch["prompt_ids"].to(self.device)
            target_ids = batch["target_ids"]
            task_types = batch["task_type"]
            
            batch_size = curve.size(0)
            
            # Decode ground truth
            ground_truth_texts = []
            for ids in target_ids:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                ground_truth_texts.append(text)
            
            # Decode prompts
            prompt_texts = []
            for ids in prompt_ids:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                prompt_texts.append(text)
            
            # Generate predictions
            predicted_texts = self.generate_from_curve(
                curve, prompt_ids, max_new_tokens=max_new_tokens
            )
            
            # Collect results
            for i in range(batch_size):
                if num_samples and sample_count >= num_samples:
                    break
                    
                result = {
                    "sample_id": sample_count,
                    "task_type": task_types[i],
                    "curve_shape": list(curve[i].shape),
                    "prompt": prompt_texts[i],
                    "ground_truth": ground_truth_texts[i],
                    "prediction": predicted_texts[i],
                    "curve_data": {
                        "cpu": curve[i, 0].cpu().tolist()[:20],  # First 20 timesteps
                        "memory": curve[i, 1].cpu().tolist()[:20],
                        "disk_io": curve[i, 2].cpu().tolist()[:20],
                        "network": curve[i, 3].cpu().tolist()[:20],
                    }
                }
                results.append(result)
                sample_count += 1
            
            if verbose:
                pbar.set_postfix({"samples": sample_count})
        
        return results


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda"
):
    """Load temporal encoder model from checkpoint."""
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Load LLM and tokenizer
    print(f"Loading LLM: {model_config['llm_name']}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create TemporalEncoder
    print("Creating TemporalEncoder...")
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
    
    # Load weights
    print("Loading checkpoint weights...")
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Global step: {checkpoint.get('global_step', 'N/A')}")
    print(f"  - Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"  - Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model, tokenizer


def print_text_comparison(results: List[Dict[str, Any]], max_display: int = 5):
    """Print text comparison in a readable format."""
    print("\n" + "="*100)
    print(f"TEXT COMPARISON RESULTS (Showing {min(max_display, len(results))} of {len(results)} samples)")
    print("="*100)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n{'─'*100}")
        print(f"SAMPLE #{result['sample_id']} | Task Type: {result['task_type']}")
        print(f"{'─'*100}")
        
        # Print curve preview
        print(f"\n📊 Resource Curve Preview (first 20 timesteps):")
        print(f"   CPU:     {', '.join([f'{v:.2f}' for v in result['curve_data']['cpu'][:10]])}...")
        print(f"   Memory:  {', '.join([f'{v:.2f}' for v in result['curve_data']['memory'][:10]])}...")
        print(f"   DiskIO:  {', '.join([f'{v:.2f}' for v in result['curve_data']['disk_io'][:10]])}...")
        print(f"   Network: {', '.join([f'{v:.2f}' for v in result['curve_data']['network'][:10]])}...")
        
        # Print prompt
        print(f"\n💬 Prompt:")
        print(f"   {result['prompt']}")
        
        # Print ground truth
        print(f"\n✓ Ground Truth:")
        print(f"   {result['ground_truth']}")
        
        # Print prediction
        print(f"\n🤖 Model Prediction:")
        print(f"   {result['prediction']}")
        
        # Simple match indicator
        gt = result['ground_truth'].strip().lower()
        pred = result['prediction'].strip().lower()
        
        if gt == pred:
            print(f"\n   ✅ EXACT MATCH!")
        elif pred.startswith(gt[:20]) or gt.startswith(pred[:20]):
            print(f"\n   ⚠️  PARTIAL MATCH (similar beginning)")
        else:
            print(f"\n   ❌ MISMATCH")
    
    print(f"\n{'='*100}\n")


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate basic statistics from results."""
    stats = {
        'total_samples': len(results),
        'by_task_type': {},
        'exact_matches': 0,
        'partial_matches': 0,
    }
    
    for result in results:
        task_type = result['task_type']
        if task_type not in stats['by_task_type']:
            stats['by_task_type'][task_type] = 0
        stats['by_task_type'][task_type] += 1
        
        # Check match
        gt = result['ground_truth'].strip().lower()
        pred = result['prediction'].strip().lower()
        
        if gt == pred:
            stats['exact_matches'] += 1
        elif pred.startswith(gt[:20]) or gt.startswith(pred[:20]):
            stats['partial_matches'] += 1
    
    stats['exact_match_rate'] = stats['exact_matches'] / stats['total_samples'] * 100
    stats['partial_match_rate'] = stats['partial_matches'] / stats['total_samples'] * 100
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Test trained temporal encoder")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to temporal encoder checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_temporal.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=20,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--max_display",
        type=int,
        default=5,
        help="Maximum number of samples to display in terminal"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/generated",
        help="Directory to save output JSON"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Random seed for test data generation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*100)
    print("TOMAS-LLM Temporal Encoder Testing")
    print("="*100)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load dataset
    print("\n[1/4] Creating test dataset...")
    tokenizer_temp = AutoTokenizer.from_pretrained(
        config['model']['llm_name'],
        trust_remote_code=True
    )
    if tokenizer_temp.pad_token is None:
        tokenizer_temp.pad_token = tokenizer_temp.eos_token
    
    dataset = TemporalPretrainDataset(
        num_samples=args.num_samples,
        tokenizer=tokenizer_temp,
        type_distribution=config['data']['type_distribution'],
        max_length=config['data']['max_length'],
        noise_level=config['data'].get('noise_level', 0.05),
        spike_probability=config['data'].get('spike_probability', 0.3),
        seed=args.seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Batch size: {args.batch_size}")
    
    # 2. Load model
    print("\n[2/4] Loading temporal encoder from checkpoint...")
    model, tokenizer = load_model_from_checkpoint(
        args.checkpoint,
        args.config,
        args.device
    )
    
    # 3. Run tests
    print("\n[3/4] Running tests...")
    tester = TemporalEncoderTester(model, tokenizer, args.device)
    results = tester.test_samples(
        dataloader, 
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    # 4. Display results
    print("\n[4/4] Displaying results...")
    print_text_comparison(results, max_display=args.max_display)
    
    # Calculate statistics
    stats = calculate_statistics(results)
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    print(f"Total samples tested: {stats['total_samples']}")
    print(f"Exact matches: {stats['exact_matches']} ({stats['exact_match_rate']:.2f}%)")
    print(f"Partial matches: {stats['partial_matches']} ({stats['partial_match_rate']:.2f}%)")
    print(f"\nBreakdown by task type:")
    for task_type, count in stats['by_task_type'].items():
        print(f"  - Type {task_type}: {count} samples")
    print("="*100 + "\n")
    
    # 5. Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, 
        f"temporal_encoder_test_{timestamp}.json"
    )
    
    output_data = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "num_samples": len(results),
            "timestamp": timestamp,
            "statistics": stats
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Full results saved to: {output_file}")
    print(f"  - You can review all {len(results)} samples in this file\n")


if __name__ == "__main__":
    main()
