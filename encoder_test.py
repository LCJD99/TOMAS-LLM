"""
Encoder Testing Script for TOMAS-LLM.
Loads a trained encoder and generates predictions to compare with ground truth.
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

from src.data.pretrain_dataset import EncoderPretrainDataset
from src.offline.pretrain_encoder import ResourceEncoderForPretraining


class EncoderTester:
    def __init__(
        self,
        encoder: ResourceEncoderForPretraining,
        llm_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ):
        self.encoder = encoder.to(device)
        self.llm_model = llm_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Get LLM dtype for compatibility
        self.llm_dtype = next(self.llm_model.parameters()).dtype
        
        # Set to evaluation mode
        self.encoder.eval()
        self.llm_model.eval()
        
    def inject_prefix_embeddings(self, encoder_embeddings, max_new_tokens=50):
        """Inject encoder embeddings as prefix and prepare for generation."""
        batch_size = encoder_embeddings.size(0)
        prefix_embeddings = encoder_embeddings.unsqueeze(1)  # [B, 1, D]
        
        return prefix_embeddings
    
    @torch.no_grad()
    def generate_from_encoder_output(
        self, 
        tool_ids: torch.Tensor, 
        resource_vectors: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """Generate text from encoder output using LLM."""
        # Get encoder embeddings
        encoder_embeddings = self.encoder(tool_ids, resource_vectors)
        prefix_embeddings = self.inject_prefix_embeddings(encoder_embeddings)
        
        # Convert to LLM dtype for compatibility
        prefix_embeddings = prefix_embeddings.to(self.llm_dtype)
        
        # Generate directly from prefix embeddings
        # Note: Some tokenizers don't have bos_token_id, so we generate directly
        outputs = self.llm_model.generate(
            inputs_embeds=prefix_embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode generated tokens
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def test_samples(
        self, 
        dataloader: DataLoader,
        num_samples: int = None
    ) -> List[Dict[str, Any]]:
        """Test encoder on samples and collect results."""
        results = []
        
        pbar = tqdm(dataloader, desc="Testing encoder", total=num_samples)
        sample_count = 0
        
        for batch in pbar:
            if num_samples and sample_count >= num_samples:
                break
                
            # Move batch to device
            tool_ids = batch["tool_id"].to(self.device)
            resource_vectors = batch["resource_vector"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            
            # Decode ground truth
            ground_truth_texts = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                ground_truth_texts.append(text)
            
            # Generate predictions
            predicted_texts = self.generate_from_encoder_output(
                tool_ids, resource_vectors, max_new_tokens=100
            )
            
            # Collect results
            for i in range(len(tool_ids)):
                if num_samples and sample_count >= num_samples:
                    break
                    
                result = {
                    "sample_id": sample_count,
                    "tool_id": int(tool_ids[i].item()),
                    "resource_vector": resource_vectors[i].cpu().tolist(),
                    "ground_truth": ground_truth_texts[i],
                    "prediction": predicted_texts[i],
                }
                results.append(result)
                sample_count += 1
            
            pbar.set_postfix({"samples": sample_count})
        
        return results


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    llm_model_name: str,
    device: str = "cuda"
) -> ResourceEncoderForPretraining:
    """Load encoder from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get model configuration from checkpoint or use defaults
    # You may need to adjust these based on your actual model config
    encoder = ResourceEncoderForPretraining("Qwen2.5-7B-Instruct")
    
    # Load state dict
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    
    print(f"Loaded encoder from {checkpoint_path}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Global step: {checkpoint.get('global_step', 'N/A')}")
    print(f"  - Best loss: {checkpoint.get('best_loss', 'N/A'):.4f}")
    
    return encoder


def main():
    parser = argparse.ArgumentParser(description="Test trained encoder")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to encoder checkpoint (.pt file)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/generated",
        help="Directory to save output JSON"
    )
    parser.add_argument(
        "--llm_model", 
        type=str, 
        default="Qwen2.5-7B-Instruct",
        help="LLM model name"
    )
    parser.add_argument(
        "--tool_registry", 
        type=str, 
        default="data/tool_registry/tools.json",
        help="Path to tool registry"
    )
    parser.add_argument(
        "--profiling_data", 
        type=str, 
        default="data/profiling/profiling.csv",
        help="Path to profiling data"
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
        default=100,
        help="Maximum new tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("TOMAS-LLM Encoder Testing")
    print("="*80)
    
    # 1. Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = EncoderPretrainDataset(
        tool_registry_path=args.tool_registry,
        profiling_data_path=args.profiling_data,
        tokenizer_name=args.llm_model,
        augmentation_mode='none',  # No augmentation for testing
        num_augmented_copies=1,
        use_variation=False,
        seed=42
    )
    
    # Create subset if needed
    if args.num_samples and args.num_samples < len(dataset):
        indices = list(range(args.num_samples))
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Batch size: {args.batch_size}")
    
    # 2. Load encoder
    print("\n[2/4] Loading encoder from checkpoint...")
    encoder = load_encoder_from_checkpoint(
        args.checkpoint,
        args.llm_model,
        args.device
    )
    
    # 3. Load LLM
    print("\n[3/4] Loading LLM backbone...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        trust_remote_code=True
    )
    
    # 4. Run tests
    print("\n[4/4] Running tests...")
    tester = EncoderTester(encoder, llm_model, tokenizer, args.device)
    results = tester.test_samples(dataloader, args.num_samples)
    
    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, 
        f"encoder_test_results_{timestamp}.json"
    )
    
    output_data = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "llm_model": args.llm_model,
            "num_samples": len(results),
            "timestamp": timestamp,
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"  - Total samples tested: {len(results)}")
    
    # Print some example results
    print("\n" + "="*80)
    print("Sample Results (first 3):")
    print("="*80)
    for i, result in enumerate(results[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Tool ID: {result['tool_id']}")
        print(f"Ground Truth: {result['ground_truth'][:100]}...")
        print(f"Prediction:   {result['prediction'][:100]}...")


if __name__ == "__main__":
    main()
