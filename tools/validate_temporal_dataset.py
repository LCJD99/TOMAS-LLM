"""
Data Validation Tool for Temporal Pretrain Dataset.

Validates and visualizes synthetic resource curves and text descriptions.
"""

import argparse
import logging
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.temporal_pretrain_dataset import TemporalPretrainDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_samples(dataset: TemporalPretrainDataset, num_samples: int = 20, output_dir: str = "data/temporal_pretrain"):
    """
    Visualize random samples from dataset.
    
    Args:
        dataset: Dataset instance
        num_samples: Number of samples to visualize
        output_dir: Output directory for HTML report
    """
    import random
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Create HTML report
    html_path = os.path.join(output_dir, "validation_report.html")
    
    with open(html_path, 'w') as f:
        f.write("<html><head><title>Temporal Dataset Validation</title>")
        f.write("<style>")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
        f.write(".sample { border: 1px solid #ccc; padding: 15px; margin: 20px 0; }")
        f.write(".prompt { background-color: #e3f2fd; padding: 10px; margin: 5px 0; }")
        f.write(".target { background-color: #fff3e0; padding: 10px; margin: 5px 0; }")
        f.write(".metadata { color: #666; font-size: 0.9em; }")
        f.write("img { max-width: 100%; height: auto; }")
        f.write("</style></head><body>")
        f.write(f"<h1>Temporal Pretrain Dataset Validation Report</h1>")
        f.write(f"<p>Total samples: {len(dataset)}</p>")
        
        # Task type distribution
        task_counts = dataset._count_task_types()
        f.write(f"<h2>Task Type Distribution</h2>")
        f.write(f"<ul>")
        for task_type, count in sorted(task_counts.items()):
            percentage = 100 * count / len(dataset)
            f.write(f"<li>Type {task_type}: {count} ({percentage:.1f}%)</li>")
        f.write(f"</ul>")
        
        # Sample visualizations
        f.write(f"<h2>Sample Visualizations ({num_samples} samples)</h2>")
        
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            
            f.write(f'<div class="sample">')
            f.write(f'<h3>Sample {i+1} (Index {idx})</h3>')
            f.write(f'<p class="metadata">Task Type: {sample["task_type"]} | ')
            f.write(f'Timesteps: {sample["metadata"]["num_timesteps"]} | ')
            f.write(f'Duration: {sample["metadata"]["duration_s"]:.1f}s</p>')
            
            # Plot curve
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Sample {i+1}: Resource Timeline (Task Type {sample["task_type"]})')
            
            curve = sample['curve'].numpy()
            resource_names = ['CPU Cores', 'CPU Memory (GB)', 'GPU SM (%)', 'GPU Memory (GB)']
            
            for j, (ax, name) in enumerate(zip(axes.flatten(), resource_names)):
                ax.plot(curve[:, j], linewidth=2)
                ax.set_xlabel('Timestep')
                ax.set_ylabel(name)
                ax.set_title(name)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"sample_{i+1}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Add to HTML
            f.write(f'<img src="sample_{i+1}.png" alt="Sample {i+1}">')
            
            # Text descriptions
            f.write(f'<div class="prompt"><strong>Prompt:</strong> {sample["metadata"]["prompt"]}</div>')
            f.write(f'<div class="target"><strong>Target:</strong> {sample["metadata"]["target"]}</div>')
            
            # Token info
            f.write(f'<p class="metadata">Prompt tokens: {sample["prompt_ids"].size(0)} | ')
            f.write(f'Target tokens: {sample["target_ids"].size(0)}</p>')
            
            f.write('</div>')
        
        f.write("</body></html>")
    
    logger.info(f"Validation report saved to {html_path}")
    return html_path


def check_text_lengths(dataset: TemporalPretrainDataset):
    """Check distribution of text lengths."""
    prompt_lengths = []
    target_lengths = []
    
    logger.info("Analyzing text lengths...")
    
    for i in range(min(1000, len(dataset))):
        sample = dataset[i]
        prompt_lengths.append(sample['prompt_ids'].size(0))
        target_lengths.append(sample['target_ids'].size(0))
    
    logger.info(f"Prompt length - Mean: {np.mean(prompt_lengths):.1f}, "
                f"Min: {np.min(prompt_lengths)}, Max: {np.max(prompt_lengths)}")
    logger.info(f"Target length - Mean: {np.mean(target_lengths):.1f}, "
                f"Min: {np.min(target_lengths)}, Max: {np.max(target_lengths)}")
    
    return prompt_lengths, target_lengths


def validate_physical_constraints(dataset: TemporalPretrainDataset, num_samples: int = 1000):
    """Validate that curves satisfy physical constraints."""
    logger.info("Validating physical constraints...")
    
    violations = 0
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        curve = sample['curve']
        
        if not dataset.curve_synthesizer.validate_physical_constraints(curve):
            violations += 1
            logger.warning(f"Sample {i} violates physical constraints")
    
    if violations == 0:
        logger.info("✓ All samples satisfy physical constraints")
    else:
        logger.warning(f"✗ {violations}/{num_samples} samples violate constraints")
    
    return violations


def main():
    parser = argparse.ArgumentParser(description="Validate Temporal Pretrain Dataset")
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--num_visualize', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='data/temporal_pretrain',
                       help='Output directory for validation report')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Model name for tokenizer')
    
    args = parser.parse_args()
    
    logger.info(f"Initializing tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    logger.info(f"Creating dataset with {args.num_samples} samples")
    dataset = TemporalPretrainDataset(
        num_samples=args.num_samples,
        tokenizer=tokenizer,
        type_distribution={"A": 0.4, "B": 0.3, "C": 0.3},
        max_length=256,
        seed=42
    )
    
    # Task type distribution
    task_counts = dataset._count_task_types()
    logger.info(f"Task type distribution: {task_counts}")
    
    # Check text lengths
    check_text_lengths(dataset)
    
    # Validate constraints
    validate_physical_constraints(dataset, num_samples=min(1000, args.num_samples))
    
    # Visualize samples
    logger.info(f"Generating visualization for {args.num_visualize} samples")
    report_path = visualize_samples(dataset, num_samples=args.num_visualize, output_dir=args.output_dir)
    
    logger.info(f"✓ Validation complete! Report: {report_path}")


if __name__ == "__main__":
    main()
