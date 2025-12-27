"""
Visualize Temporal Encoder Predictions.

Generate an HTML report showing:
1. Resource timeline curves
2. Prompts and ground truth targets
3. Model predictions
4. Highlighting of differences

Usage:
    python tools/visualize_temporal_predictions.py \
        --checkpoint checkpoints/temporal_pretrain/best_model.pt \
        --num_samples 20 \
        --output logs/temporal_pretrain/predictions_visualization.html
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
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


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    config_path = "configs/pretrain_temporal.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Load LLM and tokenizer
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_config['llm_name'],
        trust_remote_code=True,
        torch_dtype=torch.float16 if config['training'].get('fp16', False) else torch.float32
    )
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, checkpoint


def generate_curve_svg(curve: np.ndarray, width: int = 600, height: int = 200) -> str:
    """
    Generate SVG visualization of resource curve.
    
    Args:
        curve: Resource curve array (num_timesteps, 4)
        width: SVG width
        height: SVG height
    
    Returns:
        SVG string
    """
    num_timesteps = curve.shape[0]
    resource_names = ['CPU Cores', 'CPU Mem (GB)', 'GPU SM (%)', 'GPU Mem (GB)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Normalize curves to 0-1 range for plotting
    normalized_curves = []
    for i in range(4):
        col = curve[:, i]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            normalized = (col - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(col)
        normalized_curves.append(normalized)
    
    # Generate SVG paths
    svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    
    # Background
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="#f8f9fa"/>')
    
    # Grid lines
    for i in range(5):
        y = (height - 40) * i / 4 + 20
        svg_parts.append(
            f'<line x1="40" y1="{y}" x2="{width-20}" y2="{y}" '
            f'stroke="#e0e0e0" stroke-width="1"/>'
        )
    
    # Plot each resource
    for idx, (norm_curve, name, color) in enumerate(zip(normalized_curves, resource_names, colors)):
        # Generate path
        points = []
        for t, val in enumerate(norm_curve):
            x = 40 + (width - 60) * t / (num_timesteps - 1)
            y = height - 20 - (height - 40) * val
            points.append(f"{x},{y}")
        
        path_d = "M " + " L ".join(points)
        svg_parts.append(
            f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        
        # Legend
        legend_y = 15 + idx * 20
        svg_parts.append(
            f'<circle cx="{width - 150}" cy="{legend_y}" r="4" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{width - 140}" y="{legend_y + 4}" font-size="12" fill="#333">{name}</text>'
        )
    
    svg_parts.append('</svg>')
    return ''.join(svg_parts)


def highlight_diff(text1: str, text2: str) -> Tuple[str, str]:
    """
    Highlight differences between two texts.
    
    Args:
        text1: First text (ground truth)
        text2: Second text (prediction)
    
    Returns:
        Tuple of HTML-formatted texts with differences highlighted
    """
    words1 = text1.split()
    words2 = text2.split()
    
    highlighted1 = []
    highlighted2 = []
    
    # Simple word-level diff
    max_len = max(len(words1), len(words2))
    for i in range(max_len):
        w1 = words1[i] if i < len(words1) else ""
        w2 = words2[i] if i < len(words2) else ""
        
        if w1 != w2:
            if w1:
                highlighted1.append(f'<span style="background-color: #ffcccc;">{w1}</span>')
            if w2:
                highlighted2.append(f'<span style="background-color: #ffcccc;">{w2}</span>')
        else:
            if w1:
                highlighted1.append(w1)
            if w2:
                highlighted2.append(w2)
    
    return ' '.join(highlighted1), ' '.join(highlighted2)


def generate_html_report(
    samples: List[dict],
    checkpoint_info: dict,
    output_path: str
):
    """
    Generate HTML visualization report.
    
    Args:
        samples: List of sample dictionaries
        checkpoint_info: Checkpoint metadata
        output_path: Output HTML file path
    """
    html_parts = ['''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Temporal Encoder Predictions Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .sample {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .sample-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }
        .sample-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .task-type {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        .task-A { background-color: #e3f2fd; color: #1976d2; }
        .task-B { background-color: #fff3e0; color: #f57c00; }
        .task-C { background-color: #e8f5e9; color: #388e3c; }
        .curve-container {
            margin: 20px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 8px;
        }
        .text-section {
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
        }
        .text-section h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #333;
        }
        .prompt {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .ground-truth {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
        }
        .prediction {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }
        .text-content {
            font-family: 'Courier New', monospace;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
    </style>
</head>
<body>
''']
    
    # Header
    html_parts.append(f'''
    <div class="header">
        <h1>Temporal Encoder Predictions Visualization</h1>
        <p>Checkpoint: {checkpoint_info.get('checkpoint_path', 'N/A')}</p>
        <p>Epoch: {checkpoint_info.get('epoch', 'N/A')} | Step: {checkpoint_info.get('global_step', 'N/A')}</p>
        <p>Train Loss: {checkpoint_info.get('train_loss', 0):.4f} | Val Loss: {checkpoint_info.get('val_loss', 0):.4f}</p>
    </div>
''')
    
    # Statistics
    task_counts = {'A': 0, 'B': 0, 'C': 0}
    for sample in samples:
        task_counts[sample['task_type']] += 1
    
    html_parts.append(f'''
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(samples)}</div>
            <div class="stat-label">Total Samples</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{task_counts['A']}</div>
            <div class="stat-label">Type A (Trend)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{task_counts['B']}</div>
            <div class="stat-label">Type B (Bottleneck)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{task_counts['C']}</div>
            <div class="stat-label">Type C (Feasibility)</div>
        </div>
    </div>
''')
    
    # Samples
    for idx, sample in enumerate(samples):
        curve_svg = generate_curve_svg(sample['curve'])
        
        gt_highlighted, pred_highlighted = highlight_diff(
            sample['ground_truth'],
            sample['prediction']
        )
        
        html_parts.append(f'''
    <div class="sample">
        <div class="sample-header">
            <div class="sample-number">Sample #{idx + 1}</div>
            <div class="task-type task-{sample['task_type']}">Task {sample['task_type']}</div>
        </div>
        
        <div class="curve-container">
            <h3>Resource Timeline</h3>
            {curve_svg}
        </div>
        
        <div class="text-section prompt">
            <h3>Prompt</h3>
            <div class="text-content">{sample['prompt']}</div>
        </div>
        
        <div class="text-section ground-truth">
            <h3>Ground Truth</h3>
            <div class="text-content">{gt_highlighted}</div>
        </div>
        
        <div class="text-section prediction">
            <h3>Model Prediction</h3>
            <div class="text-content">{pred_highlighted}</div>
        </div>
    </div>
''')
    
    html_parts.append('''
</body>
</html>
''')
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    logger.info(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Temporal Encoder Predictions")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/pretrain_temporal.yaml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--output', type=str, 
                       default='logs/temporal_pretrain/predictions_visualization.html',
                       help='Output HTML file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    logger.info(f"Creating dataset for visualization")
    dataset = TemporalPretrainDataset(
        num_samples=args.num_samples,
        tokenizer=tokenizer,
        type_distribution=config['data']['type_distribution'],
        max_length=config['data']['max_length'],
        seed=42  # Fixed seed for reproducibility
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Collect samples
    logger.info(f"Generating predictions for {args.num_samples} samples...")
    samples = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            curve = batch['curve'].to(device)
            prompt_ids = batch['prompt_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Generate prediction
            generated_ids = model.generate(
                curve=curve,
                prompt_ids=prompt_ids,
                max_length=256,
                num_beams=1,
                do_sample=False
            )
            
            # Decode texts
            prompt_len = prompt_ids[0].size(0)
            prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            ground_truth_text = tokenizer.decode(target_ids[0], skip_special_tokens=True)
            prediction_text = tokenizer.decode(
                generated_ids[0, prompt_len:],
                skip_special_tokens=True
            )
            
            # Store sample
            samples.append({
                'curve': curve[0].cpu().numpy(),
                'prompt': prompt_text,
                'ground_truth': ground_truth_text,
                'prediction': prediction_text,
                'task_type': batch['task_type'][0]
            })
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    checkpoint_info = {
        'checkpoint_path': args.checkpoint,
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss']
    }
    
    generate_html_report(samples, checkpoint_info, args.output)
    
    logger.info(f"Visualization complete! Open {args.output} in a web browser.")


if __name__ == "__main__":
    main()
