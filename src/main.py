#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
TOMAS-LLM 主入口文件
给定一条用户指令，输出 Tool Plan（JSON格式）
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_tool_data


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """配置日志"""
    log_level = getattr(logging, config['logging']['level'])
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if config['logging']['log_to_file']:
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_dir / 'tomas_llm.log', encoding='utf-8')
        )
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TOMAS-LLM: Resource-Aware Tool Planning System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="User task description"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the tool plan (JSON)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TOMAS-LLM System Starting...")
    logger.info("=" * 60)
    
    # Load tool registry and profiling data
    logger.info("Loading tool registry and profiling data...")
    dataset = load_tool_data(
        tool_registry_path=config['data']['tool_registry_path'],
        profiling_path=config['data']['profiling_path'],
        normalize=True
    )
    logger.info(f"Loaded {len(dataset)} tool samples")
    
    # Display available tools
    tool_names = list(set(s['tool_name'] for s in dataset.get_samples()))
    logger.info(f"Available tools: {tool_names}")
    
    # TODO: Initialize encoders
    logger.info("Initializing encoders... (not yet implemented)")
    # tool_encoder = ToolEncoder(config)
    # resource_mlp = ResourceMLP(config)
    # temporal_encoder = TemporalEncoder(config)
    
    # TODO: Load LLM backbone
    logger.info("Loading LLM backbone (Qwen2.5-7B)... (not yet implemented)")
    # backbone = QwenBackbone(config)
    
    # TODO: Initialize output heads
    logger.info("Initializing output heads... (not yet implemented)")
    # tool_classifier = ToolClassifier(config)
    # resource_regressor = ResourceRegressor(config)
    
    logger.info(f"User Task: {args.task}")
    
    # TODO: Generate Tool Plan
    # 1. Encode user task
    # 2. Get current system resource snapshot + predict future timeline
    # 3. Generate plan through LLM
    # 4. Parse to structured JSON
    
    # Minimal prototype: return a placeholder plan
    tool_plan = {
        "task": args.task,
        "tool_id": "image_classification",
        "tool_name": "image_classification",
        "resource_config": {
            "cpu_core": 4,
            "cpu_mem_gb": 8.0,
            "gpu_sm": 40,
            "gpu_mem_gb": 4.0
        },
        "expected_latency_ms": 350,
        "status": "placeholder - full pipeline not yet implemented"
    }
    
    logger.info("Generated Tool Plan:")
    logger.info(json.dumps(tool_plan, indent=2, ensure_ascii=False))
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tool_plan, f, indent=2, ensure_ascii=False)
        logger.info(f"Tool Plan saved to: {args.output}")
    
    logger.info("=" * 60)
    logger.info("TOMAS-LLM System Finished")
    logger.info("=" * 60)
    
    return tool_plan


if __name__ == "__main__":
    main()
