#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
TOMAS-LLM 推理脚本

完整推理流程：
1. 加载配置和工具注册表
2. 初始化 Qwen2.5-7B-Instruct 模型
3. 初始化所有编码器（用户任务、时序、工具）
4. 加载训练好的 checkpoint（如果提供）
5. 编码用户任务和系统上下文
6. 生成工具计划
7. 解析输出并保存
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import yaml

# 添加 src 目录到路径

from src.data.loader import ToolRegistryLoader
from src.llm.qwen_backbone import QwenBackbone, ContextProjector
from src.llm.model_wrapper import TOMASSLLMModel
from src.context.user_task import TaskEmbedding, UserTaskEncoder
from src.context.latency_predictor import LatencyPredictor
from src.context.temporal_encoder import TemporalEncoder
from src.decoders.output_parser import ToolPlan


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_tool_registry(registry_path: str, logger: logging.Logger) -> Dict[int, str]:
    """
    加载工具注册表
    
    Returns:
        tool_id_to_name: 工具ID到工具名称的映射
    """
    logger.info(f"Loading tool registry from: {registry_path}")
    loader = ToolRegistryLoader(registry_path)

    tools, tool_dict = loader.load()
    
    # 创建工具ID到名称的映射
    tool_id_to_name = {i: tool.name for i, tool in enumerate(tools)}
    tool_names = [tool.name for tool in tools]
    
    logger.info(f"Loaded {len(tools)} tools: {tool_names}")
    return tool_id_to_name, tools


def initialize_model(
    config: Dict[str, Any],
    tool_id_to_name: Dict[int, str],
    logger: logging.Logger
) -> TOMASSLLMModel:
    """
    initialize TOMAS-LLM model with all components.
    
    Components:
    - Qwen2.5-7B-Instruct backbone
    - Context projector
    - User task encoder
    - Latency predictor
    - Temporal encoder
    """
    logger.info("Initializing TOMAS-LLM model...")
    
    # 1. 初始化 Qwen backbone
    logger.info("Loading Qwen2.5-7B-Instruct backbone...")
    qwen = QwenBackbone.from_config(config)
    logger.info(f"Qwen loaded: hidden_dim={qwen.hidden_dim}, vocab_size={qwen.vocab_size}")
    
    # 2. 初始化 Context Projector
    logger.info("Creating context projector...")
    projector = ContextProjector.from_config(config)
    
    # 3. 初始化 Task Encoder
    logger.info("Creating user task encoder...")
    try:
        task_encoder = UserTaskEncoder.from_config(config)
        logger.info("Task encoder initialized")
    except Exception as e:
        logger.warning(f"Could not create task encoder: {e}")
        task_encoder = None
    
    # 4. 初始化 Latency Predictor
    logger.info("Creating latency predictor...")
    try:
        latency_predictor = LatencyPredictor.from_config(config)
        logger.info("Latency predictor initialized")
    except Exception as e:
        logger.warning(f"Could not create latency predictor: {e}")
        latency_predictor = None
    
    # 5. 初始化 Temporal Encoder
    logger.info("Creating temporal encoder...")
    try:
        temporal_encoder = TemporalEncoder.from_config(config)
        logger.info("Temporal encoder initialized")
    except Exception as e:
        logger.warning(f"Could not create temporal encoder: {e}")
        temporal_encoder = None
    
    # 6. 创建完整模型
    model = TOMASSLLMModel(
        qwen_backbone=qwen,
        context_projector=projector,
        task_encoder=task_encoder,
        latency_predictor=latency_predictor,
        temporal_encoder=temporal_encoder,
        tool_id_to_name=tool_id_to_name
    )
    
    logger.info("Model initialization complete")
    return model


def load_checkpoint(
    model: TOMASSLLMModel,
    checkpoint_path: str,
    logger: logging.Logger
) -> TOMASSLLMModel:
    """
    加载训练好的模型权重
    
    注意：这里假设checkpoint是通过 torch.save(model.state_dict(), path) 保存的
    如果checkpoint格式不同，需要相应调整
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        logger.warning("Using randomly initialized weights")
        return model
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded successfully")
        
        # 打印一些统计信息
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                logger.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.warning("Continuing with randomly initialized weights")
    
    return model


def build_inference_prompt(
    user_task: str,
    tools: List,
    include_tool_descriptions: bool = True
) -> str:
    """
    构建推理prompt
    
    格式：
    System: You are a resource-aware tool planning assistant.
    
    Available Tools:
    - tool_name_1: description
    - tool_name_2: description
    ...
    
    User Task: {user_task}
    
    Plan: <TOOL_PLAN>
    """
    prompt_parts = [
        "System: You are a resource-aware tool planning assistant that selects the most appropriate tool and allocates computational resources for user tasks.",
        ""
    ]
    
    if include_tool_descriptions:
        prompt_parts.append("Available Tools:")
        for tool in tools:
            prompt_parts.append(f"- {tool.name}: {tool.desc}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        f"User Task: {user_task}",
        "",
        "Select the best tool and generate resource allocation plan.",
        "Plan:"
    ])
    
    return "\n".join(prompt_parts)


def run_inference(
    model: TOMASSLLMModel,
    user_task: str,
    tools: List,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    执行推理
    
    Returns:
        推理结果字典
    """
    logger.info("=" * 60)
    logger.info("Starting inference pipeline...")
    logger.info("=" * 60)
    
    # 设置模型为评估模式
    model.eval()
    
    # 构建prompt
    logger.info("Building inference prompt...")
    prompt = build_inference_prompt(user_task, tools)
    logger.info(f"Prompt:\n{prompt}\n")
    
    # 获取生成配置
    gen_config = config.get('llm', {}).get('generation', {})
    max_new_tokens = gen_config.get('max_new_tokens', 1024)
    temperature = gen_config.get('temperature', 0.7)
    top_p = gen_config.get('top_p', 0.9)
    do_sample = gen_config.get('do_sample', True)
    
    logger.info("Generation settings:")
    logger.info(f"  max_new_tokens: {max_new_tokens}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  top_p: {top_p}")
    logger.info(f"  do_sample: {do_sample}")
    
    # 执行生成
    logger.info("Generating with Qwen2.5-7B-Instruct...")
    with torch.no_grad():
        try:
            outputs = model.generate(
                prompt=prompt,
                user_task=user_task,
                predict_latency=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            
            generated_text = outputs[0] if isinstance(outputs, list) else outputs
            logger.info(f"\nGenerated output:\n{generated_text}\n")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    # 解析输出
    # 注意：这里需要根据实际的输出格式来解析
    # 如果模型训练时使用了特殊的输出格式，需要相应的解析逻辑
    result = {
        "user_task": user_task,
        "generated_text": generated_text,
        "tool_plan": extract_tool_plan_from_text(generated_text, tools, logger)
    }
    
    return result


def extract_tool_plan_from_text(
    generated_text: str,
    tools: List,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    从生成的文本中提取工具计划
    
    这是一个简单的启发式方法，实际使用时应该基于训练时的输出格式
    """
    tool_names = [tool.name for tool in tools]
    
    # 尝试找到提到的工具
    selected_tool = None
    for tool_name in tool_names:
        if tool_name.lower() in generated_text.lower():
            selected_tool = tool_name
            break
    
    if selected_tool is None:
        # 默认选择第一个工具
        selected_tool = tool_names[0] if tool_names else "unknown"
        logger.warning(f"Could not identify tool from output, using default: {selected_tool}")
    
    # 提取或估算资源配置
    # 这里使用简单的默认值，实际应该从模型输出中解析
    tool_plan = {
        "tool_name": selected_tool,
        "tool_id": next((i for i, t in enumerate(tools) if t.name == selected_tool), 0),
        "confidence": 0.8,  # 默认置信度
        "resource_allocation": {
            "cpu_core": 4.0,
            "cpu_mem_gb": 8.0,
            "gpu_sm": 0.0,
            "gpu_mem_gb": 0.0
        }
    }
    
    logger.info(f"Extracted tool plan: {tool_plan}")
    return tool_plan


def main():
    parser = argparse.ArgumentParser(
        description="TOMAS-LLM Inference Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, will use pretrained if not provided)"
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
        default="output/tool_plan.json",
        help="Output file path for the tool plan"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu), overrides config"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("TOMAS-LLM Inference Pipeline")
    logger.info("=" * 80)
    logger.info(f"User Task: {args.task}")
    logger.info("")
    
    try:
        # 1. 加载配置
        logger.info("Step 1: Loading configuration...")
        config = load_config(args.config)
        
        # 覆盖设备配置
        if args.device:
            config['llm']['model']['device'] = args.device
            logger.info(f"Device overridden to: {args.device}")
        
        # 2. 加载工具注册表
        logger.info("\nStep 2: Loading tool registry...")
        tool_registry_path = config['data']['tool_registry_path']
        tool_id_to_name, tools = load_tool_registry(tool_registry_path, logger)
        
        # 3. 初始化模型
        logger.info("\nStep 3: Initializing model...")
        model = initialize_model(config, tool_id_to_name, logger)
        
        # 4. 加载checkpoint（如果提供）
        if args.checkpoint:
            logger.info("\nStep 4: Loading checkpoint...")
            model = load_checkpoint(model, args.checkpoint, logger)
        else:
            logger.info("\nStep 4: Skipping checkpoint loading (using pretrained Qwen weights)")
        
        # 5. 执行推理
        logger.info("\nStep 5: Running inference...")
        result = run_inference(model, args.task, tools, config, logger)
        
        # 6. 保存结果
        logger.info("\nStep 6: Saving results...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {args.output}")
        
        # 打印总结
        logger.info("\n" + "=" * 80)
        logger.info("Inference Summary")
        logger.info("=" * 80)
        logger.info(f"User Task: {args.task}")
        logger.info(f"Selected Tool: {result['tool_plan']['tool_name']}")
        logger.info(f"Resource Allocation:")
        for key, value in result['tool_plan']['resource_allocation'].items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Inference failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
