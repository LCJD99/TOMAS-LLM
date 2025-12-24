"""
Complete TOMAS-LLM Model Wrapper.

Integrates all components:
- Section 1.x: Tool encoders (Left Panel)
- Section 2.1: User task embedding
- Section 2.2: Latency prediction
- Section 2.3: System timeline
- Section 2.4: Temporal encoding
- Section 3.x: LLM backbone (this module)
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .qwen_backbone import QwenBackbone, ContextProjector

logger = logging.getLogger(__name__)


class TOMASSLLMModel(nn.Module):
    """
    Complete TOMAS-LLM model with all components integrated.
    
    Pipeline:
    1. Encode user task (Section 2.1)
    2. Predict latency T_inf (Section 2.2)
    3. Get resource snapshot at T_inf (Section 2.3)
    4. Encode temporal features (Section 2.4)
    5. Project contexts to LLM dimension
    6. Generate with Qwen2.5 (Section 3.x)
    """
    
    def __init__(
        self,
        qwen_backbone: QwenBackbone,
        context_projector: ContextProjector,
        task_encoder: Optional[nn.Module] = None,
        latency_predictor: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        tool_encoder: Optional[nn.Module] = None
    ):
        """
        Initialize TOMAS-LLM model.
        
        Args:
            qwen_backbone: Qwen2.5 LLM backbone
            context_projector: Context embedding projector
            task_encoder: User task encoder (Section 2.1)
            latency_predictor: Latency predictor (Section 2.2)
            temporal_encoder: Temporal encoder (Section 2.4)
            tool_encoder: Tool encoder (Section 1.x)
        """
        super().__init__()
        
        self.qwen = qwen_backbone
        self.projector = context_projector
        
        # Optional components (can be set later)
        self.task_encoder = task_encoder
        self.latency_predictor = latency_predictor
        self.temporal_encoder = temporal_encoder
        self.tool_encoder = tool_encoder
    
    def encode_context(
        self,
        user_task: Optional[str] = None,
        predict_latency: bool = True,
        tool_schemas: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all context information.
        
        Args:
            user_task: User task description
            predict_latency: Whether to predict latency and encode temporal
            tool_schemas: Tool schemas for encoding
        
        Returns:
            Dictionary with all context embeddings
        """
        context = {}
        
        # 1. Encode user task (Section 2.1)
        if user_task is not None and self.task_encoder is not None:
            task_output = self.task_encoder(user_task)
            # Handle tuple output (pooled, sequence)
            task_embedding = task_output[0] if isinstance(task_output, tuple) else task_output
            context['task_embedding'] = task_embedding
        
        # 2. Predict latency (Section 2.2) and encode temporal (Section 2.4)
        if predict_latency and self.latency_predictor is not None and self.temporal_encoder is not None:
            t_inf = self.latency_predictor()
            v_temporal = self.temporal_encoder(t_inf)
            context['v_temporal'] = v_temporal
            context['t_inf'] = t_inf
        
        # 3. Encode tools (Section 1.x)
        if tool_schemas is not None and self.tool_encoder is not None:
            # Placeholder - actual implementation depends on tool encoder API
            # tool_embeddings = self.tool_encoder(tool_schemas)
            # context['tool_embeddings'] = tool_embeddings
            pass
        
        return context
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through complete model.
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            inputs_embeds: Pre-computed embeddings
            context_embeddings: Context embeddings (v_temporal, task, tools)
            labels: Labels for training
        
        Returns:
            Model outputs
        """
        # Project context embeddings
        if context_embeddings is not None:
            projected_context = self.projector(
                v_temporal=context_embeddings.get('v_temporal'),
                task_embedding=context_embeddings.get('task_embedding'),
                tool_embeddings=context_embeddings.get('tool_embeddings')
            )
        else:
            projected_context = None
        
        # Prepare inputs with context
        if input_ids is not None and projected_context is not None:
            inputs_embeds, attention_mask = self.qwen.prepare_inputs_with_context(
                input_ids,
                attention_mask,
                projected_context
            )
            input_ids = None  # Use inputs_embeds instead
        
        # Forward through Qwen
        return self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        user_task: Optional[str] = None,
        predict_latency: bool = True,
        tool_schemas: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text with full context integration.
        
        Args:
            prompt: Generation prompt
            user_task: User task for context
            predict_latency: Whether to include temporal context
            tool_schemas: Available tools
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            do_sample: Sample or greedy
            **kwargs: Additional generation args
        
        Returns:
            Generated texts
        """
        # Encode context
        context = self.encode_context(
            user_task=user_task,
            predict_latency=predict_latency,
            tool_schemas=tool_schemas
        )
        
        # Project context
        if context:
            projected_context = self.projector(
                v_temporal=context.get('v_temporal'),
                task_embedding=context.get('task_embedding'),
                tool_embeddings=context.get('tool_embeddings')
            )
        else:
            projected_context = None
        
        # Generate with Qwen
        return self.qwen.generate(
            input_text=prompt,
            context_embeddings=projected_context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TOMASSLLMModel':
        """
        Create complete TOMAS-LLM model from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized TOMASSLLMModel
        """
        # Create LLM backbone
        qwen = QwenBackbone.from_config(config)
        
        # Create context projector
        projector = ContextProjector.from_config(config)
        
        # Create optional components
        task_encoder = None
        latency_predictor = None
        temporal_encoder = None
        tool_encoder = None
        
        # Try to import and create components if enabled
        try:
            from context.user_task import UserTaskEncoder
            task_encoder = UserTaskEncoder.from_config(config)
            logger.info("Task encoder loaded")
        except Exception as e:
            logger.warning(f"Could not load task encoder: {e}")
        
        try:
            from context.latency_predictor import LatencyPredictor
            latency_config = config.get('runtime', {}).get('latency_predictor', {})
            latency_predictor = LatencyPredictor.from_config(config)
            logger.info("Latency predictor loaded")
        except Exception as e:
            logger.warning(f"Could not load latency predictor: {e}")
        
        try:
            from context.temporal_encoder import TemporalEncoder
            temporal_encoder = TemporalEncoder.from_config(config)
            logger.info("Temporal encoder loaded")
        except Exception as e:
            logger.warning(f"Could not load temporal encoder: {e}")
        
        return cls(
            qwen_backbone=qwen,
            context_projector=projector,
            task_encoder=task_encoder,
            latency_predictor=latency_predictor,
            temporal_encoder=temporal_encoder,
            tool_encoder=tool_encoder
        )
