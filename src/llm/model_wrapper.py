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
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn

from .qwen_backbone import QwenBackbone, ContextProjector
from ..decoders import (
    TokenTypeGate,
    ToolClassifier,
    ResourceRegressor,
    OutputParser,
    ToolPlan,
    TOOL_PLAN_TOKEN_OFFSET
)

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
    7. Decode outputs with output heads (Section 4.1):
       - TokenTypeGate: Route tokens to appropriate heads
       - ToolClassifier: Select tool from registry
       - ResourceRegressor: Allocate CPU/GPU resources
       - OutputParser: Convert to executable ToolPlan
    """
    
    def __init__(
        self,
        qwen_backbone: QwenBackbone,
        context_projector: ContextProjector,
        task_encoder: Optional[nn.Module] = None,
        latency_predictor: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        tool_encoder: Optional[nn.Module] = None,
        # Output heads (Section 4.1)
        token_gate: Optional[TokenTypeGate] = None,
        tool_classifier: Optional[ToolClassifier] = None,
        resource_regressor: Optional[ResourceRegressor] = None,
        output_parser: Optional[OutputParser] = None,
        tool_id_to_name: Optional[Dict[int, str]] = None,
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
            token_gate: Token type gate for routing (Section 4.1)
            tool_classifier: Tool classifier head (Section 4.1)
            resource_regressor: Resource regressor head (Section 4.1)
            output_parser: Output parser (Section 4.1)
            tool_id_to_name: Mapping from tool ID to tool name
        """
        super().__init__()
        
        self.qwen = qwen_backbone
        self.projector = context_projector
        
        # Optional components (can be set later)
        self.task_encoder = task_encoder
        self.latency_predictor = latency_predictor
        self.temporal_encoder = temporal_encoder
        self.tool_encoder = tool_encoder
        
        # Output heads (Section 4.1)
        self.token_gate = token_gate
        self.tool_classifier = tool_classifier
        self.resource_regressor = resource_regressor
        self.output_parser = output_parser
        self.tool_id_to_name = tool_id_to_name or {}
    
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
        output_hidden_states: bool = True,  # Need hidden states for output heads
        return_dict: bool = True,
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
            output_hidden_states: Whether to output hidden states (needed for output heads)
            return_dict: Whether to return dict output
        
        Returns:
            Model outputs (with hidden_states if output_hidden_states=True)
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
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def parse_output(
        self,
        generated_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
    ) -> List[ToolPlan]:
        """
        Parse generation output into executable tool plans.
        
        Args:
            generated_ids: (batch, seq) - Generated token IDs
            hidden_states: (batch, seq, hidden_dim) - LLM hidden states
            tool_embeddings: (num_tools, tool_dim) - Tool embeddings
            available_resources: (4,) or (batch, 4) - Available resources [optional]
        
        Returns:
            List of ToolPlan objects (length = batch_size)
        
        Raises:
            RuntimeError: If output_parser is not initialized
        """
        if self.output_parser is None:
            raise RuntimeError(
                "output_parser not initialized. "
                "Please set model.output_parser before calling parse_output()"
            )
        
        return self.output_parser.parse(
            generated_ids=generated_ids,
            hidden_states=hidden_states,
            tool_embeddings=tool_embeddings,
            available_resources=available_resources
        )
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        user_task: Optional[str] = None,
        predict_latency: bool = True,
        tool_schemas: Optional[List[Dict]] = None,
        tool_embeddings: Optional[torch.Tensor] = None,
        available_resources: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        parse_output: bool = False,
        return_dict_in_generate: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Union[List[str], Tuple[List[str], List[ToolPlan]]]:
        """
        Generate text with full context integration and optional output parsing.
        
        Args:
            prompt: Generation prompt
            user_task: User task for context
            predict_latency: Whether to include temporal context
            tool_schemas: Available tools
            tool_embeddings: (num_tools, tool_dim) - For output parsing
            available_resources: (4,) - For output parsing
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            do_sample: Sample or greedy
            parse_output: Whether to parse output into ToolPlan
            return_dict_in_generate: Return dict with hidden states
            output_hidden_states: Output hidden states for parsing
        
        Returns:
            If parse_output=False: List of generated texts
            If parse_output=True: Tuple of (texts, tool_plans)
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
        
        # Enable hidden states if parsing is requested
        if parse_output:
            return_dict_in_generate = True
            output_hidden_states = True
        
        # Generate with Qwen
        outputs = self.qwen.generate(
            input_text=prompt,
            context_embeddings=projected_context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_dict_in_generate=return_dict_in_generate,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        # Parse output if requested
        if parse_output:
            if not return_dict_in_generate:
                logger.warning("parse_output=True but return_dict_in_generate=False, cannot parse")
                return outputs
            
            if tool_embeddings is None:
                raise ValueError("tool_embeddings required for parse_output=True")
            
            # Extract generated sequences and hidden states
            # outputs is GenerateOutput with .sequences and .hidden_states
            generated_ids = outputs.sequences  # (batch, seq_len)
            
            # Get last layer hidden states
            # hidden_states is tuple of (num_layers, (num_gen_steps, (batch, 1, hidden_dim)))
            # We need to reconstruct the full sequence hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Concatenate hidden states across generation steps
                last_layer_hidden = [h[-1] for h in outputs.hidden_states]  # List of (batch, 1, hidden_dim)
                hidden_seq = torch.cat(last_layer_hidden, dim=1)  # (batch, gen_len, hidden_dim)
            else:
                raise RuntimeError("No hidden states in generation output")
            
            # Parse into ToolPlan
            tool_plans = self.parse_output(
                generated_ids=generated_ids,
                hidden_states=hidden_seq,
                tool_embeddings=tool_embeddings,
                available_resources=available_resources
            )
            
            # Decode text
            texts = self.qwen.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return texts, tool_plans
        
        # Return generated texts only
        return outputs
    
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
        
        # Create output heads (Section 4.1)
        token_gate = None
        tool_classifier = None
        resource_regressor = None
        output_parser = None
        tool_id_to_name = None
        
        decoder_config = config.get('decoder', {})
        if decoder_config.get('enabled', False):
            try:
                vocab_size = qwen.model.config.vocab_size
                hidden_dim = qwen.model.config.hidden_size
                
                # Create TokenTypeGate
                token_gate = TokenTypeGate(
                    vocab_size=vocab_size,
                    hidden_dim=hidden_dim
                )
                logger.info(f"TokenTypeGate created (vocab_size={vocab_size})")
                
                # Create ToolClassifier
                tool_dim = decoder_config.get('tool_dim', 1024)
                num_tools = decoder_config.get('num_tools', 10)
                tool_classifier = ToolClassifier(
                    hidden_dim=hidden_dim,
                    tool_dim=tool_dim,
                    num_tools=num_tools,
                    use_attention=decoder_config.get('use_attention', True)
                )
                logger.info(f"ToolClassifier created (num_tools={num_tools})")
                
                # Create ResourceRegressor
                resource_regressor = ResourceRegressor(
                    hidden_dim=hidden_dim,
                    use_constraint=decoder_config.get('use_constraint', True)
                )
                logger.info("ResourceRegressor created")
                
                # Create OutputParser
                output_parser = OutputParser(
                    tool_classifier=tool_classifier,
                    resource_regressor=resource_regressor,
                    token_gate=token_gate,
                    tool_id_to_name=decoder_config.get('tool_id_to_name')
                )
                logger.info("OutputParser created")
                
                tool_id_to_name = decoder_config.get('tool_id_to_name')
                
            except Exception as e:
                logger.warning(f"Could not create output heads: {e}")
        
        return cls(
            qwen_backbone=qwen,
            context_projector=projector,
            task_encoder=task_encoder,
            latency_predictor=latency_predictor,
            temporal_encoder=temporal_encoder,
            tool_encoder=tool_encoder,
            token_gate=token_gate,
            tool_classifier=tool_classifier,
            resource_regressor=resource_regressor,
            output_parser=output_parser,
            tool_id_to_name=tool_id_to_name
        )
