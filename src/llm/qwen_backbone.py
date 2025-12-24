"""
Qwen2.5 LLM Backbone Integration.

This module:
1. Loads Qwen2.5 models (7B for production, 0.5B for testing)
2. Projects context embeddings to LLM dimension
3. Injects temporal/task context as prefix tokens
4. Provides unified interface for text generation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class ContextProjector(nn.Module):
    """
    Projects various context embeddings to LLM hidden dimension.
    
    Projects:
    - v_temporal (256D) → LLM dimension (3584D for 7B, 896D for 0.5B)
    - task_embedding (896D) → LLM dimension
    - tool_embeddings (1024D) → LLM dimension
    """
    
    def __init__(
        self,
        llm_hidden_dim: int = 3584,        # Qwen2.5-7B hidden dim
        temporal_dim: int = 256,            # v_temporal dimension
        task_dim: int = 896,                # Task embedding dimension
        tool_dim: int = 1024,               # Tool encoding dimension
        num_temporal_tokens: int = 1,       # Number of temporal prefix tokens
        num_task_tokens: int = 1,           # Number of task prefix tokens
        dropout: float = 0.1
    ):
        """
        Initialize context projector.
        
        Args:
            llm_hidden_dim: LLM hidden dimension
            temporal_dim: Temporal embedding dimension (from Section 2.4)
            task_dim: Task embedding dimension (from Section 2.1)
            tool_dim: Tool encoding dimension (from Section 1.5)
            num_temporal_tokens: Number of temporal tokens to inject
            num_task_tokens: Number of task tokens to inject
            dropout: Dropout rate
        """
        super().__init__()
        
        self.llm_hidden_dim = llm_hidden_dim
        self.temporal_dim = temporal_dim
        self.task_dim = task_dim
        self.tool_dim = tool_dim
        self.num_temporal_tokens = num_temporal_tokens
        self.num_task_tokens = num_task_tokens
        
        # Temporal projection: 256D → LLM_dim × num_temporal_tokens
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, llm_hidden_dim * num_temporal_tokens),
            nn.LayerNorm(llm_hidden_dim * num_temporal_tokens),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(llm_hidden_dim * num_temporal_tokens, llm_hidden_dim * num_temporal_tokens)
        )
        
        # Task projection: 896D → LLM_dim × num_task_tokens
        self.task_proj = nn.Sequential(
            nn.Linear(task_dim, llm_hidden_dim * num_task_tokens),
            nn.LayerNorm(llm_hidden_dim * num_task_tokens),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(llm_hidden_dim * num_task_tokens, llm_hidden_dim * num_task_tokens)
        )
        
        # Tool projection: 1024D → LLM_dim (per tool)
        self.tool_proj = nn.Sequential(
            nn.Linear(tool_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        v_temporal: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None,
        tool_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Project context embeddings to LLM dimension.
        
        Args:
            v_temporal: Temporal embedding (batch, 256) or (256,)
            task_embedding: Task embedding (batch, seq_len, 896) or (seq_len, 896)
            tool_embeddings: Tool encodings (batch, num_tools, 1024) or (num_tools, 1024)
        
        Returns:
            Dictionary with projected embeddings:
            {
                'temporal_tokens': (batch, num_temporal_tokens, llm_hidden_dim),
                'task_tokens': (batch, num_task_tokens, llm_hidden_dim),
                'tool_tokens': (batch, num_tools, llm_hidden_dim)
            }
        """
        outputs = {}
        
        # Project temporal embedding
        if v_temporal is not None:
            if v_temporal.dim() == 1:
                v_temporal = v_temporal.unsqueeze(0)  # (1, 256)
            
            batch_size = v_temporal.size(0)
            temporal_proj = self.temporal_proj(v_temporal)  # (batch, llm_dim * num_tokens)
            temporal_tokens = temporal_proj.view(batch_size, self.num_temporal_tokens, self.llm_hidden_dim)
            outputs['temporal_tokens'] = temporal_tokens
        
        # Project task embedding (use mean pooling if sequence)
        if task_embedding is not None:
            if task_embedding.dim() == 2:
                task_embedding = task_embedding.unsqueeze(0)  # (1, seq_len, 896)
            
            batch_size = task_embedding.size(0)
            # Mean pool over sequence dimension
            task_pooled = task_embedding.mean(dim=1)  # (batch, 896)
            task_proj = self.task_proj(task_pooled)  # (batch, llm_dim * num_tokens)
            task_tokens = task_proj.view(batch_size, self.num_task_tokens, self.llm_hidden_dim)
            outputs['task_tokens'] = task_tokens
        
        # Project tool embeddings
        if tool_embeddings is not None:
            if tool_embeddings.dim() == 2:
                tool_embeddings = tool_embeddings.unsqueeze(0)  # (1, num_tools, 1024)
            
            batch_size, num_tools, _ = tool_embeddings.shape
            # Project each tool
            tool_tokens = self.tool_proj(tool_embeddings)  # (batch, num_tools, llm_dim)
            outputs['tool_tokens'] = tool_tokens
        
        return outputs
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ContextProjector':
        """Create ContextProjector from configuration."""
        llm_config = config.get('llm', {})
        model_config = llm_config.get('model', {})
        projector_config = llm_config.get('context_projector', {})
        
        # Get LLM hidden dimension from model name
        model_name = model_config.get('name', 'Qwen/Qwen2.5-7B-Instruct')
        if '0.5B' in model_name:
            llm_hidden_dim = 896
        elif '7B' in model_name:
            llm_hidden_dim = 3584
        else:
            llm_hidden_dim = projector_config.get('llm_hidden_dim', 3584)
        
        return cls(
            llm_hidden_dim=llm_hidden_dim,
            temporal_dim=projector_config.get('temporal_dim', 256),
            task_dim=projector_config.get('task_dim', 896),
            tool_dim=projector_config.get('tool_dim', 1024),
            num_temporal_tokens=projector_config.get('num_temporal_tokens', 1),
            num_task_tokens=projector_config.get('num_task_tokens', 1),
            dropout=projector_config.get('dropout', 0.1)
        )


class QwenBackbone(nn.Module):
    """
    Qwen2.5 LLM backbone with context injection.
    
    Supports:
    - Qwen2.5-7B-Instruct (production)
    - Qwen2.5-0.5B-Instruct (testing)
    - Context prefix tokens injection
    - Text generation with constraints
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cpu",
        dtype: str = "float32",
        use_flash_attn: bool = False,
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize Qwen backbone.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            dtype: Data type (float32, float16, bfloat16)
            use_flash_attn: Use flash attention (requires compatible GPU)
            trust_remote_code: Trust remote code from HuggingFace
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.dtype_str = dtype
        self.use_flash_attn = use_flash_attn
        
        # Determine torch dtype
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        logger.info(f"Loading Qwen model: {model_name}")
        logger.info(f"Device: {device}, dtype: {dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.dtype,
        }
        
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not quantized
        if not (load_in_8bit or load_in_4bit):
            self.model = self.model.to(device)
        
        # Get model config
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Hidden dim: {self.hidden_dim}, Vocab size: {self.vocab_size}")
    
    def prepare_inputs_with_context(
        self,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare LLM inputs with context prefix tokens.
        
        Args:
            text_tokens: Text token IDs (batch, seq_len)
            text_attention_mask: Attention mask (batch, seq_len)
            context_embeddings: Dict with 'temporal_tokens', 'task_tokens', 'tool_tokens'
        
        Returns:
            inputs_embeds: (batch, total_len, hidden_dim)
            attention_mask: (batch, total_len)
        """
        batch_size = text_tokens.size(0)
        
        # Get text embeddings
        text_embeds = self.model.get_input_embeddings()(text_tokens)  # (batch, seq_len, hidden_dim)
        
        # Prepare context tokens
        if context_embeddings is not None:
            context_parts = []
            context_mask_parts = []
            
            # Add temporal tokens
            if 'temporal_tokens' in context_embeddings:
                temporal = context_embeddings['temporal_tokens']  # (batch, num_temporal, hidden_dim)
                context_parts.append(temporal)
                context_mask_parts.append(torch.ones(
                    batch_size, temporal.size(1),
                    dtype=text_attention_mask.dtype,
                    device=text_attention_mask.device
                ))
            
            # Add task tokens
            if 'task_tokens' in context_embeddings:
                task = context_embeddings['task_tokens']  # (batch, num_task, hidden_dim)
                context_parts.append(task)
                context_mask_parts.append(torch.ones(
                    batch_size, task.size(1),
                    dtype=text_attention_mask.dtype,
                    device=text_attention_mask.device
                ))
            
            # Add tool tokens
            if 'tool_tokens' in context_embeddings:
                tools = context_embeddings['tool_tokens']  # (batch, num_tools, hidden_dim)
                context_parts.append(tools)
                context_mask_parts.append(torch.ones(
                    batch_size, tools.size(1),
                    dtype=text_attention_mask.dtype,
                    device=text_attention_mask.device
                ))
            
            # Concatenate: [context_tokens, text_tokens]
            if context_parts:
                context_embeds = torch.cat(context_parts, dim=1)  # (batch, total_context, hidden_dim)
                context_mask = torch.cat(context_mask_parts, dim=1)  # (batch, total_context)
                
                # If text batch size > context batch size, expand context to match
                if text_embeds.size(0) > context_embeds.size(0):
                    # Expand single context to all batch items
                    context_embeds = context_embeds.expand(text_embeds.size(0), -1, -1)
                    context_mask = context_mask.expand(text_embeds.size(0), -1)
                
                inputs_embeds = torch.cat([context_embeds, text_embeds], dim=1)
                attention_mask = torch.cat([context_mask, text_attention_mask], dim=1)
            else:
                inputs_embeds = text_embeds
                attention_mask = text_attention_mask
        else:
            inputs_embeds = text_embeds
            attention_mask = text_attention_mask
        
        return inputs_embeds, attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass through Qwen model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )
    
    def generate(
        self,
        input_text: Union[str, List[str]],
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text with context injection.
        
        Args:
            input_text: Input prompt(s)
            context_embeddings: Context embeddings to inject
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        # Tokenize input
        if isinstance(input_text, str):
            input_text = [input_text]
        
        encoded = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Prepare inputs with context
        inputs_embeds, attention_mask = self.prepare_inputs_with_context(
            input_ids,
            attention_mask,
            context_embeddings
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts
    
    @classmethod
    def from_config(cls, config: Dict) -> 'QwenBackbone':
        """Create QwenBackbone from configuration."""
        llm_config = config.get('llm', {})
        model_config = llm_config.get('model', {})
        
        return cls(
            model_name=model_config.get('name', 'Qwen/Qwen2.5-7B-Instruct'),
            device=model_config.get('device', 'cpu'),
            dtype=model_config.get('dtype', 'float32'),
            use_flash_attn=model_config.get('use_flash_attn', False),
            trust_remote_code=model_config.get('trust_remote_code', True),
            load_in_8bit=model_config.get('load_in_8bit', False),
            load_in_4bit=model_config.get('load_in_4bit', False)
        )
