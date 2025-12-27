"""
Temporal LLM Wrapper for Modality Alignment Pretraining.

Integrates TemporalEncoder (trainable) with frozen LLM (Qwen2.5).
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from .temporal_encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class TemporalLLMWrapper(nn.Module):
    """
    Wrapper that combines TemporalEncoder with frozen LLM.
    
    Architecture:
    - TemporalEncoder (trainable): CNN + Projector
    - LLM (frozen): Qwen2.5 for text generation
    
    Training objective: Causal Language Modeling Loss
    """
    
    def __init__(
        self,
        temporal_encoder: TemporalEncoder,
        llm_model: PreTrainedModel,
        llm_embedding_dim: int = 3584,
        freeze_llm: bool = True
    ):
        """
        Initialize wrapper.
        
        Args:
            temporal_encoder: Trainable TemporalEncoder instance
            llm_model: Pretrained LLM (e.g., Qwen2.5)
            llm_embedding_dim: LLM embedding dimension
            freeze_llm: Whether to freeze LLM parameters
        """
        super().__init__()
        
        self.temporal_encoder = temporal_encoder
        self.llm_model = llm_model
        self.llm_embedding_dim = llm_embedding_dim
        
        # Freeze LLM if requested
        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model.eval()
            logger.info("LLM parameters frozen")
        
        # Verify projector exists and matches dimension
        if temporal_encoder.projector is None:
            raise ValueError("TemporalEncoder must have projector for LLM alignment")
        
        if temporal_encoder.llm_embedding_dim != llm_embedding_dim:
            logger.warning(
                f"TemporalEncoder embedding dim ({temporal_encoder.llm_embedding_dim}) "
                f"!= LLM embedding dim ({llm_embedding_dim})"
            )
    
    def _extend_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extend attention mask to include temporal token.
        
        Args:
            attention_mask: Original mask of shape (batch, seq_len)
        
        Returns:
            Extended mask of shape (batch, 1 + seq_len)
        """
        batch_size = attention_mask.size(0)
        # Add attention for temporal token (always 1)
        temporal_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        return torch.cat([temporal_mask, attention_mask], dim=1)
    
    def _prepare_labels(
        self,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int
    ) -> torch.Tensor:
        """
        Prepare labels for loss computation.
        
        Only compute loss on target tokens, mask prompt and temporal token.
        
        Args:
            prompt_ids: Prompt token IDs (batch, prompt_len)
            target_ids: Target token IDs (batch, target_len)
            prompt_len: Length including temporal token (1 + original_prompt_len)
        
        Returns:
            Labels tensor (batch, total_len) with -100 for masked positions
        """
        batch_size = prompt_ids.size(0)
        target_len = target_ids.size(0) if target_ids.dim() == 1 else target_ids.size(1)
        
        # Total sequence: [temporal_token] + prompt + target
        total_len = prompt_len + target_len
        
        # Initialize all to -100 (ignore)
        labels = torch.full(
            (batch_size, total_len),
            -100,
            dtype=torch.long,
            device=target_ids.device
        )
        
        # Only compute loss on target part
        labels[:, prompt_len:prompt_len + target_len] = target_ids
        
        return labels
    
    def forward(
        self,
        curve: torch.Tensor,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            curve: Resource curves (batch, num_timesteps, 4)
            prompt_ids: Prompt token IDs (batch, prompt_len)
            target_ids: Target token IDs (batch, target_len)
            attention_mask: Attention mask for prompt+target (batch, seq_len)
            return_outputs: Whether to return full outputs (for evaluation)
        
        Returns:
            loss: Causal LM loss (scalar)
            or outputs: Full model outputs if return_outputs=True
        """
        batch_size = curve.size(0)
        
        # 1. Encode temporal curves
        v_temporal = self.temporal_encoder.forward_batch(curve)  # (batch, llm_embedding_dim)
        
        # 2. Get LLM embeddings for prompt
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)  # (batch, prompt_len, llm_dim)
        
        # 3. Prepend temporal embedding as first token
        combined_embeds = torch.cat([
            v_temporal.unsqueeze(1),  # (batch, 1, llm_dim)
            prompt_embeds             # (batch, prompt_len, llm_dim)
        ], dim=1)  # (batch, 1 + prompt_len, llm_dim)
        
        # 4. Get target embeddings (for teacher forcing)
        target_embeds = self.llm_model.get_input_embeddings()(target_ids)  # (batch, target_len, llm_dim)
        
        # 5. Combine all embeddings
        full_embeds = torch.cat([
            combined_embeds,  # temporal + prompt
            target_embeds     # target
        ], dim=1)  # (batch, 1 + prompt_len + target_len, llm_dim)
        
        # 6. Extend attention mask
        extended_mask = self._extend_attention_mask(attention_mask)  # (batch, 1 + seq_len)
        
        # 7. Prepare labels
        prompt_len = 1 + prompt_ids.size(1)  # Including temporal token
        labels = self._prepare_labels(prompt_ids, target_ids, prompt_len)
        
        # 8. Forward through LLM
        outputs = self.llm_model(
            inputs_embeds=full_embeds,
            attention_mask=extended_mask,
            labels=labels,
            return_dict=True
        )
        
        if return_outputs:
            return outputs
        else:
            return outputs.loss
    
    def generate(
        self,
        curve: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_length: int = 256,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from temporal curve and prompt.
        
        Args:
            curve: Resource curve (1, num_timesteps, 4) or (num_timesteps, 4)
            prompt_ids: Prompt token IDs (1, prompt_len) or (prompt_len,)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs (1, generated_len)
        """
        # Ensure batch dimension
        if curve.dim() == 2:
            curve = curve.unsqueeze(0)  # (1, num_timesteps, 4)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # (1, prompt_len)
        
        batch_size = curve.size(0)
        
        # 1. Encode temporal curve
        v_temporal = self.temporal_encoder.forward_batch(curve)  # (1, llm_embedding_dim)
        
        # 2. Get prompt embeddings
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)  # (1, prompt_len, llm_dim)
        
        # 3. Combine temporal + prompt
        combined_embeds = torch.cat([
            v_temporal.unsqueeze(1),  # (1, 1, llm_dim)
            prompt_embeds             # (1, prompt_len, llm_dim)
        ], dim=1)  # (1, 1 + prompt_len, llm_dim)
        
        # 4. Create attention mask
        attention_mask = torch.ones(batch_size, combined_embeds.size(1), device=curve.device)
        
        # 5. Generate
        # Note: Some models don't support inputs_embeds for generation
        # In that case, we need to use a different approach
        try:
            outputs = self.llm_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.llm_model.config.eos_token_id,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Direct generation from embeddings failed: {e}")
            logger.warning("Falling back to forward-only mode (no generation)")
            raise NotImplementedError(
                "This LLM does not support generation from embeddings. "
                "Use forward() for training only."
            )
        
        return outputs
    
    def train(self, mode: bool = True):
        """
        Set training mode.
        
        Only affects TemporalEncoder; LLM remains frozen.
        """
        super().train(mode)
        # Keep LLM in eval mode
        self.llm_model.eval()
        return self
    
    def get_trainable_parameters(self):
        """Get parameters that require gradients."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
