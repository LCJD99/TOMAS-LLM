"""
Token Type Gate / Switch

Routes LLM outputs to appropriate decoding heads:
- Standard vocab tokens → LM head (explanatory text)
- Special tokens (out-of-vocab) → Resource allocation heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# Special token type: Any token outside standard vocabulary
# Use vocab_size as the base for special tokens
# Example: vocab_size = 151936 (Qwen2.5)
#   - 151936: TOOL_PLAN (single token encoding both tool_id and resource_cfg)
#   - 151937+: Reserved for future extensions
TOOL_PLAN_TOKEN_OFFSET = 0  # offset from vocab_size


class TokenTypeGate(nn.Module):
    """
    Token Type Gate / Switch
    
    Routes LLM decoder outputs to appropriate heads based on token type:
    
    Path A (Standard vocab): 
        token_id in [0, vocab_size) → LM head → explanatory text
        
    Path B (Special token - TOOL_PLAN):
        token_id >= vocab_size → Resource allocation heads
        Single special token encodes both:
        → ToolClassifier(hidden) → tool_id
        → ResourceRegressor(hidden, tool_emb, temporal) → resources
    
    Architecture:
        ┌──────────────────────────────────────────────────────┐
        │              LLM Hidden States                        │
        │                (batch, seq, hidden_dim)              │
        └────────────────────┬─────────────────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │    TokenTypeGate         │
                │  token >= vocab_size?    │
                └────────┬────────┬────────┘
                         │        │
              ┌──────────▼──┐  ┌─▼─────────────────────────┐
              │  Path A     │  │  Path B                   │
              │  (标准词表)  │  │  (TOOL_PLAN token)        │
              └──────┬──────┘  └───┬───────────────────────┘
                     │             │
              ┌──────▼──────┐  ┌───▼────────────────────────┐
              │  LM Head    │  │ Dual Heads (from 1 hidden) │
              │  (vocab)    │  │ - ToolClassifier           │
              │             │  │ - ResourceRegressor        │
              └─────────────┘  └────────────────────────────┘
    
    Key Design:
        - One special token (TOOL_PLAN) at position i
        - hidden_states[i] fed to BOTH heads simultaneously:
          * ToolClassifier extracts tool selection info
          * ResourceRegressor extracts resource allocation info
        - Like encoder: single representation, multiple decodings
    
    Usage:
        gate = TokenTypeGate(vocab_size=151936)
        
        # During generation
        next_token = sample_next_token(...)  # May be TOOL_PLAN (>= vocab_size)
        
        if gate.is_special_token_batch(next_token).any():
            # Extract TOOL_PLAN hidden states
            special_hidden = gate.extract_special_positions(...)
            
            # Decode both tool_id and resources from same hidden state
            tool_id = tool_classifier(special_hidden)
            resources = resource_regressor(special_hidden, tool_emb, temporal)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
    ):
        """
        Args:
            vocab_size: Size of standard vocabulary (e.g., 151936 for Qwen2.5)
            hidden_dim: LLM hidden dimension (896 for 0.5B, 3584 for 7B)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.tool_plan_token_id = vocab_size + TOOL_PLAN_TOKEN_OFFSET
        
        # Simple rule-based routing (no learned gate needed)
        self.use_learned_gate = False
    
    def is_special_token_batch(
        self, 
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Check which tokens in a batch are special tokens.
        
        Args:
            token_ids: (batch, seq) - Token IDs
        
        Returns:
            is_special: (batch, seq) - Boolean mask, True for special tokens
        
        Note:
            Any token_id >= vocab_size is considered special.
            This is general and doesn't require pre-defined token lists.
        """
        is_special = token_ids >= self.vocab_size
        return is_special
    
    def get_routing_mask(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get routing masks for different token types.
        
        Args:
            token_ids: (batch, seq) - Generated token IDs
            hidden_states: (batch, seq, hidden_dim) - LLM hidden states (optional)
        
        Returns:
            masks: Dict with routing information
                - 'standard_mask': (batch, seq) - Standard vocab tokens
                - 'special_mask': (batch, seq) - Special tokens (TOOL_PLAN)
        """
        standard_mask = token_ids < self.vocab_size
        special_mask = token_ids >= self.vocab_size
        
        masks = {
            'standard_mask': standard_mask,
            'special_mask': special_mask,
        }
        
        return masks
    
    def extract_special_positions(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract hidden states at special token (TOOL_PLAN) positions.
        
        Args:
            token_ids: (batch, seq) - Token IDs
            hidden_states: (batch, seq, hidden_dim) - Hidden states
        
        Returns:
            special_hidden: (num_special, hidden_dim) - Hidden states at TOOL_PLAN positions
            special_indices: (num_special, 2) - (batch_idx, seq_idx) for each TOOL_PLAN token
        
        Note:
            Each TOOL_PLAN token's hidden state will be fed to BOTH:
            - ToolClassifier (to extract tool_id)
            - ResourceRegressor (to extract resource allocation)
        """
        masks = self.get_routing_mask(token_ids, hidden_states)
        mask = masks['special_mask']
        
        # Get positions where mask is True
        special_indices = torch.nonzero(mask, as_tuple=False)  # (num_special, 2)
        
        if special_indices.size(0) == 0:
            # No special tokens found
            return torch.zeros(0, self.hidden_dim, device=hidden_states.device), special_indices
        
        # Extract hidden states at these positions
        batch_idx = special_indices[:, 0]
        seq_idx = special_indices[:, 1]
        special_hidden = hidden_states[batch_idx, seq_idx]  # (num_special, hidden_dim)
        
        return special_hidden, special_indices
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: route hidden states to appropriate heads.
        
        Args:
            hidden_states: (batch, seq, hidden_dim) - LLM hidden states
            token_ids: (batch, seq) - Token IDs (if available)
                If None, use learned gate classifier
        
        Returns:
            routing_info: Dict containing:
                - 'masks': Routing masks
                - 'standard_hidden': Hidden states for standard tokens
                - 'special_hidden': Hidden states for special tokens
                - 'special_indices': Positions of special tokens
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if token_ids is not None:
            # Rule-based routing
            masks = self.get_routing_mask(token_ids, hidden_states)
            special_hidden, special_indices = self.extract_special_positions(
                token_ids, hidden_states
            )
            
            # Extract standard token hidden states
            standard_mask = masks['standard_mask']
            standard_indices = torch.nonzero(standard_mask, as_tuple=False)
            if standard_indices.size(0) > 0:
                batch_idx = standard_indices[:, 0]
                seq_idx = standard_indices[:, 1]
                standard_hidden = hidden_states[batch_idx, seq_idx]
            else:
                standard_hidden = torch.zeros(0, hidden_dim, device=hidden_states.device)
            
        else:
            # Learned gate classifier
            if self.gate_classifier is None:
                raise ValueError("Learned gate not available. Provide token_ids for rule-based routing.")
            
            # Predict token type
            gate_logits = self.gate_classifier(hidden_states)  # (batch, seq, 2)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            is_special_pred = gate_probs[:, :, 1] > 0.5  # (batch, seq)
            
            # Create masks
            masks = {
                'standard_mask': ~is_special_pred,
                'special_mask': is_special_pred,
            }
            
            # Extract hidden states
            special_indices = torch.nonzero(is_special_pred, as_tuple=False)
            if special_indices.size(0) > 0:
                batch_idx = special_indices[:, 0]
                seq_idx = special_indices[:, 1]
                special_hidden = hidden_states[batch_idx, seq_idx]
            else:
                special_hidden = torch.zeros(0, hidden_dim, device=hidden_states.device)
            
            standard_indices = torch.nonzero(~is_special_pred, as_tuple=False)
            if standard_indices.size(0) > 0:
                batch_idx = standard_indices[:, 0]
                seq_idx = standard_indices[:, 1]
                standard_hidden = hidden_states[batch_idx, seq_idx]
            else:
                standard_hidden = torch.zeros(0, hidden_dim, device=hidden_states.device)
        
        routing_info = {
            'masks': masks,
            'standard_hidden': standard_hidden,
            'special_hidden': special_hidden,
            'special_indices': special_indices,
        }
        
        return routing_info
    
    def __repr__(self):
        return (
            f"TokenTypeGate(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  tool_plan_token_id={self.tool_plan_token_id},\n"
            f"  use_learned_gate={self.use_learned_gate}\n"
            f")"
        )
