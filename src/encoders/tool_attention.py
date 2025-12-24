"""
Multi-head Self-Attention for Tool Set Contextualization.

This module applies self-attention over the set of tool-aware embeddings to
capture inter-tool relationships and context. Each tool's representation is
enhanced by attending to all other tools in the set.

Input: v_toolaware (num_tools, d_toolaware) - individual tool embeddings
Output: h_toolset (num_tools, d_toolaware) - contextualized tool embeddings
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ToolSetAttention(nn.Module):
    """
    Multi-head self-attention layer for tool set contextualization.
    
    This applies self-attention over a set of tools, allowing each tool's
    representation to be influenced by all other tools in the set.
    
    Architecture:
        Input → MultiheadAttention → Residual → LayerNorm → Output
    
    The attention mechanism captures:
    - Tool similarity and relationships
    - Resource competition/complementarity
    - Contextual dependencies between tools
    """
    
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        """
        Initialize tool set attention layer.
        
        Args:
            d_model: Dimension of tool embeddings (d_toolaware)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            batch_first: If True, input shape is (batch, seq, feature)
        """
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout for residual connection
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized ToolSetAttention: d_model={d_model}, "
                   f"num_heads={num_heads}, head_dim={self.head_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention over tool set.
        
        Args:
            x: Tool embeddings, shape (batch, num_tools, d_model) or (num_tools, d_model)
            key_padding_mask: Mask for padding tools, shape (batch, num_tools)
            attn_mask: Attention mask, shape (num_tools, num_tools)
            return_attention: If True, return (output, attention_weights)
        
        Returns:
            If return_attention=False:
                Contextualized tool embeddings, same shape as input
            If return_attention=True:
                (output, attention_weights) tuple
        """
        # Handle unbatched input
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Self-attention (query = key = value = x)
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False  # Return per-head attention weights
        )
        
        # Residual connection + dropout
        x = x + self.dropout(attn_output)
        
        # Layer normalization
        x = self.norm(x)
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            x = x.squeeze(0)
            if return_attention and attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)
        
        if return_attention:
            return x, attn_weights
        return x
    
    def extra_repr(self) -> str:
        """String representation for print()."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, head_dim={self.head_dim}"


class ToolSetEncoder(nn.Module):
    """
    Multi-layer transformer encoder for tool set contextualization.
    
    Stacks multiple ToolSetAttention layers with optional feedforward networks
    to create deep contextualization of tool representations.
    
    Architecture:
        Input → [Attention → FFN]×N → Output
    
    Where FFN (FeedForward Network) is optional.
    """
    
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        dim_feedforward: Optional[int] = None,
        use_ffn: bool = False
    ):
        """
        Initialize multi-layer tool set encoder.
        
        Args:
            d_model: Dimension of tool embeddings
            num_heads: Number of attention heads per layer
            num_layers: Number of attention layers
            dropout: Dropout probability
            dim_feedforward: Feedforward network hidden dimension (default: 4*d_model)
            use_ffn: Whether to include feedforward networks after attention
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_ffn = use_ffn
        
        # Create attention layers
        self.layers = nn.ModuleList([
            ToolSetAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Optional feedforward networks
        if use_ffn:
            if dim_feedforward is None:
                dim_feedforward = 4 * d_model
            
            self.ffn_layers = nn.ModuleList([
                self._make_ffn(d_model, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.ffn_layers = None
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized ToolSetEncoder: {num_layers} layers, "
                   f"d_model={d_model}, num_heads={num_heads}, "
                   f"use_ffn={use_ffn}, params={param_count}")
    
    def _make_ffn(self, d_model: int, dim_feedforward: int, dropout: float) -> nn.Module:
        """Create a feedforward network block."""
        return nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Apply multi-layer self-attention encoding.
        
        Args:
            x: Tool embeddings, shape (batch, num_tools, d_model) or (num_tools, d_model)
            key_padding_mask: Padding mask, shape (batch, num_tools)
            attn_mask: Attention mask, shape (num_tools, num_tools)
            return_all_attentions: If True, return (output, [attn_weights_per_layer])
        
        Returns:
            If return_all_attentions=False:
                Contextualized tool embeddings, same shape as input
            If return_all_attentions=True:
                (output, attention_weights_list) tuple
        """
        all_attentions = [] if return_all_attentions else None
        
        for i, layer in enumerate(self.layers):
            # Self-attention
            if return_all_attentions:
                x, attn_weights = layer(x, key_padding_mask, attn_mask, return_attention=True)
                all_attentions.append(attn_weights)
            else:
                x = layer(x, key_padding_mask, attn_mask, return_attention=False)
            
            # Optional feedforward network
            if self.use_ffn:
                residual = x
                x = self.ffn_layers[i](x)
                x = residual + x  # Residual connection
        
        if return_all_attentions:
            return x, all_attentions
        return x
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ToolSetEncoder':
        """
        Create ToolSetEncoder from configuration dictionary.
        
        Args:
            config: Configuration dictionary with model.tool_attention section
        
        Returns:
            Initialized ToolSetEncoder instance
        """
        # Get d_model from concatenation dimensions
        d_tool = config['model']['tool_encoder']['d_tool']
        d_resource = config['model']['resource_mlp']['d_resource']
        d_model = d_tool + d_resource  # d_toolaware
        
        # Get attention config
        attn_config = config['model']['tool_attention']
        num_heads = attn_config.get('num_heads', 8)
        num_layers = attn_config.get('num_layers', 1)
        dropout = attn_config.get('dropout', 0.1)
        dim_feedforward = attn_config.get('dim_feedforward', None)
        use_ffn = attn_config.get('use_ffn', False)
        
        return cls(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            use_ffn=use_ffn
        )
    
    def get_output_dim(self) -> int:
        """Get output dimension (same as d_model)."""
        return self.d_model
    
    def extra_repr(self) -> str:
        """String representation for print()."""
        return (f"num_layers={self.num_layers}, d_model={self.d_model}, "
                f"num_heads={self.num_heads}, use_ffn={self.use_ffn}")


class CompleteToolEncoder(nn.Module):
    """
    Complete tool encoding pipeline: semantic + resource + attention.
    
    This is the full Left Panel implementation, combining:
    1. ToolEncoder: Semantic embeddings
    2. ResourceMLP: Resource profile embeddings
    3. ToolAwareEmbedding: Concatenation
    4. ToolSetEncoder: Self-attention contextualization
    
    Input: Tool names/texts + resource vectors
    Output: Contextualized tool set embeddings
    """
    
    def __init__(
        self,
        tool_encoder: nn.Module,
        resource_mlp: nn.Module,
        concatenator: nn.Module,
        attention_encoder: ToolSetEncoder
    ):
        """
        Initialize complete tool encoder.
        
        Args:
            tool_encoder: ToolEncoder instance
            resource_mlp: ResourceMLP instance
            concatenator: ToolAwareEmbedding instance
            attention_encoder: ToolSetEncoder instance
        """
        super().__init__()
        
        self.tool_encoder = tool_encoder
        self.resource_mlp = resource_mlp
        self.concatenator = concatenator
        self.attention_encoder = attention_encoder
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized CompleteToolEncoder with {param_count} parameters")
    
    def forward(
        self,
        tool_names: Optional[list] = None,
        tool_texts: Optional[list] = None,
        resource_vectors: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Encode tools with full pipeline.
        
        Args:
            tool_names: Tool names (for name-based encoder)
            tool_texts: Tool descriptions (for text-based encoder)
            resource_vectors: Resource feature vectors, shape (num_tools, 6)
            use_cache: Whether to use cached tool embeddings
            return_attention: If True, return attention weights
        
        Returns:
            If return_attention=False:
                Contextualized tool embeddings, shape (num_tools, d_model)
            If return_attention=True:
                (embeddings, attention_weights) tuple
        """
        if resource_vectors is None:
            raise ValueError("resource_vectors required")
        
        # 1. Encode tool semantics
        tool_embeddings = self.tool_encoder(
            tool_names=tool_names,
            tool_texts=tool_texts,
            use_cache=use_cache
        )
        
        # 2. Project resource profiles
        resource_embeddings = self.resource_mlp(resource_vectors)
        
        # 3. Concatenate to tool-aware embeddings
        toolaware_embeddings = self.concatenator(tool_embeddings, resource_embeddings)
        
        # 4. Self-attention contextualization
        if return_attention:
            h_toolset, attn_weights = self.attention_encoder(
                toolaware_embeddings,
                return_all_attentions=True
            )
            return h_toolset, attn_weights
        else:
            h_toolset = self.attention_encoder(toolaware_embeddings)
            return h_toolset
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        tool_names: Optional[list] = None,
        encoder_type: str = "name"
    ) -> 'CompleteToolEncoder':
        """
        Create CompleteToolEncoder from configuration.
        
        Args:
            config: Configuration dictionary
            tool_names: List of tool names (required for name encoder)
            encoder_type: Type of tool encoder ("name" or "text")
        
        Returns:
            Initialized CompleteToolEncoder instance
        """
        from .tool_encoder import ToolEncoder
        from .resource_mlp import ResourceMLP
        from .concatenation import ToolAwareEmbedding
        
        # Create components
        tool_encoder = ToolEncoder(config, tool_names=tool_names, encoder_type=encoder_type)
        resource_mlp = ResourceMLP.from_config(config)
        concatenator = ToolAwareEmbedding.from_config(config)
        attention_encoder = ToolSetEncoder.from_config(config)
        
        return cls(tool_encoder, resource_mlp, concatenator, attention_encoder)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.attention_encoder.get_output_dim()
