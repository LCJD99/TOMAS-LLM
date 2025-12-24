"""
Tool-Aware Embedding - Concatenation of semantic and resource embeddings.

This module combines tool semantic embeddings (Stream A) with resource profile
embeddings (Stream B) to create resource-aware tool embeddings.

v_toolaware = [v_tool || v_resource]
Dimension: d_tool + d_resource
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ToolAwareEmbedding(nn.Module):
    """
    Combines tool semantic embeddings with resource profile embeddings.
    
    This module takes two inputs:
    - Tool embeddings (from ToolEncoder): shape (batch_size, d_tool)
    - Resource embeddings (from ResourceMLP): shape (batch_size, d_resource)
    
    And concatenates them to produce:
    - Tool-aware embeddings: shape (batch_size, d_tool + d_resource)
    
    The dimensions d_tool and d_resource are configurable hyperparameters
    that should match the output dimensions of ToolEncoder and ResourceMLP.
    """
    
    def __init__(
        self,
        d_tool: int = 768,
        d_resource: int = 256,
        validate_dims: bool = True
    ):
        """
        Initialize tool-aware embedding module.
        
        Args:
            d_tool: Dimension of tool semantic embeddings (from ToolEncoder)
            d_resource: Dimension of resource profile embeddings (from ResourceMLP)
            validate_dims: Whether to validate input dimensions during forward pass
        """
        super().__init__()
        
        self.d_tool = d_tool
        self.d_resource = d_resource
        self.d_toolaware = d_tool + d_resource
        self.validate_dims = validate_dims
        
        logger.info(f"Initialized ToolAwareEmbedding: d_tool={d_tool}, "
                   f"d_resource={d_resource}, d_toolaware={self.d_toolaware}")
    
    def forward(
        self,
        tool_embeddings: torch.Tensor,
        resource_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate tool and resource embeddings.
        
        Args:
            tool_embeddings: Tool semantic embeddings, shape (batch_size, d_tool) or (d_tool,)
            resource_embeddings: Resource profile embeddings, shape (batch_size, d_resource) or (d_resource,)
        
        Returns:
            Concatenated embeddings, shape (batch_size, d_toolaware) or (d_toolaware,)
        
        Raises:
            ValueError: If input dimensions don't match expected dimensions
        """
        # Handle both batched and unbatched inputs
        tool_is_1d = tool_embeddings.dim() == 1
        resource_is_1d = resource_embeddings.dim() == 1
        
        # Ensure consistent batching
        if tool_is_1d != resource_is_1d:
            raise ValueError(
                f"Tool and resource embeddings must have same batch dimension. "
                f"Got tool.dim={tool_embeddings.dim()}, resource.dim={resource_embeddings.dim()}"
            )
        
        if tool_is_1d:
            tool_embeddings = tool_embeddings.unsqueeze(0)
            resource_embeddings = resource_embeddings.unsqueeze(0)
        
        # Validate dimensions if enabled
        if self.validate_dims:
            if tool_embeddings.size(-1) != self.d_tool:
                raise ValueError(
                    f"Expected tool embedding dimension {self.d_tool}, "
                    f"got {tool_embeddings.size(-1)}"
                )
            
            if resource_embeddings.size(-1) != self.d_resource:
                raise ValueError(
                    f"Expected resource embedding dimension {self.d_resource}, "
                    f"got {resource_embeddings.size(-1)}"
                )
            
            # Check batch sizes match
            if tool_embeddings.size(0) != resource_embeddings.size(0):
                raise ValueError(
                    f"Batch sizes must match. Got tool batch_size={tool_embeddings.size(0)}, "
                    f"resource batch_size={resource_embeddings.size(0)}"
                )
        
        # Concatenate along feature dimension
        toolaware_embeddings = torch.cat([tool_embeddings, resource_embeddings], dim=-1)
        
        # Remove batch dimension if inputs were 1D
        if tool_is_1d:
            toolaware_embeddings = toolaware_embeddings.squeeze(0)
        
        return toolaware_embeddings
    
    def split(self, toolaware_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split tool-aware embeddings back into tool and resource components.
        
        This is useful for analysis or visualization purposes.
        
        Args:
            toolaware_embeddings: Concatenated embeddings, shape (..., d_toolaware)
        
        Returns:
            Tuple of (tool_embeddings, resource_embeddings)
                - tool_embeddings: shape (..., d_tool)
                - resource_embeddings: shape (..., d_resource)
        """
        if toolaware_embeddings.size(-1) != self.d_toolaware:
            raise ValueError(
                f"Expected toolaware embedding dimension {self.d_toolaware}, "
                f"got {toolaware_embeddings.size(-1)}"
            )
        
        tool_embeddings = toolaware_embeddings[..., :self.d_tool]
        resource_embeddings = toolaware_embeddings[..., self.d_tool:]
        
        return tool_embeddings, resource_embeddings
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ToolAwareEmbedding':
        """
        Create ToolAwareEmbedding from configuration dictionary.
        
        Args:
            config: Configuration dictionary with encoder dimensions
        
        Returns:
            Initialized ToolAwareEmbedding instance
        """
        d_tool = config['model']['tool_encoder']['d_tool']
        d_resource = config['model']['resource_mlp']['d_resource']
        
        return cls(d_tool=d_tool, d_resource=d_resource)
    
    def get_output_dim(self) -> int:
        """Get output dimension (d_toolaware = d_tool + d_resource)."""
        return self.d_toolaware
    
    def extra_repr(self) -> str:
        """String representation for print()."""
        return f"d_tool={self.d_tool}, d_resource={self.d_resource}, d_toolaware={self.d_toolaware}"


class ResourceAwareToolEncoder(nn.Module):
    """
    End-to-end encoder combining ToolEncoder, ResourceMLP, and ToolAwareEmbedding.
    
    This is a convenience wrapper that combines all three components:
    1. ToolEncoder: Encodes tool names/descriptions → tool embeddings
    2. ResourceMLP: Projects resource features → resource embeddings
    3. ToolAwareEmbedding: Concatenates both → tool-aware embeddings
    
    Useful for training and inference pipelines.
    """
    
    def __init__(
        self,
        tool_encoder: nn.Module,
        resource_mlp: nn.Module,
        concatenator: Optional[ToolAwareEmbedding] = None
    ):
        """
        Initialize resource-aware tool encoder.
        
        Args:
            tool_encoder: ToolEncoder instance
            resource_mlp: ResourceMLP instance
            concatenator: Optional ToolAwareEmbedding instance (auto-created if None)
        """
        super().__init__()
        
        self.tool_encoder = tool_encoder
        self.resource_mlp = resource_mlp
        
        if concatenator is None:
            # Auto-create concatenator with dimensions from encoders
            d_tool = tool_encoder.d_tool
            d_resource = resource_mlp.d_resource
            concatenator = ToolAwareEmbedding(d_tool, d_resource)
        
        self.concatenator = concatenator
        
        logger.info(f"Initialized ResourceAwareToolEncoder with "
                   f"d_toolaware={self.concatenator.d_toolaware}")
    
    def forward(
        self,
        tool_names: Optional[list] = None,
        tool_texts: Optional[list] = None,
        resource_vectors: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Encode tools with resource awareness.
        
        Args:
            tool_names: Tool names (for name-based encoder or cache keys)
            tool_texts: Tool text descriptions (for text-based encoder)
            resource_vectors: Resource feature vectors, shape (batch_size, 6)
            use_cache: Whether to use cached tool embeddings
        
        Returns:
            Tool-aware embeddings, shape (batch_size, d_toolaware)
        """
        if resource_vectors is None:
            raise ValueError("resource_vectors required")
        
        # Encode tools
        tool_embeddings = self.tool_encoder(
            tool_names=tool_names,
            tool_texts=tool_texts,
            use_cache=use_cache
        )
        
        # Project resources
        resource_embeddings = self.resource_mlp(resource_vectors)
        
        # Concatenate
        toolaware_embeddings = self.concatenator(tool_embeddings, resource_embeddings)
        
        return toolaware_embeddings
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.concatenator.get_output_dim()
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        tool_names: Optional[list] = None,
        encoder_type: str = "name"
    ) -> 'ResourceAwareToolEncoder':
        """
        Create ResourceAwareToolEncoder from configuration.
        
        Args:
            config: Configuration dictionary
            tool_names: List of tool names (required for name encoder)
            encoder_type: Type of tool encoder ("name" or "text")
        
        Returns:
            Initialized ResourceAwareToolEncoder instance
        """
        from .tool_encoder import ToolEncoder
        from .resource_mlp import ResourceMLP
        
        # Create components
        tool_encoder = ToolEncoder(config, tool_names=tool_names, encoder_type=encoder_type)
        resource_mlp = ResourceMLP.from_config(config)
        concatenator = ToolAwareEmbedding.from_config(config)
        
        return cls(tool_encoder, resource_mlp, concatenator)
