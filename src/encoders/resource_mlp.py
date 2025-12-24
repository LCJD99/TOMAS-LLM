"""
Resource MLP (Stream B) - Resource profile projection.

This module projects low-dimensional numerical resource features (from profiling data)
into a high-dimensional latent space for fusion with semantic tool embeddings.

The input features are expected to be normalized (z-score) and consist of:
- input_size_encoded (ordinal: 0/1/2 for small/medium/large)
- cpu_core
- cpu_mem_gb
- gpu_sm (percentage 0-100)
- gpu_mem_gb
- latency_ms

Architecture: Linear -> ReLU -> Linear
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResourceMLP(nn.Module):
    """
    MLP for projecting resource profiling features to high-dimensional latent space.
    
    Takes normalized 6D resource vectors and projects them to d_resource dimensions
    via a two-layer MLP with ReLU activation.
    
    Architecture:
        Input (6D) → Linear(6, hidden_dim) → ReLU → Linear(hidden_dim, d_resource) → Output (d_resource)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 512,
        d_resource: int = 256,
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        """
        Initialize Resource MLP.
        
        Args:
            input_dim: Input feature dimension (default: 6)
            hidden_dim: Hidden layer dimension (default: 512)
            d_resource: Output dimension for resource embeddings (default: 256)
            dropout: Dropout probability (default: 0.0, no dropout)
            use_batch_norm: Whether to use batch normalization (default: False)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_resource = d_resource
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # First layer: input_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Optional batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        
        # Activation
        self.activation = nn.ReLU()
        
        # Optional dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Second layer: hidden_dim -> d_resource
        self.fc2 = nn.Linear(hidden_dim, d_resource)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized ResourceMLP: {input_dim} -> {hidden_dim} -> {d_resource}, "
                   f"dropout={dropout}, batch_norm={use_batch_norm}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
               Expected to be normalized resource features
        
        Returns:
            Output tensor of shape (batch_size, d_resource) or (d_resource,)
        """
        # Handle both batched and unbatched inputs
        input_is_1d = x.dim() == 1
        if input_is_1d:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Validate input dimension
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(-1)}"
            )
        
        # First layer
        out = self.fc1(x)
        
        # Batch normalization (if enabled)
        if self.bn1 is not None:
            out = self.bn1(out)
        
        # Activation
        out = self.activation(out)
        
        # Dropout (if enabled)
        if self.dropout_layer is not None:
            out = self.dropout_layer(out)
        
        # Second layer
        out = self.fc2(out)
        
        # Remove batch dimension if input was 1D
        if input_is_1d:
            out = out.squeeze(0)
        
        return out
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ResourceMLP':
        """
        Create ResourceMLP from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'model.resource_mlp' section
        
        Returns:
            Initialized ResourceMLP instance
        """
        mlp_config = config['model']['resource_mlp']
        
        return cls(
            input_dim=mlp_config.get('input_features', 6),
            hidden_dim=mlp_config.get('hidden_dim', 512),
            d_resource=mlp_config.get('d_resource', 256),
            dropout=mlp_config.get('dropout', 0.0),
            use_batch_norm=mlp_config.get('use_batch_norm', False)
        )


class ResourceNormalizer:
    """
    Utility class for normalizing and denormalizing resource features.
    
    Stores normalization statistics (mean, std) computed from training data
    and applies them during inference to ensure consistent preprocessing.
    """
    
    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        feature_names: Optional[list] = None
    ):
        """
        Initialize resource normalizer.
        
        Args:
            mean: Mean values for each feature (shape: input_dim)
            std: Standard deviation for each feature (shape: input_dim)
            feature_names: Names of features (for logging/debugging)
        """
        self.mean = mean
        self.std = std
        self.feature_names = feature_names or []
        
        if mean is not None and std is not None:
            logger.info(f"Initialized ResourceNormalizer with {len(mean)} features")
    
    def fit(self, features: torch.Tensor, feature_names: Optional[list] = None):
        """
        Compute normalization statistics from data.
        
        Args:
            features: Feature tensor of shape (N, input_dim)
            feature_names: Optional feature names
        """
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0)
        
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
        if feature_names:
            self.feature_names = feature_names
        
        logger.info(f"Fitted normalizer on {features.size(0)} samples")
        logger.info(f"  Mean: {self.mean.tolist()}")
        logger.info(f"  Std: {self.std.tolist()}")
    
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization.
        
        Args:
            features: Feature tensor (any shape with last dim = input_dim)
        
        Returns:
            Normalized features
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # Move stats to same device as input
        mean = self.mean.to(features.device)
        std = self.std.to(features.device)
        
        return (features - mean) / std
    
    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reverse z-score normalization.
        
        Args:
            features: Normalized feature tensor
        
        Returns:
            Original-scale features
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # Move stats to same device as input
        mean = self.mean.to(features.device)
        std = self.std.to(features.device)
        
        return features * std + mean
    
    def state_dict(self) -> Dict:
        """Return state dictionary for saving."""
        return {
            'mean': self.mean,
            'std': self.std,
            'feature_names': self.feature_names
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from dictionary."""
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.feature_names = state_dict.get('feature_names', [])
        logger.info(f"Loaded normalizer state with {len(self.mean)} features")
