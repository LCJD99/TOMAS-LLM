"""
Resource Regressor Head

MLP regressor for resource allocation (CPU, GPU, memory).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import numpy as np


class ResourceRegressor(nn.Module):
    """
    Resource Regressor Head - Single-Input Decoder
    
    Decodes resource allocation from LLM hidden states.
    
    Design Philosophy:
        Through fine-tuning, the LLM's TOOL_PLAN token hidden state already
        encodes all semantic information about resource requirements:
        - Task complexity → CPU/GPU needs
        - Tool selection → Resource profile
        - Temporal context → Available resources
        
        This module simply **decodes** the hidden state into explicit values,
        similar to how ToolClassifier decodes tool_id.
    
    Outputs:
        - cpu_core: Number of CPU cores (float, will be rounded)
        - cpu_mem_gb: CPU memory in GB
        - gpu_sm: GPU streaming multiprocessors (0 if no GPU)
        - gpu_mem_gb: GPU memory in GB (0 if no GPU)
    
    Architecture:
        ┌─────────────────────────────────────────┐
        │  Input:                                  │
        │  - hidden_states: (batch, hidden_dim)   │
        │    (from TOOL_PLAN token position)      │
        │    Already contains ALL semantic info   │
        └──────────────────┬──────────────────────┘
                           │
                ┌──────────▼──────────┐
                │  MLP Decoder        │
                │  hidden_dim →       │
                │  hidden_dim//2 →    │
                │  hidden_dim//4 →    │
                │  4 (resources)      │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Activation         │
                │  - Sigmoid → [0,1]  │
                │  (normalized)       │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Denormalize        │
                │  [0,1] → actual     │
                │  values             │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Clamp & Round      │
                │  - To available     │
                │  - Round integers   │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Output:            │
                │  (cpu_core,         │
                │   cpu_mem,          │
                │   gpu_sm,           │
                │   gpu_mem)          │
                └─────────────────────┘
    
    Training:
        - Loss: MSE or Huber loss on normalized values [0,1]
        - Constraint loss: Penalty for exceeding available resources
    
    Inference:
        - Decode hidden → normalized [0,1]
        - Denormalize to actual values
        - Clamp to available resources
        - Round cpu_core and gpu_sm to integers
    
    Usage:
        regressor = ResourceRegressor(
            hidden_dim=3584  # Qwen2.5-7B
        )
        
        # During inference at TOOL_PLAN position
        resource_params = regressor(
            hidden_states       # (batch, 3584) - Contains ALL semantic info!
        )
        # Returns: (batch, 4) normalized values [0,1]
        
        # Or get denormalized predictions:
        allocation = regressor.predict(
            hidden_states,
            available_resources  # Optional clamping
        )
        # Returns dict with actual values
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_constraint: bool = True,
    ):
        """
        Args:
            hidden_dim: LLM hidden dimension (896 for 0.5B, 3584 for 7B)
            dropout: Dropout probability
            use_constraint: Whether to enforce resource constraints
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_constraint = use_constraint
        
        # MLP decoder: hidden_dim → 4 normalized resources
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 4),  # [cpu_core, cpu_mem, gpu_sm, gpu_mem]
            nn.Sigmoid(),  # Output normalized values [0, 1]
        )
        
        # Resource limits (default maximum, can be overridden)
        self.register_buffer('default_max_resources', torch.tensor([
            64.0,   # max_cpu_core
            256.0,  # max_cpu_mem_gb
            128.0,  # max_gpu_sm
            80.0,   # max_gpu_mem_gb
        ]))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass - Decode hidden states to normalized resources.
        
        Args:
            hidden_states: (batch, hidden_dim) - LLM hidden at TOOL_PLAN position
                Contains all semantic information from fine-tuning:
                - Task requirements
                - Tool selection
                - Temporal context (from prefix tokens)
        
        Returns:
            normalized_resources: (batch, 4) - Normalized values in [0, 1]
                [cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb]
        """
        # Decode to normalized resources [0, 1]
        normalized_resources = self.regressor(hidden_states)  # (batch, 4)
        
        return normalized_resources
    
    def predict(
        self,
        hidden_states: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict resource allocation with denormalization.
        
        Args:
            hidden_states: (batch, hidden_dim) - LLM hidden states
            available_resources: (4,) or (batch, 4) - Max resources for clamping [optional]
        
        Returns:
            allocation: Dict containing:
                - 'cpu_core': (batch,) - Number of CPU cores (rounded)
                - 'cpu_mem_gb': (batch,) - CPU memory in GB
                - 'gpu_sm': (batch,) - GPU SMs (rounded, 0 if no GPU)
                - 'gpu_mem_gb': (batch,) - GPU memory in GB
                - 'normalized': (batch, 4) - Normalized values [0,1]
        """
        # Get normalized predictions [0, 1]
        normalized = self.forward(hidden_states)  # (batch, 4)
        
        # Denormalize to actual values
        resources = normalized * self.default_max_resources.unsqueeze(0)  # Scale to max
        
        # Apply constraints if available
        if available_resources is not None:
            batch_size = resources.size(0)
            if available_resources.dim() == 1:
                available_resources = available_resources.unsqueeze(0).expand(batch_size, -1)
            resources = torch.min(resources, available_resources)
        
        cpu_core = resources[:, 0]
        cpu_mem_gb = resources[:, 1]
        gpu_sm = resources[:, 2]
        gpu_mem_gb = resources[:, 3]
        
        allocation = {
            'cpu_core': cpu_core.round(),  # Integer cores
            'cpu_mem_gb': cpu_mem_gb,
            'gpu_sm': gpu_sm.round(),      # Integer SMs
            'gpu_mem_gb': gpu_mem_gb,
            'normalized': normalized,      # Keep normalized for loss computation
        }
        
        return allocation
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_resources: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
        loss_type: str = 'huber',
        constraint_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regression loss on normalized values.
        
        Args:
            hidden_states: (batch, hidden_dim)
            target_resources: (batch, 4) - Ground truth resources (actual values)
            available_resources: (batch, 4) - Available resources [optional]
            loss_type: 'mse', 'huber', or 'smooth_l1'
            constraint_weight: Weight for constraint violation penalty
        
        Returns:
            losses: Dict containing:
                - 'total': Total loss
                - 'regression': Regression loss (MSE/Huber) on normalized values
                - 'constraint': Constraint violation penalty [if use_constraint=True]
        """
        # Predict normalized resources [0, 1]
        predicted_normalized = self.forward(hidden_states)
        
        # Normalize targets to [0, 1]
        target_normalized = target_resources / self.default_max_resources.unsqueeze(0)
        
        # Regression loss on normalized values
        if loss_type == 'mse':
            regression_loss = nn.functional.mse_loss(predicted_normalized, target_normalized)
        elif loss_type == 'huber':
            regression_loss = nn.functional.huber_loss(predicted_normalized, target_normalized)
        elif loss_type == 'smooth_l1':
            regression_loss = nn.functional.smooth_l1_loss(predicted_normalized, target_normalized)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Constraint loss (penalty for exceeding available resources)
        constraint_loss = torch.tensor(0.0, device=predicted_normalized.device)
        if self.use_constraint and available_resources is not None:
            # Denormalize predictions for constraint check
            predicted_actual = predicted_normalized * self.default_max_resources.unsqueeze(0)
            
            batch_size = predicted_actual.size(0)
            if available_resources.dim() == 1:
                available_resources = available_resources.unsqueeze(0).expand(batch_size, -1)
            
            # Penalty for exceeding limits
            excess = torch.relu(predicted_actual - available_resources)  # (batch, 4)
            constraint_loss = excess.pow(2).sum(dim=-1).mean()  # Quadratic penalty
        
        # Total loss
        total_loss = regression_loss + constraint_weight * constraint_loss
        
        losses = {
            'total': total_loss,
            'regression': regression_loss,
            'constraint': constraint_loss,
        }
        
        return losses
    
    def __repr__(self):
        return (
            f"ResourceRegressor(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim=4 (normalized),\n"
            f"  use_constraint={self.use_constraint}\n"
            f")"
        )


class ResourceRegressorWithNormalization(nn.Module):
    """
    Resource Regressor with input/output normalization.
    
    Normalizes resource values to [0, 1] for better training stability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        resource_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        dropout: float = 0.1,
        use_constraint: bool = True,
    ):
        """
        Args:
            hidden_dim: LLM hidden dimension
            resource_stats: Dict with (mean, std) for each resource type
                Example: {
                    'cpu_core': (8.0, 4.0),
                    'cpu_mem_gb': (32.0, 16.0),
                    'gpu_sm': (20.0, 10.0),
                    'gpu_mem_gb': (8.0, 4.0)
                }
            dropout: Dropout probability
            use_constraint: Whether to enforce constraints
        """
        super().__init__()
        
        # Base regressor
        self.regressor = ResourceRegressor(
            hidden_dim, dropout, use_constraint
        )
        
        # Normalization statistics
        if resource_stats is not None:
            self.use_normalization = True
            
            # Extract means and stds
            means = torch.tensor([
                resource_stats['cpu_core'][0],
                resource_stats['cpu_mem_gb'][0],
                resource_stats['gpu_sm'][0],
                resource_stats['gpu_mem_gb'][0],
            ])
            stds = torch.tensor([
                resource_stats['cpu_core'][1],
                resource_stats['cpu_mem_gb'][1],
                resource_stats['gpu_sm'][1],
                resource_stats['gpu_mem_gb'][1],
            ])
            
            self.register_buffer('resource_means', means)
            self.register_buffer('resource_stds', stds)
        else:
            self.use_normalization = False
    
    def normalize(self, resources: torch.Tensor) -> torch.Tensor:
        """Normalize resources to standard scale."""
        if self.use_normalization:
            return (resources - self.resource_means) / self.resource_stds
        return resources
    
    def denormalize(self, normalized_resources: torch.Tensor) -> torch.Tensor:
        """Denormalize resources to original scale."""
        if self.use_normalization:
            return normalized_resources * self.resource_stds + self.resource_means
        return normalized_resources
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward - returns normalized resources from base regressor."""
        return self.regressor(hidden_states)
    
    def predict(self, *args, **kwargs):
        """Predict with denormalization."""
        return self.regressor.predict(*args, **kwargs)
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_resources: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Compute loss with custom normalization."""
        # Normalize targets using custom stats
        normalized_targets = self.normalize(target_resources)
        normalized_available = None if available_resources is None else self.normalize(available_resources)
        
        # Compute loss in normalized space
        return self.regressor.compute_loss(
            hidden_states,
            normalized_targets,
            normalized_available,
            **kwargs
        )
