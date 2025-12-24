"""
1D-CNN Temporal Encoder for System Resource Timeline.

This module:
1. Extracts timeline data from T_inf onwards
2. Normalizes resource values
3. Applies 1D-CNN to extract temporal features
4. Outputs v_temporal embedding for LLM injection

Input: Timeline data [time_ms, cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]
Output: v_temporal - temporal resource embedding
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timeline import SystemTimeline

logger = logging.getLogger(__name__)


class ResourceNormalizer:
    """
    Normalizes resource values for neural network input.
    
    Supports multiple normalization strategies:
    - minmax: Scale to [0, 1]
    - standard: Zero mean, unit variance
    - none: No normalization
    """
    
    def __init__(
        self,
        method: str = "minmax",
        resource_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ("minmax", "standard", "none")
            resource_ranges: Expected ranges for each resource type
                {
                    'cpu_cores': (min, max),
                    'cpu_mem_gb': (min, max),
                    'gpu_sm': (min, max),
                    'gpu_mem_gb': (min, max)
                }
        """
        self.method = method
        
        # Default resource ranges
        if resource_ranges is None:
            resource_ranges = {
                'cpu_cores': (0.0, 32.0),      # 0-32 cores
                'cpu_mem_gb': (0.0, 128.0),    # 0-128 GB
                'gpu_sm': (0.0, 100.0),         # 0-100%
                'gpu_mem_gb': (0.0, 80.0)      # 0-80 GB
            }
        self.resource_ranges = resource_ranges
        
        # Statistics for standard normalization
        self.mean = None
        self.std = None
    
    def fit(self, data: torch.Tensor):
        """
        Fit normalizer to data (for standard normalization).
        
        Args:
            data: Tensor of shape (num_timesteps, num_features)
        """
        if self.method == "standard":
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True) + 1e-8
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data.
        
        Args:
            data: Tensor of shape (num_timesteps, 4) or (batch, num_timesteps, 4)
                  Order: [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]
        
        Returns:
            Normalized tensor with same shape
        """
        if self.method == "none":
            return data
        
        elif self.method == "minmax":
            # MinMax normalization to [0, 1]
            ranges = torch.tensor([
                self.resource_ranges['cpu_cores'],
                self.resource_ranges['cpu_mem_gb'],
                self.resource_ranges['gpu_sm'],
                self.resource_ranges['gpu_mem_gb']
            ], dtype=data.dtype, device=data.device)
            
            min_vals = ranges[:, 0]  # (4,)
            max_vals = ranges[:, 1]  # (4,)
            
            # Handle different input shapes
            if data.dim() == 2:  # (T, 4)
                min_vals = min_vals.unsqueeze(0)  # (1, 4)
                max_vals = max_vals.unsqueeze(0)
            elif data.dim() == 3:  # (B, T, 4)
                min_vals = min_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
                max_vals = max_vals.unsqueeze(0).unsqueeze(0)
            
            normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)
            return torch.clamp(normalized, 0.0, 1.0)
        
        elif self.method == "standard":
            # Z-score normalization
            if self.mean is None:
                # Fit on current data
                self.fit(data if data.dim() == 2 else data.view(-1, data.size(-1)))
            
            return (data - self.mean) / self.std
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        Args:
            data: Normalized tensor
        
        Returns:
            Denormalized tensor
        """
        if self.method == "none":
            return data
        
        elif self.method == "minmax":
            ranges = torch.tensor([
                self.resource_ranges['cpu_cores'],
                self.resource_ranges['cpu_mem_gb'],
                self.resource_ranges['gpu_sm'],
                self.resource_ranges['gpu_mem_gb']
            ], dtype=data.dtype, device=data.device)
            
            min_vals = ranges[:, 0]
            max_vals = ranges[:, 1]
            
            if data.dim() == 2:
                min_vals = min_vals.unsqueeze(0)
                max_vals = max_vals.unsqueeze(0)
            elif data.dim() == 3:
                min_vals = min_vals.unsqueeze(0).unsqueeze(0)
                max_vals = max_vals.unsqueeze(0).unsqueeze(0)
            
            return data * (max_vals - min_vals) + min_vals
        
        elif self.method == "standard":
            return data * self.std + self.mean
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class TemporalCNN(nn.Module):
    """
    1D-CNN for temporal feature extraction from resource timeline.
    
    Architecture:
        Input: (batch, channels=4, time_steps)
        Conv1D layers with different kernel sizes to capture:
        - Short-term fluctuations (kernel=3)
        - Medium-term trends (kernel=5)
        - Long-term patterns (kernel=7)
        Output: v_temporal embedding
    """
    
    def __init__(
        self,
        in_channels: int = 4,          # cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb
        hidden_channels: int = 64,     # Hidden dimension
        output_dim: int = 256,         # v_temporal dimension
        num_layers: int = 3,           # Number of conv layers
        kernel_sizes: List[int] = None,  # Kernel sizes for each layer
        pooling: str = "adaptive_avg"  # Pooling method
    ):
        """
        Initialize Temporal CNN.
        
        Args:
            in_channels: Number of input channels (resource types)
            hidden_channels: Hidden channel dimension
            output_dim: Output embedding dimension
            num_layers: Number of convolutional layers
            kernel_sizes: List of kernel sizes (default: [3, 5, 7])
            pooling: Pooling method ("adaptive_avg", "max", "avg", "flatten")
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7][:num_layers]
        self.kernel_sizes = kernel_sizes
        
        # Build conv layers
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            layers.append(nn.Conv1d(
                current_channels,
                hidden_channels,
                kernel_size=kernel_sizes[i],
                padding=kernel_sizes[i] // 2,  # Same padding
                bias=True
            ))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_channels))
            current_channels = hidden_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Pooling layer
        if pooling == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
            fc_input_dim = hidden_channels
        elif pooling == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool1d(1)
            fc_input_dim = hidden_channels
        elif pooling == "flatten":
            self.pool = None  # Will be handled in forward
            fc_input_dim = None  # Will be computed dynamically
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Output projection
        if pooling != "flatten":
            self.fc = nn.Linear(fc_input_dim, output_dim)
        else:
            # Will be created in forward based on actual sequence length
            self.fc = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels=4, time_steps)
        
        Returns:
            v_temporal: Tensor of shape (batch, output_dim)
        """
        # Convolutional layers
        x = self.conv_layers(x)  # (batch, hidden_channels, time_steps)
        
        # Pooling
        if self.pooling in ["adaptive_avg", "adaptive_max"]:
            x = self.pool(x)  # (batch, hidden_channels, 1)
            x = x.squeeze(-1)  # (batch, hidden_channels)
        elif self.pooling == "flatten":
            x = x.flatten(1)  # (batch, hidden_channels * time_steps)
            
            # Create FC layer if not exists
            if self.fc is None:
                self.fc = nn.Linear(x.size(1), self.output_dim).to(x.device)
        
        # Output projection
        x = self.fc(x)  # (batch, output_dim)
        
        return x


class TemporalEncoder(nn.Module):
    """
    Complete temporal encoding pipeline:
    1. Extract timeline from T_inf onwards
    2. Normalize resource values
    3. Apply 1D-CNN
    4. Output v_temporal embedding
    """
    
    def __init__(
        self,
        timeline: Optional[SystemTimeline] = None,
        normalizer: Optional[ResourceNormalizer] = None,
        cnn_config: Optional[Dict] = None,
        min_timesteps: int = 5,           # Minimum number of timesteps
        max_timesteps: int = 50,          # Maximum number of timesteps
        time_granularity_ms: int = 100    # Time step granularity
    ):
        """
        Initialize Temporal Encoder.
        
        Args:
            timeline: SystemTimeline instance
            normalizer: ResourceNormalizer instance
            cnn_config: Configuration for TemporalCNN
            min_timesteps: Minimum number of timesteps to use
            max_timesteps: Maximum number of timesteps to use
            time_granularity_ms: Timestep granularity in milliseconds
        """
        super().__init__()
        
        self.timeline = timeline
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        self.time_granularity_ms = time_granularity_ms
        
        # Normalizer
        if normalizer is None:
            normalizer = ResourceNormalizer(method="minmax")
        self.normalizer = normalizer
        
        # CNN
        if cnn_config is None:
            cnn_config = {
                'in_channels': 4,
                'hidden_channels': 64,
                'output_dim': 256,
                'num_layers': 3,
                'pooling': 'adaptive_avg'
            }
        self.cnn = TemporalCNN(**cnn_config)
    
    def extract_timeline_window(
        self,
        t_inf_ms: float,
        t_end_ms: Optional[float] = None
    ) -> torch.Tensor:
        """
        Extract timeline data from T_inf onwards.
        
        Args:
            t_inf_ms: Start time (usually predicted latency)
            t_end_ms: End time (if None, use timeline's max time)
        
        Returns:
            Tensor of shape (num_timesteps, 4) with resource values
        """
        if self.timeline is None or self.timeline.timeline_df is None:
            raise RuntimeError("Timeline not loaded")
        
        # Determine time range
        if t_end_ms is None:
            t_end_ms = self.timeline.time_range[1]
        
        # Ensure we have enough time window
        if t_end_ms <= t_inf_ms:
            t_end_ms = t_inf_ms + self.min_timesteps * self.time_granularity_ms
        
        # Generate time points
        num_steps = int((t_end_ms - t_inf_ms) / self.time_granularity_ms) + 1
        num_steps = min(max(num_steps, self.min_timesteps), self.max_timesteps)
        
        time_points = [
            t_inf_ms + i * self.time_granularity_ms
            for i in range(num_steps)
        ]
        
        # Query snapshots
        snapshots = self.timeline.get_batch_snapshots(time_points)
        
        # Convert to tensor
        data = torch.tensor([
            [s['cpu_cores'], s['cpu_mem_gb'], s['gpu_sm'], s['gpu_mem_gb']]
            for s in snapshots
        ], dtype=torch.float32)
        
        return data  # (num_timesteps, 4)
    
    def forward(
        self,
        t_inf_ms: Union[float, torch.Tensor],
        t_end_ms: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode temporal resource availability.
        
        Args:
            t_inf_ms: Predicted latency or start time (scalar or tensor)
            t_end_ms: End time (if None, use timeline's max)
        
        Returns:
            v_temporal: Temporal embedding of shape (output_dim,) or (batch, output_dim)
        """
        # Handle tensor input
        if isinstance(t_inf_ms, torch.Tensor):
            if t_inf_ms.dim() == 0:  # Scalar tensor
                t_inf_ms = t_inf_ms.item()
            elif t_inf_ms.numel() == 1:  # Single element tensor (e.g., from LatencyPredictor)
                t_inf_ms = t_inf_ms.item()
            elif t_inf_ms.dim() == 1:  # Batch of scalars
                # Process each in batch
                batch_embeddings = []
                for t in t_inf_ms:
                    emb = self.forward(t.item(), t_end_ms)
                    batch_embeddings.append(emb)
                return torch.stack(batch_embeddings)  # (batch, output_dim)
            else:
                t_inf_ms = t_inf_ms.item()
        
        # Extract timeline window
        timeline_data = self.extract_timeline_window(t_inf_ms, t_end_ms)  # (T, 4)
        
        # Normalize
        normalized_data = self.normalizer.normalize(timeline_data)  # (T, 4)
        
        # Reshape for CNN: (batch=1, channels=4, time_steps=T)
        cnn_input = normalized_data.transpose(0, 1).unsqueeze(0)  # (1, 4, T)
        
        # Apply CNN
        v_temporal = self.cnn(cnn_input)  # (1, output_dim)
        
        return v_temporal.squeeze(0)  # (output_dim,)
    
    def set_timeline(self, timeline: SystemTimeline):
        """Update timeline."""
        self.timeline = timeline
    
    @classmethod
    def from_config(cls, config: Dict, timeline: Optional[SystemTimeline] = None) -> 'TemporalEncoder':
        """
        Create TemporalEncoder from configuration.
        
        Args:
            config: Configuration dictionary
            timeline: SystemTimeline instance (if None, will try to create from config)
        
        Returns:
            Initialized TemporalEncoder instance
        """
        # Get temporal encoder config
        temporal_config = config.get('runtime', {}).get('temporal_encoder', {})
        
        # Create timeline if not provided
        if timeline is None:
            timeline_config = config.get('runtime', {}).get('timeline', {})
            csv_path = timeline_config.get('csv_path')
            if csv_path:
                from .timeline import SystemTimeline
                timeline = SystemTimeline(
                    csv_path=csv_path,
                    interpolation=timeline_config.get('interpolation', 'linear')
                )
        
        # Create normalizer
        norm_method = temporal_config.get('normalization', 'minmax')
        normalizer = ResourceNormalizer(method=norm_method)
        
        # CNN config
        cnn_config = {
            'in_channels': 4,
            'hidden_channels': temporal_config.get('hidden_channels', 64),
            'output_dim': temporal_config.get('output_dim', 256),
            'num_layers': temporal_config.get('num_layers', 3),
            'pooling': temporal_config.get('pooling', 'adaptive_avg')
        }
        
        # Create encoder
        return cls(
            timeline=timeline,
            normalizer=normalizer,
            cnn_config=cnn_config,
            min_timesteps=temporal_config.get('min_timesteps', 5),
            max_timesteps=temporal_config.get('max_timesteps', 50),
            time_granularity_ms=temporal_config.get('time_granularity_ms', 100)
        )
