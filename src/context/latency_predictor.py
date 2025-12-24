"""
Latency Prediction Module (T_inf).

This module predicts the inference latency for tool execution based on
tool characteristics and resource allocation. Currently implements a naive
approach with fixed/rule-based predictions.

Future: Can be replaced with learned latency predictor.
"""

import logging
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LatencyPredictor(nn.Module):
    """
    Predicts tool execution latency (T_inf) in milliseconds.
    
    Modes:
    1. Fixed: Returns constant latency for all tools
    2. Rule-based: Simple heuristics based on tool type and resources
    3. Learned: Neural network predictor (future implementation)
    """
    
    def __init__(
        self,
        mode: str = "fixed",
        fixed_latency_ms: float = 500.0,
        enable_learning: bool = False,
        hidden_dim: int = 128
    ):
        """
        Initialize latency predictor.
        
        Args:
            mode: Prediction mode ("fixed", "rule_based", "learned")
            fixed_latency_ms: Fixed latency value (for "fixed" mode)
            enable_learning: Whether to enable learned prediction
            hidden_dim: Hidden dimension for learned predictor
        """
        super().__init__()
        
        self.mode = mode
        self.fixed_latency_ms = fixed_latency_ms
        self.enable_learning = enable_learning
        
        # Rule-based latency lookup table (tool_name -> base_latency_ms)
        self.latency_table = {
            'image_classification': 150.0,
            'text_summarization': 300.0,
            'video_transcoding': 2000.0,
            'sentiment_analysis': 100.0,
            'object_detection': 200.0,
            'machine_translation': 400.0,
            'speech_recognition': 800.0,
            'data_preprocessing': 250.0,
            # Default for unknown tools
            'default': 500.0
        }
        
        # Learned predictor (optional)
        if enable_learning:
            # Input: [tool_embedding (optional), resource_vector (6D)]
            # For naive version, just use resource vector
            self.predictor = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Ensure positive output
            )
            logger.info(f"Initialized learned latency predictor with {hidden_dim}D hidden layer")
        else:
            self.predictor = None
        
        logger.info(f"Initialized LatencyPredictor: mode={mode}, "
                   f"fixed_latency={fixed_latency_ms}ms, learning={enable_learning}")
    
    def predict_fixed(self, batch_size: int = 1) -> torch.Tensor:
        """
        Return fixed latency for all samples.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Latency predictions (batch_size,) in milliseconds
        """
        return torch.full((batch_size,), self.fixed_latency_ms)
    
    def predict_rule_based(
        self,
        tool_names: Optional[List[str]] = None,
        resource_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict latency using simple rules.
        
        Rules:
        1. Base latency from tool type
        2. Scale by input size (from resource_vector[0])
        3. Adjust for GPU availability (from resource_vector[3:5])
        
        Args:
            tool_names: List of tool names
            resource_vectors: Resource vectors (batch, 6)
                [input_size, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb, latency_ms]
        
        Returns:
            Latency predictions (batch,) in milliseconds
        """
        if tool_names is None:
            # No tool info, use default
            batch_size = resource_vectors.size(0) if resource_vectors is not None else 1
            base_latency = torch.full((batch_size,), self.latency_table['default'])
        else:
            # Get base latency from table
            base_latency = torch.tensor([
                self.latency_table.get(name, self.latency_table['default'])
                for name in tool_names
            ])
        
        if resource_vectors is None:
            return base_latency
        
        # Extract features
        input_size = resource_vectors[:, 0]  # Normalized input size
        gpu_sm = resource_vectors[:, 3]      # GPU SM utilization
        gpu_mem = resource_vectors[:, 4]     # GPU memory
        
        # Rule 1: Scale by input size
        # input_size is normalized: -1 (small), 0 (medium), 1 (large)
        # Map to scale factors: 0.5, 1.0, 2.0
        size_scale = torch.where(
            input_size < -0.5,
            torch.tensor(0.5),
            torch.where(input_size > 0.5, torch.tensor(2.0), torch.tensor(1.0))
        )
        
        # Rule 2: GPU acceleration factor
        # If GPU available (gpu_sm > 0 or gpu_mem > 0), reduce latency
        has_gpu = ((gpu_sm > 0) | (gpu_mem > 0)).float()
        gpu_factor = 1.0 - 0.3 * has_gpu  # 30% reduction with GPU
        
        # Combine rules
        predicted_latency = base_latency * size_scale * gpu_factor
        
        return predicted_latency
    
    def predict_learned(
        self,
        resource_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict latency using learned model.
        
        Args:
            resource_vectors: Resource vectors (batch, 6)
        
        Returns:
            Latency predictions (batch,) in milliseconds
        """
        if self.predictor is None:
            raise ValueError("Learned predictor not initialized. Set enable_learning=True")
        
        # Forward through MLP
        latency = self.predictor(resource_vectors).squeeze(-1)
        
        return latency
    
    def forward(
        self,
        tool_names: Optional[Union[str, List[str]]] = None,
        resource_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict latency based on configured mode.
        
        Args:
            tool_names: Tool name(s)
            resource_vectors: Resource vectors (batch, 6)
        
        Returns:
            Latency predictions (batch,) in milliseconds
        """
        # Handle single tool name
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        
        # Determine batch size
        if resource_vectors is not None:
            batch_size = resource_vectors.size(0)
        elif tool_names is not None:
            batch_size = len(tool_names)
        else:
            batch_size = 1
        
        # Predict based on mode
        if self.mode == "fixed":
            return self.predict_fixed(batch_size)
        
        elif self.mode == "rule_based":
            return self.predict_rule_based(tool_names, resource_vectors)
        
        elif self.mode == "learned":
            if resource_vectors is None:
                raise ValueError("resource_vectors required for learned mode")
            return self.predict_learned(resource_vectors)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def set_mode(self, mode: str):
        """Change prediction mode."""
        if mode not in ["fixed", "rule_based", "learned"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        logger.info(f"Latency predictor mode changed to: {mode}")
    
    def update_latency_table(self, tool_name: str, latency_ms: float):
        """Update latency table for specific tool."""
        self.latency_table[tool_name] = latency_ms
        logger.info(f"Updated latency for {tool_name}: {latency_ms}ms")
    
    @classmethod
    def from_config(cls, config: Dict) -> 'LatencyPredictor':
        """
        Create LatencyPredictor from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized LatencyPredictor instance
        """
        latency_config = config['model']['latency_predictor']
        
        # Determine mode
        if not latency_config.get('enabled', False):
            mode = 'fixed'
        else:
            mode = latency_config.get('mode', 'rule_based')
        
        return cls(
            mode=mode,
            fixed_latency_ms=latency_config.get('fixed_t_inf_ms', 500.0),
            enable_learning=latency_config.get('enable_learning', False),
            hidden_dim=latency_config.get('hidden_dim', 128)
        )
    
    def extra_repr(self) -> str:
        """String representation."""
        return f"mode={self.mode}, fixed_latency={self.fixed_latency_ms}ms"


class LatencyAwareModule(nn.Module):
    """
    Wrapper that combines latency prediction with other components.
    
    This can be used to integrate latency awareness into the tool planning
    pipeline, allowing the model to consider execution time when selecting tools.
    """
    
    def __init__(
        self,
        latency_predictor: LatencyPredictor,
        use_latency_in_planning: bool = True
    ):
        """
        Initialize latency-aware module.
        
        Args:
            latency_predictor: LatencyPredictor instance
            use_latency_in_planning: Whether to use latency in planning decisions
        """
        super().__init__()
        
        self.latency_predictor = latency_predictor
        self.use_latency_in_planning = use_latency_in_planning
        
        logger.info(f"Initialized LatencyAwareModule: use_in_planning={use_latency_in_planning}")
    
    def forward(
        self,
        tool_names: Optional[List[str]] = None,
        resource_vectors: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict latencies and return structured output.
        
        Args:
            tool_names: Tool names
            resource_vectors: Resource vectors
        
        Returns:
            Dictionary with:
                - latencies: Predicted latencies (batch,)
                - latency_penalty: Penalty term for optimization (optional)
        """
        # Predict latencies
        latencies = self.latency_predictor(tool_names, resource_vectors)
        
        output = {
            'latencies': latencies
        }
        
        # Optionally compute penalty for long latencies
        if self.use_latency_in_planning:
            # Penalty increases with latency (log scale)
            # This can be used in loss functions
            latency_penalty = torch.log(1 + latencies / 100.0)  # Normalize to ~100ms scale
            output['latency_penalty'] = latency_penalty
        
        return output
    
    @classmethod
    def from_config(cls, config: Dict) -> 'LatencyAwareModule':
        """Create from configuration."""
        latency_predictor = LatencyPredictor.from_config(config)
        latency_config = config['model']['latency_predictor']
        
        return cls(
            latency_predictor=latency_predictor,
            use_latency_in_planning=latency_config.get('use_in_planning', True)
        )
