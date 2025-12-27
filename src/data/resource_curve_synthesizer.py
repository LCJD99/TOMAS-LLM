"""
Resource Curve Synthesizer for Temporal Pretraining.

Generates synthetic resource timeline curves with various patterns:
- Linear, Exponential, Sinusoidal, Step, Plateau trends
- Gaussian noise and spike injection
- Physical constraints validation
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ResourceCurveSynthesizer:
    """
    Synthesizes diverse resource timeline curves for training.
    
    Generates curves with different trend patterns, noise, and spikes
    while maintaining physical constraints.
    """
    
    # Resource value ranges
    RESOURCE_RANGES = {
        'cpu_cores': (0.0, 128.0),
        'cpu_mem_gb': (0.0, 512.0),
        'gpu_sm': (0.0, 100.0),
        'gpu_mem_gb': (0.0, 80.0)
    }
    
    # Resource names mapping
    RESOURCE_NAMES = ['cpu_cores', 'cpu_mem_gb', 'gpu_sm', 'gpu_mem_gb']
    
    def __init__(
        self,
        time_granularity_ms: int = 100,
        noise_level: float = 0.05,
        spike_probability: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize synthesizer.
        
        Args:
            time_granularity_ms: Time step granularity in milliseconds
            noise_level: Gaussian noise standard deviation (as fraction of value)
            spike_probability: Probability of injecting a spike
            seed: Random seed for reproducibility
        """
        self.time_granularity_ms = time_granularity_ms
        self.noise_level = noise_level
        self.spike_probability = spike_probability
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_curve(
        self,
        num_timesteps: int,
        pattern: Optional[str] = None,
        resource_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a single resource curve.
        
        Args:
            num_timesteps: Number of time steps
            pattern: Trend pattern ('linear', 'exponential', 'sinusoidal', 'step', 'plateau', None for random)
            resource_idx: Resource index (0-3, None for random)
        
        Returns:
            Array of shape (num_timesteps,) with resource values
        """
        if pattern is None:
            pattern = random.choice(['linear', 'exponential', 'sinusoidal', 'step', 'plateau'])
        
        if resource_idx is None:
            resource_idx = random.randint(0, 3)
        
        resource_name = self.RESOURCE_NAMES[resource_idx]
        min_val, max_val = self.RESOURCE_RANGES[resource_name]
        
        # Generate base curve
        t = np.linspace(0, 1, num_timesteps)
        
        if pattern == 'linear':
            # Linear increase or decrease
            direction = random.choice([1, -1])
            start = random.uniform(min_val + 0.2 * (max_val - min_val), 
                                  max_val - 0.2 * (max_val - min_val))
            end = start + direction * random.uniform(0.3, 0.7) * (max_val - min_val)
            curve = np.linspace(start, end, num_timesteps)
        
        elif pattern == 'exponential':
            # Exponential growth or decay
            direction = random.choice([1, -1])
            if direction == 1:  # Growth
                start = random.uniform(min_val, min_val + 0.3 * (max_val - min_val))
                end = random.uniform(max_val - 0.3 * (max_val - min_val), max_val)
                curve = start + (end - start) * (np.exp(3 * t) - 1) / (np.exp(3) - 1)
            else:  # Decay
                start = random.uniform(max_val - 0.3 * (max_val - min_val), max_val)
                end = random.uniform(min_val, min_val + 0.3 * (max_val - min_val))
                curve = start + (end - start) * (1 - np.exp(-3 * t))
        
        elif pattern == 'sinusoidal':
            # Sinusoidal fluctuation
            mean = random.uniform(min_val + 0.3 * (max_val - min_val), 
                                 max_val - 0.3 * (max_val - min_val))
            amplitude = random.uniform(0.1, 0.3) * (max_val - min_val)
            frequency = random.uniform(1, 3)
            phase = random.uniform(0, 2 * np.pi)
            curve = mean + amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        elif pattern == 'step':
            # Step change
            step_point = random.randint(num_timesteps // 4, 3 * num_timesteps // 4)
            level1 = random.uniform(min_val + 0.2 * (max_val - min_val), 
                                   max_val - 0.4 * (max_val - min_val))
            level2 = random.uniform(min_val + 0.2 * (max_val - min_val), 
                                   max_val - 0.2 * (max_val - min_val))
            curve = np.concatenate([
                np.full(step_point, level1),
                np.full(num_timesteps - step_point, level2)
            ])
        
        elif pattern == 'plateau':
            # Plateau with sudden change
            plateau_end = random.randint(num_timesteps // 3, 2 * num_timesteps // 3)
            plateau_level = random.uniform(min_val + 0.3 * (max_val - min_val), 
                                          max_val - 0.3 * (max_val - min_val))
            
            # Sudden change
            change_direction = random.choice([1, -1])
            final_level = plateau_level + change_direction * random.uniform(0.2, 0.5) * (max_val - min_val)
            
            curve = np.concatenate([
                np.full(plateau_end, plateau_level),
                np.linspace(plateau_level, final_level, num_timesteps - plateau_end)
            ])
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Apply physical constraints
        curve = np.clip(curve, min_val, max_val)
        
        # Add Gaussian noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * (max_val - min_val), num_timesteps)
            curve = curve + noise
            curve = np.clip(curve, min_val, max_val)
        
        return curve
    
    def inject_spike(
        self,
        curve: np.ndarray,
        resource_idx: int,
        num_spikes: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Inject spikes (sudden drops) into a curve.
        
        Args:
            curve: Original curve array
            resource_idx: Resource index for constraint checking
            num_spikes: Number of spikes to inject (None for random 1-3)
        
        Returns:
            Modified curve and list of spike metadata
        """
        if num_spikes is None:
            num_spikes = random.randint(1, 3)
        
        resource_name = self.RESOURCE_NAMES[resource_idx]
        min_val, max_val = self.RESOURCE_RANGES[resource_name]
        
        num_timesteps = len(curve)
        curve_with_spikes = curve.copy()
        spike_info = []
        
        # Avoid boundary (first and last 10%)
        valid_range = range(int(0.1 * num_timesteps), int(0.9 * num_timesteps))
        
        for _ in range(num_spikes):
            # Random spike location
            spike_start = random.choice(list(valid_range))
            spike_duration = random.randint(1, 5)  # 1-5 timesteps
            spike_end = min(spike_start + spike_duration, num_timesteps)
            
            # Spike depth: drop to 20%-50% of original value
            depth_factor = random.uniform(0.2, 0.5)
            
            for i in range(spike_start, spike_end):
                original_val = curve_with_spikes[i]
                spike_val = original_val * depth_factor
                curve_with_spikes[i] = max(spike_val, min_val)
            
            # Record spike info
            min_spike_val = curve_with_spikes[spike_start:spike_end].min()
            spike_time_ms = spike_start * self.time_granularity_ms
            
            spike_info.append({
                'position': spike_start,
                'time_ms': spike_time_ms,
                'time_s': spike_time_ms / 1000.0,
                'duration': spike_end - spike_start,
                'min_value': min_spike_val,
                'resource_idx': resource_idx
            })
        
        return curve_with_spikes, spike_info
    
    def generate_full_timeline(
        self,
        num_timesteps: Optional[int] = None,
        patterns: Optional[List[str]] = None,
        inject_spikes: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate a full 4-resource timeline.
        
        Args:
            num_timesteps: Number of time steps (None for random [20, 100])
            patterns: List of 4 patterns for each resource (None for random)
            inject_spikes: Whether to inject spikes
        
        Returns:
            Tensor of shape (num_timesteps, 4) and metadata dict
        """
        if num_timesteps is None:
            num_timesteps = random.randint(20, 100)
        
        if patterns is None:
            patterns = [None] * 4
        
        # Generate curves for each resource
        curves = []
        all_spike_info = []
        
        for resource_idx in range(4):
            curve = self.generate_curve(
                num_timesteps=num_timesteps,
                pattern=patterns[resource_idx],
                resource_idx=resource_idx
            )
            
            # Inject spikes if requested
            if inject_spikes and random.random() < self.spike_probability:
                curve, spike_info = self.inject_spike(curve, resource_idx)
                all_spike_info.extend(spike_info)
            
            curves.append(curve)
        
        # Stack to (num_timesteps, 4)
        timeline = np.stack(curves, axis=1)
        
        # Convert to tensor
        timeline_tensor = torch.tensor(timeline, dtype=torch.float32)
        
        # Metadata
        metadata = {
            'num_timesteps': num_timesteps,
            'duration_s': num_timesteps * self.time_granularity_ms / 1000.0,
            'time_granularity_ms': self.time_granularity_ms,
            'patterns': patterns,
            'spikes': all_spike_info if inject_spikes else []
        }
        
        return timeline_tensor, metadata
    
    def validate_physical_constraints(self, timeline: torch.Tensor) -> bool:
        """
        Validate that timeline satisfies physical constraints.
        
        Args:
            timeline: Tensor of shape (num_timesteps, 4)
        
        Returns:
            True if valid, False otherwise
        """
        timeline_np = timeline.numpy() if isinstance(timeline, torch.Tensor) else timeline
        
        for resource_idx, resource_name in enumerate(self.RESOURCE_NAMES):
            min_val, max_val = self.RESOURCE_RANGES[resource_name]
            resource_values = timeline_np[:, resource_idx]
            
            # Check bounds
            if np.any(resource_values < min_val) or np.any(resource_values > max_val):
                return False
            
            # Check for extreme single-step changes (>50% except for spikes)
            diffs = np.abs(np.diff(resource_values))
            max_allowed_change = 0.5 * (max_val - min_val)
            
            # Allow some violations for intentional spikes
            extreme_changes = diffs > max_allowed_change
            if np.sum(extreme_changes) > len(diffs) * 0.1:  # More than 10% extreme changes
                return False
        
        return True
