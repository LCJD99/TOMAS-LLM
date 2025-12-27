"""
Text Description Generator for Temporal Pretraining.

Generates natural language descriptions for resource curves:
- Type A: Trend descriptions
- Type B: Bottleneck spotting
- Type C: Feasibility QA
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class TextDescriptionGenerator:
    """
    Generates text descriptions for resource timeline curves.
    
    Supports three task types:
    - Type A: Trend Description (macro patterns)
    - Type B: Bottleneck Spotting (spike detection)
    - Type C: Feasibility QA (scheduling logic)
    """
    
    # Resource type names for text generation
    RESOURCE_TYPE_NAMES = {
        'cpu_cores': ['CPU cores', 'available CPU cores', 'CPU core availability'],
        'cpu_mem_gb': ['CPU memory', 'system memory', 'RAM'],
        'gpu_sm': ['GPU utilization', 'GPU SM usage', 'GPU compute usage'],
        'gpu_mem_gb': ['GPU memory', 'GPU VRAM', 'video memory']
    }
    
    # Trend type vocabulary
    TREND_TYPES = {
        'increasing': ['increasing', 'rising', 'growing', 'ascending', 'climbing'],
        'decreasing': ['decreasing', 'falling', 'dropping', 'descending', 'declining'],
        'stable': ['stable', 'constant', 'steady', 'flat', 'unchanging']
    }
    
    # Change description vocabulary
    CHANGE_DESCRIPTIONS = {
        'increasing': ['rising steadily', 'growing continuously', 'increasing gradually', 'climbing slowly', 'ascending smoothly'],
        'decreasing': ['dropping sharply', 'falling rapidly', 'decreasing quickly', 'declining steeply', 'reducing significantly'],
        'stable': ['remaining constant', 'staying steady', 'fluctuating minimally', 'holding stable', 'maintaining level']
    }
    
    def __init__(
        self,
        time_granularity_ms: int = 100,
        diversity_level: float = 0.8,
        seed: Optional[int] = None
    ):
        """
        Initialize text generator.
        
        Args:
            time_granularity_ms: Time step granularity in milliseconds
            diversity_level: Text diversity level (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.time_granularity_ms = time_granularity_ms
        self.diversity_level = diversity_level
        
        if seed is not None:
            random.seed(seed)
    
    def _format_resource_value(self, resource_idx: int, value: float) -> str:
        """Format resource value with appropriate units."""
        resource_name = ['cpu_cores', 'cpu_mem_gb', 'gpu_sm', 'gpu_mem_gb'][resource_idx]
        
        if resource_name == 'cpu_cores':
            return f"{int(value)} cores"
        elif resource_name == 'cpu_mem_gb':
            if value >= 1:
                return f"{value:.1f}GB"
            else:
                return f"{int(value * 1024)}MB"
        elif resource_name == 'gpu_sm':
            return f"{value:.1f}%"
        elif resource_name == 'gpu_mem_gb':
            if value >= 1:
                return f"{value:.1f}GB"
            else:
                return f"{int(value * 1024)}MB"
        return str(value)
    
    def _get_resource_type_name(self, resource_idx: int) -> str:
        """Get random resource type name."""
        resource_name = ['cpu_cores', 'cpu_mem_gb', 'gpu_sm', 'gpu_mem_gb'][resource_idx]
        return random.choice(self.RESOURCE_TYPE_NAMES[resource_name])
    
    def _detect_trend(self, curve: np.ndarray) -> Tuple[str, float]:
        """
        Detect trend type and magnitude.
        
        Args:
            curve: Resource values array
        
        Returns:
            Trend type and change magnitude
        """
        start_val = np.mean(curve[:len(curve)//10])  # First 10%
        end_val = np.mean(curve[-len(curve)//10:])   # Last 10%
        
        change = end_val - start_val
        relative_change = change / (start_val + 1e-6)
        
        # Determine trend
        if abs(relative_change) < 0.1:
            trend = 'stable'
        elif change > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return trend, abs(change)
    
    def generate_type_a(
        self,
        timeline: torch.Tensor,
        metadata: Dict,
        resource_idx: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Generate Type A: Trend Description.
        
        Args:
            timeline: Timeline tensor (num_timesteps, 4)
            metadata: Timeline metadata
            resource_idx: Resource to describe (None for random)
        
        Returns:
            (prompt, target) strings
        """
        if resource_idx is None:
            resource_idx = random.randint(0, 3)
        
        curve = timeline[:, resource_idx].numpy()
        duration_s = metadata['duration_s']
        
        # Detect trend
        trend_type, change_magnitude = self._detect_trend(curve)
        
        # Get values
        start_val = curve[0]
        end_val = curve[-1]
        
        # Generate prompt
        resource_type = self._get_resource_type_name(resource_idx)
        prompt = f"Analyze the {resource_type} trend over the next {duration_s:.1f} seconds."
        
        # Generate target
        trend_word = random.choice(self.TREND_TYPES[trend_type])
        change_desc = random.choice(self.CHANGE_DESCRIPTIONS[trend_type])
        
        start_str = self._format_resource_value(resource_idx, start_val)
        end_str = self._format_resource_value(resource_idx, end_val)
        
        # Template variations
        templates = [
            f"{resource_type} shows a {trend_word} trend, {change_desc} from {start_str} to {end_str} over {duration_s:.1f} seconds.",
            f"The {resource_type} exhibits a {trend_word} pattern, {change_desc} from {start_str} to {end_str}.",
            f"Over {duration_s:.1f} seconds, {resource_type} is {trend_word}, {change_desc} from {start_str} to {end_str}.",
        ]
        
        target = random.choice(templates)
        
        # 30% chance to add quantitative description
        if random.random() < 0.3 and trend_type != 'stable':
            rate = change_magnitude / duration_s
            rate_str = self._format_resource_value(resource_idx, rate)
            target += f" The rate of change is approximately {rate_str} per second."
        
        return prompt, target
    
    def generate_type_b(
        self,
        timeline: torch.Tensor,
        metadata: Dict,
        spike_info: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """
        Generate Type B: Bottleneck Spotting.
        
        Args:
            timeline: Timeline tensor (num_timesteps, 4)
            metadata: Timeline metadata
            spike_info: Pre-existing spike information (or will be found)
        
        Returns:
            (prompt, target) strings
        """
        # If no spike info provided, find minimum in a random resource
        if not spike_info:
            resource_idx = random.randint(0, 3)
            curve = timeline[:, resource_idx].numpy()
            min_idx = np.argmin(curve)
            min_val = curve[min_idx]
            time_s = min_idx * self.time_granularity_ms / 1000.0
        else:
            # Use provided spike info
            spike = random.choice(spike_info)
            resource_idx = spike['resource_idx']
            min_val = spike['min_value']
            time_s = spike['time_s']
        
        # Generate prompt
        resource_type = self._get_resource_type_name(resource_idx)
        
        prompt_templates = [
            f"Identify the minimum available {resource_type} in the timeline.",
            f"Find the lowest point of {resource_type} availability.",
            f"Locate the bottleneck in {resource_type} usage.",
        ]
        prompt = random.choice(prompt_templates)
        
        # Generate target
        min_str = self._format_resource_value(resource_idx, min_val)
        
        target_templates = [
            f"The minimum available {resource_type} drops to {min_str} at t={time_s:.1f}s.",
            f"The lowest {resource_type} availability is {min_str}, occurring at t={time_s:.1f}s.",
            f"{resource_type} reaches its minimum of {min_str} at time t={time_s:.1f}s.",
        ]
        target = random.choice(target_templates)
        
        # 50% chance to add duration info if spike info available
        if spike_info and random.random() < 0.5:
            spike = spike_info[0] if not isinstance(spike_info, dict) else spike_info
            if 'duration' in spike:
                duration_s = spike['duration'] * self.time_granularity_ms / 1000.0
                target += f" This bottleneck lasts for approximately {duration_s:.2f} seconds."
        
        return prompt, target
    
    def generate_type_c(
        self,
        timeline: torch.Tensor,
        metadata: Dict,
        resource_idx: Optional[int] = None,
        requirement_multiplier: Optional[float] = None
    ) -> Tuple[str, str]:
        """
        Generate Type C: Feasibility QA.
        
        Args:
            timeline: Timeline tensor (num_timesteps, 4)
            metadata: Timeline metadata
            resource_idx: Resource to check (None for random)
            requirement_multiplier: Requirement as multiple of average (None for random)
        
        Returns:
            (prompt, target) strings
        """
        if resource_idx is None:
            resource_idx = random.randint(0, 3)
        
        curve = timeline[:, resource_idx].numpy()
        avg_val = np.mean(curve)
        current_val = curve[0]
        max_val = np.max(curve)
        
        # Generate requirement
        if requirement_multiplier is None:
            requirement_multiplier = random.choice([0.8, 1.2, 1.5])
        
        requirement = avg_val * requirement_multiplier
        requirement_str = self._format_resource_value(resource_idx, requirement)
        
        # Check feasibility
        immediately_feasible = current_val >= requirement
        eventually_feasible = max_val >= requirement
        
        # Find when it becomes available
        if eventually_feasible and not immediately_feasible:
            available_idx = np.where(curve >= requirement)[0]
            if len(available_idx) > 0:
                first_available = available_idx[0]
                available_time_s = first_available * self.time_granularity_ms / 1000.0
                future_val = curve[first_available]
            else:
                eventually_feasible = False
        
        # Generate prompt
        resource_type = self._get_resource_type_name(resource_idx)
        prompt = f"Can a task requiring {requirement_str} {resource_type} be scheduled immediately?"
        
        # Generate target based on feasibility
        current_str = self._format_resource_value(resource_idx, current_val)
        
        if immediately_feasible:
            target_templates = [
                f"Yes, {resource_type} has {current_str} available, which exceeds the required {requirement_str}.",
                f"Yes, current {resource_type} availability ({current_str}) satisfies the requirement of {requirement_str}.",
                f"Yes, the task can be scheduled immediately with {current_str} {resource_type} available.",
            ]
            target = random.choice(target_templates)
        
        elif eventually_feasible:
            future_str = self._format_resource_value(resource_idx, future_val)
            target_templates = [
                f"No, current available {resource_type} is {current_str}. It will become available after t={available_time_s:.1f}s when it reaches {future_str}.",
                f"No, immediate scheduling is not possible. {resource_type} will be sufficient ({future_str}) at t={available_time_s:.1f}s.",
                f"Not immediately. Current {resource_type} is {current_str}, but it will reach {future_str} at t={available_time_s:.1f}s.",
            ]
            target = random.choice(target_templates)
        
        else:
            max_str = self._format_resource_value(resource_idx, max_val)
            target_templates = [
                f"No, the required {requirement_str} exceeds the maximum available {resource_type} ({max_str}) throughout the timeline.",
                f"No, this task cannot be scheduled. Maximum {resource_type} availability is only {max_str}.",
                f"Not feasible. The requirement ({requirement_str}) exceeds peak {resource_type} availability ({max_str}).",
            ]
            target = random.choice(target_templates)
        
        return prompt, target
    
    def generate(
        self,
        task_type: str,
        timeline: torch.Tensor,
        metadata: Dict,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Generate prompt-target pair for any task type.
        
        Args:
            task_type: "A", "B", or "C"
            timeline: Timeline tensor
            metadata: Timeline metadata
            **kwargs: Additional arguments for specific generators
        
        Returns:
            (prompt, target) strings
        """
        if task_type == "A":
            return self.generate_type_a(timeline, metadata, **kwargs)
        elif task_type == "B":
            return self.generate_type_b(timeline, metadata, **kwargs)
        elif task_type == "C":
            return self.generate_type_c(timeline, metadata, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
