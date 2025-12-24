"""
System Timeline & Resource Snapshot Management.

This module handles:
1. Reading system resource timeline from CSV
2. Predicting resource availability at future timestamps (especially T_inf)
3. Providing resource snapshots for tool planning

Data Format (input/system_profiling.csv):
    time_ms, cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb
    0, 16, 64.0, 80, 40.0
    100, 16, 64.0, 80, 40.0
    ...

Where:
    - time_ms: Time from submission (milliseconds)
    - cpu_cores: Available CPU cores
    - cpu_mem_gb: Available CPU memory (GB)
    - gpu_sm: Available GPU SM percentage (0-100)
    - gpu_mem_gb: Available GPU memory (GB)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SystemTimeline:
    """
    Manages system resource timeline data.
    
    Reads resource availability from CSV and provides query interface
    for resource snapshots at specific timestamps.
    """
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        interpolation: str = "linear"
    ):
        """
        Initialize system timeline.
        
        Args:
            csv_path: Path to system_profiling.csv
            interpolation: Interpolation method ("linear", "nearest", "previous")
        """
        self.csv_path = csv_path
        self.interpolation = interpolation
        
        self.timeline_df = None
        self.time_range = None
        
        if csv_path is not None:
            self.load_timeline(csv_path)
    
    def load_timeline(self, csv_path: str):
        """
        Load resource timeline from CSV file.
        
        Expected columns: time_ms, cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Timeline CSV not found: {csv_path}")
        
        # Load CSV
        self.timeline_df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ['time_ms', 'cpu_cores', 'cpu_mem_gb', 'gpu_sm', 'gpu_mem_gb']
        missing = set(required_cols) - set(self.timeline_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Sort by time
        self.timeline_df = self.timeline_df.sort_values('time_ms').reset_index(drop=True)
        
        # Store time range
        self.time_range = (
            self.timeline_df['time_ms'].min(),
            self.timeline_df['time_ms'].max()
        )
        
        logger.info(f"Loaded timeline: {len(self.timeline_df)} snapshots, "
                   f"time range: {self.time_range[0]:.0f}ms - {self.time_range[1]:.0f}ms")
    
    def get_snapshot_at(
        self,
        time_ms: float,
        allow_extrapolation: bool = False
    ) -> Dict[str, float]:
        """
        Get resource snapshot at specific timestamp.
        
        Args:
            time_ms: Timestamp in milliseconds
            allow_extrapolation: If True, allow queries beyond time range
        
        Returns:
            Dictionary with resource values:
                {
                    'cpu_cores': float,
                    'cpu_mem_gb': float,
                    'gpu_sm': float,
                    'gpu_mem_gb': float
                }
        """
        if self.timeline_df is None:
            raise RuntimeError("Timeline not loaded. Call load_timeline() first.")
        
        # Check bounds
        if not allow_extrapolation:
            if time_ms < self.time_range[0] or time_ms > self.time_range[1]:
                raise ValueError(
                    f"Time {time_ms}ms out of range {self.time_range}. "
                    f"Set allow_extrapolation=True to allow."
                )
        
        # Find closest timestamps
        times = self.timeline_df['time_ms'].values
        
        if self.interpolation == "nearest":
            # Nearest neighbor
            idx = np.argmin(np.abs(times - time_ms))
            row = self.timeline_df.iloc[idx]
        
        elif self.interpolation == "previous":
            # Previous value (step function)
            idx = np.searchsorted(times, time_ms, side='right') - 1
            idx = max(0, min(idx, len(times) - 1))
            row = self.timeline_df.iloc[idx]
        
        elif self.interpolation == "linear":
            # Linear interpolation
            if time_ms <= times[0]:
                row = self.timeline_df.iloc[0]
            elif time_ms >= times[-1]:
                row = self.timeline_df.iloc[-1]
            else:
                # Find surrounding points
                idx_after = np.searchsorted(times, time_ms)
                idx_before = idx_after - 1
                
                t0, t1 = times[idx_before], times[idx_after]
                alpha = (time_ms - t0) / (t1 - t0)
                
                # Interpolate each resource
                row0 = self.timeline_df.iloc[idx_before]
                row1 = self.timeline_df.iloc[idx_after]
                
                return {
                    'cpu_cores': row0['cpu_cores'] * (1 - alpha) + row1['cpu_cores'] * alpha,
                    'cpu_mem_gb': row0['cpu_mem_gb'] * (1 - alpha) + row1['cpu_mem_gb'] * alpha,
                    'gpu_sm': row0['gpu_sm'] * (1 - alpha) + row1['gpu_sm'] * alpha,
                    'gpu_mem_gb': row0['gpu_mem_gb'] * (1 - alpha) + row1['gpu_mem_gb'] * alpha
                }
        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")
        
        # Return snapshot
        return {
            'cpu_cores': float(row['cpu_cores']),
            'cpu_mem_gb': float(row['cpu_mem_gb']),
            'gpu_sm': float(row['gpu_sm']),
            'gpu_mem_gb': float(row['gpu_mem_gb'])
        }
    
    def get_batch_snapshots(
        self,
        times_ms: List[float]
    ) -> List[Dict[str, float]]:
        """
        Get multiple resource snapshots.
        
        Args:
            times_ms: List of timestamps
        
        Returns:
            List of resource dictionaries
        """
        return [self.get_snapshot_at(t) for t in times_ms]
    
    def to_tensor(
        self,
        snapshot: Dict[str, float],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Convert snapshot to tensor.
        
        Args:
            snapshot: Resource snapshot dictionary
            device: Target device
        
        Returns:
            Tensor of shape (4,) with [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]
        """
        return torch.tensor([
            snapshot['cpu_cores'],
            snapshot['cpu_mem_gb'],
            snapshot['gpu_sm'],
            snapshot['gpu_mem_gb']
        ], dtype=torch.float32, device=device)
    
    @classmethod
    def from_config(cls, config: Dict) -> 'SystemTimeline':
        """
        Create SystemTimeline from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized SystemTimeline instance
        """
        csv_path = config.get('runtime', {}).get('timeline', {}).get('csv_path')
        interpolation = config.get('runtime', {}).get('timeline', {}).get('interpolation', 'linear')
        
        return cls(csv_path=csv_path, interpolation=interpolation)


class ResourcePredictor(nn.Module):
    """
    Predicts resource availability at T_inf (after latency).
    
    Given:
    - Current system timeline
    - Predicted latency T_inf
    
    Returns:
    - Resource snapshot at time T_inf
    """
    
    def __init__(
        self,
        timeline: Optional[SystemTimeline] = None,
        default_snapshot: Optional[Dict[str, float]] = None
    ):
        """
        Initialize resource predictor.
        
        Args:
            timeline: SystemTimeline instance
            default_snapshot: Default resource values if timeline unavailable
        """
        super().__init__()
        
        self.timeline = timeline
        
        # Default snapshot if timeline not available
        if default_snapshot is None:
            default_snapshot = {
                'cpu_cores': 16.0,
                'cpu_mem_gb': 64.0,
                'gpu_sm': 80.0,
                'gpu_mem_gb': 40.0
            }
        self.default_snapshot = default_snapshot
    
    def forward(
        self,
        t_inf_ms: Union[float, torch.Tensor],
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, float]]:
        """
        Predict resource availability at T_inf.
        
        Args:
            t_inf_ms: Predicted latency in milliseconds (scalar or tensor)
            return_dict: If True, return dictionary; else return tensor
        
        Returns:
            If return_dict=True:
                Dictionary with resource values
            If return_dict=False:
                Tensor (4,) with [cpu_cores, cpu_mem_gb, gpu_sm, gpu_mem_gb]
        """
        # Handle tensor input
        if isinstance(t_inf_ms, torch.Tensor):
            t_inf_ms = t_inf_ms.item()
        
        # Get snapshot from timeline
        if self.timeline is not None and self.timeline.timeline_df is not None:
            try:
                snapshot = self.timeline.get_snapshot_at(
                    t_inf_ms,
                    allow_extrapolation=True
                )
            except Exception as e:
                logger.warning(f"Timeline query failed: {e}. Using default snapshot.")
                snapshot = self.default_snapshot
        else:
            snapshot = self.default_snapshot
        
        # Return format
        if return_dict:
            return snapshot
        else:
            return self.timeline.to_tensor(snapshot) if self.timeline else torch.tensor([
                snapshot['cpu_cores'],
                snapshot['cpu_mem_gb'],
                snapshot['gpu_sm'],
                snapshot['gpu_mem_gb']
            ], dtype=torch.float32)
    
    def set_timeline(self, timeline: SystemTimeline):
        """Update timeline."""
        self.timeline = timeline
    
    def set_default_snapshot(self, snapshot: Dict[str, float]):
        """Update default snapshot."""
        self.default_snapshot = snapshot
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ResourcePredictor':
        """
        Create ResourcePredictor from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized ResourcePredictor instance
        """
        # Try to load timeline if path provided
        timeline_config = config.get('runtime', {}).get('timeline', {})
        csv_path = timeline_config.get('csv_path')
        
        timeline = None
        if csv_path:
            try:
                timeline = SystemTimeline(csv_path, interpolation=timeline_config.get('interpolation', 'linear'))
            except Exception as e:
                logger.warning(f"Failed to load timeline: {e}")
        
        # Get default snapshot
        naive_mode = config.get('runtime', {}).get('naive_mode', {})
        default_snapshot = None
        if naive_mode.get('enabled'):
            # Use first values from naive mode as default
            default_snapshot = {
                'cpu_cores': float(naive_mode.get('fixed_cpu_free_cores', [16])[0]),
                'cpu_mem_gb': float(naive_mode.get('fixed_cpu_mem_gb', [64])[0]),
                'gpu_sm': float(naive_mode.get('fixed_gpu_free_sm_ratio', [0.8])[0]) * 100,
                'gpu_mem_gb': float(naive_mode.get('fixed_gpu_free_mem_gb', [40])[0])
            }
        
        return cls(timeline=timeline, default_snapshot=default_snapshot)
