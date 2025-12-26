"""
Data loaders for tool registry and profiling data.

This module provides utilities to load, validate, and join tool semantic
information with resource profiling data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError

from src.schemas.tool_schema import ToolSchema, ProfilingSchema


logger = logging.getLogger(__name__)


class ToolRegistryLoader:
    """
    Loader for tool registry (semantic descriptions).
    
    Loads tool definitions from JSON file and validates them against schema.
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize the tool registry loader.
        
        Args:
            registry_path: Path to tools.json file
        """
        self.registry_path = Path(registry_path)
        self.tools: List[ToolSchema] = []
        self.tool_dict: Dict[str, ToolSchema] = {}
        
    def load(self) -> Tuple[List[ToolSchema], Dict[str, ToolSchema]]:
        """
        Load and validate tool registry from JSON.
        
        Returns:
            Tuple of (tool_list, tool_dict) where:
                - tool_list: List of validated ToolSchema objects
                - tool_dict: Dict mapping tool name to ToolSchema
        
        Raises:
            FileNotFoundError: If registry file doesn't exist
            ValueError: If validation fails or duplicate tool names found
        """
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Tool registry not found: {self.registry_path}")
        
        logger.info(f"Loading tool registry from: {self.registry_path}")
        
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            raw_tools = json.load(f)
        
        if not isinstance(raw_tools, list):
            raise ValueError("Tool registry must be a JSON array")
        
        # Validate each tool
        validated_tools = []
        seen_names = set()
        
        for idx, tool_data in enumerate(raw_tools):
            try:
                tool = ToolSchema(**tool_data)
                
                # Check for duplicate names
                if tool.name in seen_names:
                    raise ValueError(f"Duplicate tool name: {tool.name}")
                
                seen_names.add(tool.name)
                validated_tools.append(tool)
                
            except ValidationError as e:
                logger.error(f"Validation failed for tool at index {idx}: {e}")
                raise ValueError(f"Tool validation failed at index {idx}") from e
        
        self.tools = validated_tools
        self.tool_dict = {tool.name: tool for tool in validated_tools}
        
        logger.info(f"Successfully loaded {len(self.tools)} tools")
        return self.tools, self.tool_dict
    
    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return [tool.name for tool in self.tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get mapping of tool name to description."""
        return {tool.name: tool.desc for tool in self.tools}


class ProfilingDataLoader:
    """
    Loader for profiling data (resource consumption metrics).
    
    Loads profiling CSV and validates entries against schema.
    """
    
    # Column names for profiling CSV
    REQUIRED_COLUMNS = [
        'tool', 'input_size', 'cpu_core', 'cpu_mem_gb',
        'gpu_sm', 'gpu_mem_gb', 'latency_ms'
    ]
    
    # Input size bucket order
    INPUT_SIZE_ORDER = ['small', 'medium', 'large']
    
    def __init__(self, profiling_path: str, tool_registry: Optional[ToolRegistryLoader] = None):
        """
        Initialize the profiling data loader.
        
        Args:
            profiling_path: Path to profiling.csv file
            tool_registry: Optional tool registry loader for validation
        """
        self.profiling_path = Path(profiling_path)
        self.tool_registry = tool_registry
        self.profiling_df: Optional[pd.DataFrame] = None
        self.profiling_data: List[ProfilingSchema] = []
        
    def load(self) -> pd.DataFrame:
        """
        Load and validate profiling data from CSV.
        
        Returns:
            Validated profiling DataFrame
        
        Raises:
            FileNotFoundError: If profiling file doesn't exist
            ValueError: If validation fails or required columns missing
        """
        if not self.profiling_path.exists():
            raise FileNotFoundError(f"Profiling data not found: {self.profiling_path}")
        
        logger.info(f"Loading profiling data from: {self.profiling_path}")
        
        # Load CSV
        df = pd.read_csv(self.profiling_path)
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate each row
        validated_rows = []
        for idx, row in df.iterrows():
            try:
                profiling_entry = ProfilingSchema(
                    tool=row['tool'],
                    input_size=row['input_size'],
                    cpu_core=int(row['cpu_core']),
                    cpu_mem_gb=float(row['cpu_mem_gb']),
                    gpu_sm=int(row['gpu_sm']),
                    gpu_mem_gb=float(row['gpu_mem_gb']),
                    latency_ms=float(row['latency_ms'])
                )
                
                # Check if tool exists in registry (if provided)
                if self.tool_registry and profiling_entry.tool not in self.tool_registry.tool_dict:
                    raise ValueError(
                        f"Tool '{profiling_entry.tool}' in profiling data "
                        f"not found in tool registry"
                    )
                
                validated_rows.append(profiling_entry)
                
            except ValidationError as e:
                logger.error(f"Validation failed for profiling row {idx}: {e}")
                raise ValueError(f"Profiling validation failed at row {idx}") from e
        
        self.profiling_data = validated_rows
        self.profiling_df = df
        
        logger.info(f"Successfully loaded {len(self.profiling_data)} profiling entries")
        return self.profiling_df
    
    def get_profiling_matrix(self, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Convert profiling data to numerical matrix for model training.
        
        Args:
            normalize: Whether to normalize features (z-score normalization)
        
        Returns:
            Tuple of (feature_matrix, metadata) where:
                - feature_matrix: numpy array of shape (N, 6) with features:
                  [input_size_encoded, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb, latency_ms]
                - metadata: Dict with normalization stats and mappings
        """
        if self.profiling_df is None:
            raise RuntimeError("Must call load() before get_profiling_matrix()")
        
        df = self.profiling_df.copy()
        
        # Encode input_size as ordinal (0=small, 1=medium, 2=large)
        input_size_map = {size: idx for idx, size in enumerate(self.INPUT_SIZE_ORDER)}
        df['input_size_encoded'] = df['input_size'].map(input_size_map)
        
        # Extract feature columns
        feature_cols = ['input_size_encoded', 'cpu_core', 'cpu_mem_gb', 
                       'gpu_sm', 'gpu_mem_gb', 'latency_ms']
        
        feature_matrix = df[feature_cols].values.astype(np.float32)
        
        # Compute normalization stats
        metadata = {
            'feature_names': feature_cols,
            'input_size_map': input_size_map,
            'mean': None,
            'std': None
        }
        
        if normalize:
            # Z-score normalization
            mean = feature_matrix.mean(axis=0)
            std = feature_matrix.std(axis=0)
            
            # Avoid division by zero
            std[std == 0] = 1.0
            
            feature_matrix = (feature_matrix - mean) / std
            
            metadata['mean'] = mean.tolist()
            metadata['std'] = std.tolist()
            
            logger.info("Applied z-score normalization to profiling features")
        
        logger.info(f"Generated profiling matrix with shape: {feature_matrix.shape}")
        return feature_matrix, metadata
    
    def get_tool_profiling(self, tool_name: str) -> pd.DataFrame:
        """
        Get all profiling entries for a specific tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            DataFrame with profiling data for the specified tool
        """
        if self.profiling_df is None:
            raise RuntimeError("Must call load() before get_tool_profiling()")
        
        return self.profiling_df[self.profiling_df['tool'] == tool_name]


class ToolDataset:
    """
    Combined dataset that joins tool registry with profiling data.
    
    This creates training samples where each tool is represented by:
    - Semantic text (name + description)
    - Numerical resource profile vector
    """
    
    def __init__(
        self,
        tool_registry_loader: ToolRegistryLoader,
        profiling_loader: ProfilingDataLoader,
        normalize: bool = True
    ):
        """
        Initialize the combined tool dataset.
        
        Args:
            tool_registry_loader: Loaded tool registry
            profiling_loader: Loaded profiling data
            normalize: Whether to normalize profiling features
        """
        self.tool_registry = tool_registry_loader
        self.profiling_loader = profiling_loader
        self.normalize = normalize
        
        # Validate that profiling data references valid tools
        self._validate_join()
        
        # Prepare training samples
        self.samples = self._prepare_samples()
        
    def _validate_join(self):
        """Ensure all profiling entries reference tools in the registry."""
        registry_tools = set(self.tool_registry.tool_dict.keys())
        profiling_tools = set(self.profiling_loader.profiling_df['tool'].unique())
        
        # Check for tools in profiling but not in registry
        missing_tools = profiling_tools - registry_tools
        if missing_tools:
            raise ValueError(
                f"Tools in profiling data but not in registry: {missing_tools}"
            )
        
        logger.info(f"Join validation passed: {len(profiling_tools)} tools have profiling data")
    
    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare training samples combining semantic and resource information.
        
        Returns:
            List of sample dictionaries, each containing:
                - tool_name: str
                - tool_desc: str
                - semantic_text: str (formatted template)
                - input_size: str
                - resource_vector: numpy array
        """
        samples = []
        
        # Get profiling matrix
        profiling_matrix, metadata = self.profiling_loader.get_profiling_matrix(
            normalize=self.normalize
        )
        
        # Iterate through profiling entries
        for idx, profiling_entry in enumerate(self.profiling_loader.profiling_data):
            tool_name = profiling_entry.tool
            tool_schema = self.tool_registry.tool_dict[tool_name]
            
            # Create semantic text template
            semantic_text = f"Tool: {tool_schema.name}\nDescription: {tool_schema.desc}"
            
            sample = {
                'tool_name': tool_name,
                'tool_desc': tool_schema.desc,
                'semantic_text': semantic_text,
                'input_size': profiling_entry.input_size,
                'resource_vector': profiling_matrix[idx],
                'resource_raw': {
                    'cpu_core': profiling_entry.cpu_core,
                    'cpu_mem_gb': profiling_entry.cpu_mem_gb,
                    'gpu_sm': profiling_entry.gpu_sm,
                    'gpu_mem_gb': profiling_entry.gpu_mem_gb,
                    'latency_ms': profiling_entry.latency_ms
                }
            }
            
            samples.append(sample)
        
        logger.info(f"Prepared {len(samples)} training samples")
        return samples
    
    def get_samples(self) -> List[Dict]:
        """Get all prepared training samples."""
        return self.samples
    
    def get_tool_samples(self, tool_name: str) -> List[Dict]:
        """
        Get all samples for a specific tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            List of samples for the specified tool
        """
        return [s for s in self.samples if s['tool_name'] == tool_name]
    
    def to_torch_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Convert resource vectors to PyTorch tensors.
        
        Returns:
            Dictionary with:
                - resource_vectors: Tensor of shape (N, 6)
        """
        resource_vectors = np.stack([s['resource_vector'] for s in self.samples])
        
        return {
            'resource_vectors': torch.from_numpy(resource_vectors).float()
        }
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample by index."""
        return self.samples[idx]


def load_tool_data(
    tool_registry_path: str,
    profiling_path: str,
    normalize: bool = True
) -> ToolDataset:
    """
    Convenience function to load and combine tool registry and profiling data.
    
    Args:
        tool_registry_path: Path to tools.json
        profiling_path: Path to profiling.csv
        normalize: Whether to normalize profiling features
    
    Returns:
        Combined ToolDataset ready for training
    
    Example:
        >>> dataset = load_tool_data('data/tool_registry/tools.json',
        ...                          'data/profiling/profiling.csv')
        >>> print(f"Loaded {len(dataset)} samples")
        >>> sample = dataset[0]
        >>> print(sample['semantic_text'])
        >>> print(sample['resource_vector'])
    """
    # Load tool registry
    registry_loader = ToolRegistryLoader(tool_registry_path)
    registry_loader.load()
    
    # Load profiling data
    profiling_loader = ProfilingDataLoader(profiling_path, tool_registry=registry_loader)
    profiling_loader.load()
    
    # Create combined dataset
    dataset = ToolDataset(registry_loader, profiling_loader, normalize=normalize)
    
    return dataset
