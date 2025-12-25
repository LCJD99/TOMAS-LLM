"""
Configuration Lookup Table Builder for TOMAS-LLM Phase 0.

Builds mapping tables that enable the "Weights as Memory" architecture:
1. Config ID → Resource Configuration (for final output lookup)
2. Tool ID → Config ID List (for hierarchical decoder masking)

This module processes static tool registry and profiling data to create
the lookup infrastructure needed for training and inference.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ConfigLookupBuilder:
    """
    Builds configuration lookup tables for hierarchical decoding.
    
    Processes tool registry + profiling data to create:
    - config_lookup.json: {config_id: {tool_name, cpu, gpu, ...}}
    - tool_mask_map.json: {tool_id: [list_of_config_ids]}
    
    These tables enable:
    1. Final resource plan retrieval from Config ID
    2. Masking invalid Config IDs during inference
    """
    
    def __init__(self):
        """Initialize lookup builder."""
        self.tool_registry: List[Dict[str, str]] = []
        self.profiling_data: pd.DataFrame = pd.DataFrame()
        
        # Generated lookup tables
        self.config_lookup: Dict[int, Dict[str, Any]] = {}
        self.tool_mask_map: Dict[int, List[int]] = {}
        
        # Statistics
        self.total_configs = 0
        self.configs_per_tool: Dict[str, int] = {}
    
    def load_tool_registry(self, registry_path: str) -> List[Dict[str, str]]:
        """
        Load tool registry from JSON file.
        
        Args:
            registry_path: Path to tools.json file
            
        Returns:
            List of tool dictionaries with 'name' and 'desc' fields
        """
        registry_file = Path(registry_path)
        
        if not registry_file.exists():
            raise FileNotFoundError(f"Tool registry not found: {registry_path}")
        
        logger.info(f"Loading tool registry from: {registry_path}")
        
        with open(registry_file, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        if not isinstance(tools, list):
            raise ValueError("Tool registry must be a JSON array")
        
        # Validate tool structure
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict) or 'name' not in tool or 'desc' not in tool:
                raise ValueError(f"Invalid tool structure at index {i}: missing 'name' or 'desc'")
        
        self.tool_registry = tools
        logger.info(f"Loaded {len(tools)} tools: {[t['name'] for t in tools]}")
        
        return tools
    
    def load_profiling_data(self, profiling_path: str) -> pd.DataFrame:
        """
        Load profiling data from CSV file.
        
        Expected columns:
        - tool_name: Name of the tool
        - input_size_encoded: Encoded input size (0/1/2 for small/medium/large)
        - cpu_core: CPU cores required
        - cpu_mem_gb: CPU memory in GB
        - gpu_sm: GPU streaming multiprocessors (percentage 0-100)
        - gpu_mem_gb: GPU memory in GB
        - latency_ms: Latency in milliseconds
        
        Args:
            profiling_path: Path to profiling.csv file
            
        Returns:
            Loaded profiling DataFrame
        """
        profiling_file = Path(profiling_path)
        
        if not profiling_file.exists():
            raise FileNotFoundError(f"Profiling data not found: {profiling_path}")
        
        logger.info(f"Loading profiling data from: {profiling_path}")
        
        df = pd.read_csv(profiling_file)
        
        # Validate required columns
        required_columns = [
            'tool_name', 'input_size_encoded', 'cpu_core', 
            'cpu_mem_gb', 'gpu_sm', 'gpu_mem_gb', 'latency_ms'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Basic data validation
        if df.empty:
            raise ValueError("Profiling data is empty")
        
        # Log statistics
        total_rows = len(df)
        unique_tools = df['tool_name'].nunique()
        
        logger.info(f"Loaded profiling data: {total_rows} rows, {unique_tools} unique tools")
        
        # Log tool distribution
        tool_counts = df['tool_name'].value_counts()
        logger.info("Tool distribution:")
        for tool_name, count in tool_counts.items():
            logger.info(f"  {tool_name}: {count} configurations")
            self.configs_per_tool[tool_name] = count
        
        self.profiling_data = df
        return df
    
    def build_config_lookup(self) -> Dict[int, Dict[str, Any]]:
        """
        Build configuration lookup table.
        
        Maps Config ID (0-1700) to complete resource configuration including
        tool information and resource requirements.
        
        Returns:
            Dictionary mapping config_id to resource configuration
        """
        if self.tool_registry == [] or self.profiling_data.empty:
            raise RuntimeError("Must load tool registry and profiling data first")
        
        logger.info("Building configuration lookup table...")
        
        config_lookup = {}
        config_id = 0
        
        # Create tool name to ID mapping
        tool_name_to_id = {tool['name']: i for i, tool in enumerate(self.tool_registry)}
        
        # Process each profiling entry
        for _, row in self.profiling_data.iterrows():
            tool_name = row['tool_name']
            
            # Validate tool exists in registry
            if tool_name not in tool_name_to_id:
                logger.warning(f"Tool '{tool_name}' in profiling data not found in registry, skipping")
                continue
            
            tool_id = tool_name_to_id[tool_name]
            
            # Find corresponding tool info
            tool_info = next(tool for tool in self.tool_registry if tool['name'] == tool_name)
            
            # Build complete configuration
            config = {
                # Tool information
                'tool_id': tool_id,
                'tool_name': tool_name,
                'tool_description': tool_info['desc'],
                
                # Resource configuration
                'input_size_encoded': int(row['input_size_encoded']),
                'cpu_core': float(row['cpu_core']),
                'cpu_mem_gb': float(row['cpu_mem_gb']),
                'gpu_sm': float(row['gpu_sm']),
                'gpu_mem_gb': float(row['gpu_mem_gb']),
                'latency_ms': float(row['latency_ms']),
                
                # Additional metadata
                'config_id': config_id
            }
            
            config_lookup[config_id] = config
            config_id += 1
        
        self.total_configs = config_id
        self.config_lookup = config_lookup
        
        logger.info(f"Built configuration lookup with {self.total_configs} configurations")
        return config_lookup
    
    def build_tool_mask_map(self) -> Dict[int, List[int]]:
        """
        Build tool masking map for hierarchical decoder.
        
        Maps Tool ID to list of valid Config IDs belonging to that tool.
        This enables masking during inference to ensure only valid
        tool-config combinations are selected.
        
        Returns:
            Dictionary mapping tool_id to list of valid config_ids
        """
        if not self.config_lookup:
            raise RuntimeError("Must build config lookup first")
        
        logger.info("Building tool masking map...")
        
        tool_mask_map = {}
        
        # Group config IDs by tool ID
        for config_id, config in self.config_lookup.items():
            tool_id = config['tool_id']
            
            if tool_id not in tool_mask_map:
                tool_mask_map[tool_id] = []
            
            tool_mask_map[tool_id].append(config_id)
        
        # Sort config ID lists for consistency
        for tool_id in tool_mask_map:
            tool_mask_map[tool_id] = sorted(tool_mask_map[tool_id])
        
        self.tool_mask_map = tool_mask_map
        
        # Log statistics
        logger.info("Tool masking map statistics:")
        for tool_id, config_ids in tool_mask_map.items():
            tool_name = self.config_lookup[config_ids[0]]['tool_name']  # Get tool name from first config
            logger.info(f"  Tool {tool_id} ({tool_name}): {len(config_ids)} configurations")
        
        return tool_mask_map
    
    def save_lookup_tables(
        self,
        config_lookup_path: str = "assets/config_lookup.json",
        tool_mask_path: str = "assets/tool_mask_map.json"
    ) -> Tuple[str, str]:
        """
        Save lookup tables to JSON files.
        
        Args:
            config_lookup_path: Path to save config lookup table
            tool_mask_path: Path to save tool masking map
            
        Returns:
            Tuple of (config_lookup_path, tool_mask_path)
        """
        if not self.config_lookup or not self.tool_mask_map:
            raise RuntimeError("Must build lookup tables first")
        
        # Create output directory
        config_file = Path(config_lookup_path)
        mask_file = Path(tool_mask_path)
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config lookup table
        logger.info(f"Saving config lookup table to: {config_lookup_path}")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config_lookup, f, indent=2, ensure_ascii=False)
        
        # Save tool masking map
        logger.info(f"Saving tool masking map to: {tool_mask_path}")
        with open(mask_file, 'w', encoding='utf-8') as f:
            json.dump(self.tool_mask_map, f, indent=2, ensure_ascii=False)
        
        logger.info("Successfully saved all lookup tables")
        return str(config_file), str(mask_file)
    
    def validate_consistency(self) -> bool:
        """
        Validate consistency between lookup tables and source data.
        
        Returns:
            True if all validations pass
        """
        if not self.config_lookup or not self.tool_mask_map:
            logger.error("Cannot validate: lookup tables not built")
            return False
        
        logger.info("Validating lookup table consistency...")
        
        try:
            # Check total configuration count
            expected_total = len(self.profiling_data)
            actual_total = len(self.config_lookup)
            
            if expected_total != actual_total:
                logger.error(f"Config count mismatch: expected {expected_total}, got {actual_total}")
                return False
            
            # Check config ID range
            config_ids = set(self.config_lookup.keys())
            expected_ids = set(range(self.total_configs))
            
            if config_ids != expected_ids:
                logger.error(f"Config ID range mismatch: missing/extra IDs")
                return False
            
            # Check tool masking map completeness
            all_mask_configs = set()
            for tool_id, config_list in self.tool_mask_map.items():
                all_mask_configs.update(config_list)
            
            if all_mask_configs != config_ids:
                logger.error("Tool masking map doesn't cover all configurations")
                return False
            
            # Check tool ID consistency
            tool_registry_count = len(self.tool_registry)
            mask_tool_ids = set(self.tool_mask_map.keys())
            expected_tool_ids = set(range(tool_registry_count))
            
            if mask_tool_ids != expected_tool_ids:
                logger.error(f"Tool ID mismatch in masking map")
                return False
            
            logger.info("All validation checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the lookup tables.
        
        Returns:
            Dictionary with statistics and metadata
        """
        if not self.config_lookup:
            return {"error": "Lookup tables not built"}
        
        # Tool statistics
        tool_stats = {}
        for tool_id, config_list in self.tool_mask_map.items():
            tool_name = self.config_lookup[config_list[0]]['tool_name']
            tool_stats[tool_name] = {
                'tool_id': tool_id,
                'config_count': len(config_list),
                'config_id_range': [min(config_list), max(config_list)]
            }
        
        # Resource statistics (optional analysis)
        resource_stats = {
            'total_configs': self.total_configs,
            'tools_count': len(self.tool_registry),
            'avg_configs_per_tool': self.total_configs / len(self.tool_registry)
        }
        
        return {
            'summary': resource_stats,
            'tool_breakdown': tool_stats,
            'validation_passed': self.validate_consistency()
        }
    
    @classmethod
    def build_from_data(
        cls,
        tool_registry_path: str,
        profiling_data_path: str,
        output_dir: str = "assets"
    ) -> 'ConfigLookupBuilder':
        """
        Convenience method to build all lookup tables from data files.
        
        Args:
            tool_registry_path: Path to tools.json
            profiling_data_path: Path to profiling.csv
            output_dir: Directory to save output files
            
        Returns:
            Configured ConfigLookupBuilder instance
        """
        builder = cls()
        
        # Load data
        builder.load_tool_registry(tool_registry_path)
        builder.load_profiling_data(profiling_data_path)
        
        # Build lookup tables
        builder.build_config_lookup()
        builder.build_tool_mask_map()
        
        # Save tables
        config_path = f"{output_dir}/config_lookup.json"
        mask_path = f"{output_dir}/tool_mask_map.json"
        builder.save_lookup_tables(config_path, mask_path)
        
        # Validate and log statistics
        stats = builder.get_statistics()
        logger.info("Lookup table generation complete:")
        logger.info(f"  Total configurations: {stats['summary']['total_configs']}")
        logger.info(f"  Tools: {stats['summary']['tools_count']}")
        logger.info(f"  Validation: {'PASSED' if stats['validation_passed'] else 'FAILED'}")
        
        return builder