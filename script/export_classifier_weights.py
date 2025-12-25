"""
Export Classifier Weights from Pre-trained Encoder.

After the encoder is pre-trained, this script:
1. Loads the trained encoder
2. Processes all 1701 (Tool, Resource) configurations
3. Generates [1701, 3584] weight matrix
4. Saves as config_weights.pt for HierarchicalDecoder initialization
5. Builds lookup tables (config_lookup.json, tool_mask_map.json)

This weight matrix will be used to initialize the config_head in the main model.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
from tqdm import tqdm

from src.offline.pretrain_encoder import ResourceEncoderForPretraining
from src.offline.lookup_builder import ConfigLookupBuilder


class WeightExporter:
    """Exports pre-trained encoder weights as classifier weight matrix."""
    
    def __init__(
        self,
        encoder: ResourceEncoderForPretraining,
        device: str = "cuda"
    ):
        """
        Initialize the weight exporter.
        
        Args:
            encoder: Pre-trained ResourceEncoderForPretraining
            device: Device to run inference on
        """
        self.encoder = encoder.to(device)
        self.encoder.eval()  # Set to evaluation mode
        self.device = device
    
    def load_configurations(
        self,
        tool_registry_path: str,
        profiling_data_path: str
    ) -> tuple:
        """
        Load all tool configurations.
        
        Args:
            tool_registry_path: Path to tools.json
            profiling_data_path: Path to profiling.csv
        
        Returns:
            Tuple of (config_list, tool_name_to_id)
            config_list: List of dicts with keys: config_id, tool_name, tool_id, resource_vector
        """
        # Load tool registry
        with open(tool_registry_path, 'r') as f:
            tools = json.load(f)
        
        tool_name_to_id = {tool["name"]: idx for idx, tool in enumerate(tools)}
        
        # Load profiling data
        df = pd.read_csv(profiling_data_path)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            "tool": "tool_name",
            "cpu_core": "cpu_cores",
            "cpu_mem_gb": "memory_gb",
            "gpu_mem_gb": "gpu_memory_gb"
        })
        
        # Build configuration list
        config_list = []
        config_id = 0
        
        for _, row in df.iterrows():
            tool_name = row["tool_name"]
            
            # Skip if tool not in registry
            if tool_name not in tool_name_to_id:
                print(f"Warning: Tool '{tool_name}' not found in registry, skipping...")
                continue
            
            tool_id = tool_name_to_id[tool_name]
            
            # Build resource vector
            # Convert input_size to numeric if it's a string
            input_size = row["input_size"]
            if isinstance(input_size, str):
                size_map = {"small": 0, "medium": 1, "large": 2}
                input_size_numeric = size_map.get(input_size, 0)
            else:
                input_size_numeric = float(input_size)
            
            resource_vector = [
                input_size_numeric,
                float(row["cpu_cores"]),
                float(row["memory_gb"]),
                float(row["gpu_sm"]),
                float(row["gpu_memory_gb"]),
                float(row["latency_ms"])
            ]
            
            config_list.append({
                "config_id": config_id,
                "tool_name": tool_name,
                "tool_id": tool_id,
                "resource_vector": resource_vector,
                "input_size": row["input_size"],  # Keep original for lookup table
                "cpu_cores": int(row["cpu_cores"]),
                "memory_gb": float(row["memory_gb"]),
                "gpu_sm": int(row["gpu_sm"]),
                "gpu_memory_gb": float(row["gpu_memory_gb"]),
                "latency_ms": float(row["latency_ms"])
            })
            
            config_id += 1
        
        return config_list, tool_name_to_id
    
    def generate_weight_matrix(
        self,
        config_list: List[Dict],
        batch_size: int = 64
    ) -> torch.Tensor:
        """
        Generate weight matrix from all configurations.
        
        Args:
            config_list: List of configuration dictionaries
            batch_size: Batch size for inference
        
        Returns:
            Weight matrix of shape [num_configs, hidden_dim]
        """
        num_configs = len(config_list)
        hidden_dim = self.encoder.llm_hidden_dim
        
        # Allocate weight matrix
        weight_matrix = torch.zeros(num_configs, hidden_dim, dtype=torch.float32)
        
        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, num_configs, batch_size), desc="Generating weights"):
                batch_configs = config_list[i:i+batch_size]
                
                # Prepare batch
                tool_ids = torch.tensor(
                    [cfg["tool_id"] for cfg in batch_configs],
                    dtype=torch.long,
                    device=self.device
                )
                
                resource_vectors = torch.tensor(
                    [cfg["resource_vector"] for cfg in batch_configs],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Generate embeddings
                embeddings = self.encoder(tool_ids, resource_vectors)  # [batch, hidden]
                
                # Store in weight matrix
                weight_matrix[i:i+len(batch_configs)] = embeddings.cpu()
        
        return weight_matrix
    
    def build_lookup_tables(
        self,
        config_list: List[Dict]
    ) -> tuple:
        """
        Build lookup tables for config ID mapping.
        
        Args:
            config_list: List of configuration dictionaries
        
        Returns:
            Tuple of (config_lookup, tool_mask_map)
        """
        # Build config_lookup: config_id -> full config info
        config_lookup = {}
        for cfg in config_list:
            config_lookup[cfg["config_id"]] = {
                "tool_name": cfg["tool_name"],
                "input_size": cfg["input_size"],
                "cpu_cores": cfg["cpu_cores"],
                "memory_gb": cfg["memory_gb"],
                "gpu_sm": cfg["gpu_sm"],
                "gpu_memory_gb": cfg["gpu_memory_gb"],
                "latency_ms": cfg["latency_ms"]
            }
        
        # Build tool_mask_map: tool_name -> list of valid config_ids
        tool_mask_map = {}
        for cfg in config_list:
            tool_name = cfg["tool_name"]
            if tool_name not in tool_mask_map:
                tool_mask_map[tool_name] = []
            tool_mask_map[tool_name].append(cfg["config_id"])
        
        # Sort config IDs for each tool
        for tool_name in tool_mask_map:
            tool_mask_map[tool_name].sort()
        
        return config_lookup, tool_mask_map
    
    def save_assets(
        self,
        weight_matrix: torch.Tensor,
        config_lookup: Dict,
        tool_mask_map: Dict,
        output_dir: str
    ):
        """
        Save all exported assets.
        
        Args:
            weight_matrix: [num_configs, hidden_dim] tensor
            config_lookup: Config ID to config info mapping
            tool_mask_map: Tool name to valid config IDs mapping
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save weight matrix
        weight_path = os.path.join(output_dir, "config_weights.pt")
        torch.save({
            "weights": weight_matrix,
            "num_configs": weight_matrix.size(0),
            "hidden_dim": weight_matrix.size(1),
            "description": "Pre-trained encoder weights for config classification head"
        }, weight_path)
        print(f"  ✓ Saved weight matrix to {weight_path}")
        print(f"    Shape: {list(weight_matrix.shape)}")
        
        # Save config lookup
        lookup_path = os.path.join(output_dir, "config_lookup.json")
        with open(lookup_path, 'w') as f:
            json.dump(config_lookup, f, indent=2)
        print(f"  ✓ Saved config lookup to {lookup_path}")
        print(f"    Entries: {len(config_lookup)}")
        
        # Save tool mask map
        mask_path = os.path.join(output_dir, "tool_mask_map.json")
        with open(mask_path, 'w') as f:
            json.dump(tool_mask_map, f, indent=2)
        print(f"  ✓ Saved tool mask map to {mask_path}")
        print(f"    Tools: {len(tool_mask_map)}")
        
        # Print sample from each
        print("\n  Sample Entries:")
        print(f"    Config 0: {config_lookup[0]}")
        first_tool = list(tool_mask_map.keys())[0]
        print(f"    Tool '{first_tool}' has {len(tool_mask_map[first_tool])} configs: {tool_mask_map[first_tool][:5]}...")


def main():
    parser = argparse.ArgumentParser(description="Export classifier weights from pre-trained encoder")
    
    # Input arguments
    parser.add_argument("--encoder_checkpoint", type=str, required=True,
                        help="Path to pre-trained encoder checkpoint (.pt file)")
    parser.add_argument("--tool_registry", type=str, default="data/tool_registry/tools.json",
                        help="Path to tool registry JSON")
    parser.add_argument("--profiling_data", type=str, default="data/profiling/profiling.csv",
                        help="Path to profiling CSV")
    
    # Model arguments
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B",
                        help="LLM model name (must match pre-training)")
    parser.add_argument("--cache_dir", type=str, default="hub",
                        help="Cache directory for models")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="assets",
                        help="Output directory for exported assets")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for weight generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference")
    
    args = parser.parse_args()
    
    print("="*80)
    print("TOMAS-LLM Weight Exporter")
    print("="*80)
    print(f"Encoder Checkpoint: {args.encoder_checkpoint}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # 1. Load pre-trained encoder
    print("\n[1/4] Loading pre-trained encoder...")
    
    # Initialize encoder architecture
    encoder = ResourceEncoderForPretraining(
        llm_model_name=args.llm_model,
        freeze_semantic=True,
        cache_dir=args.cache_dir
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.encoder_checkpoint, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    
    print(f"  ✓ Loaded encoder from {args.encoder_checkpoint}")
    if "epoch" in checkpoint:
        print(f"    Trained for {checkpoint['epoch']+1} epochs")
    if "metrics" in checkpoint and "train_loss" in checkpoint["metrics"]:
        print(f"    Final loss: {checkpoint['metrics']['train_loss']:.4f}")
    
    # 2. Load configurations
    print("\n[2/4] Loading configurations...")
    exporter = WeightExporter(encoder, device=args.device)
    config_list, tool_name_to_id = exporter.load_configurations(
        args.tool_registry,
        args.profiling_data
    )
    
    print(f"  ✓ Loaded {len(config_list)} configurations")
    print(f"  ✓ Tools: {len(tool_name_to_id)}")
    
    # Verify expected count
    expected_configs = 1701
    if len(config_list) != expected_configs:
        print(f"  ⚠️  Warning: Expected {expected_configs} configs, got {len(config_list)}")
    
    # 3. Generate weight matrix
    print("\n[3/4] Generating weight matrix...")
    weight_matrix = exporter.generate_weight_matrix(
        config_list,
        batch_size=args.batch_size
    )
    
    print(f"  ✓ Generated weight matrix: {list(weight_matrix.shape)}")
    print(f"    dtype: {weight_matrix.dtype}")
    print(f"    device: cpu (for saving)")
    
    # 4. Build lookup tables
    print("\n[4/4] Building lookup tables...")
    config_lookup, tool_mask_map = exporter.build_lookup_tables(config_list)
    
    print(f"  ✓ Built config_lookup with {len(config_lookup)} entries")
    print(f"  ✓ Built tool_mask_map for {len(tool_mask_map)} tools")
    
    # Print tool distribution
    print("\n  Tool Distribution:")
    for tool_name, config_ids in sorted(tool_mask_map.items()):
        print(f"    {tool_name}: {len(config_ids)} configs")
    
    # Save all assets
    print("\n" + "="*80)
    print("Saving Assets")
    print("="*80)
    exporter.save_assets(weight_matrix, config_lookup, tool_mask_map, args.output_dir)
    
    print("\n" + "="*80)
    print("Export Complete!")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  1. {os.path.join(args.output_dir, 'config_weights.pt')}")
    print(f"  2. {os.path.join(args.output_dir, 'config_lookup.json')}")
    print(f"  3. {os.path.join(args.output_dir, 'tool_mask_map.json')}")
    print("\nThese files are ready to be used in the main TOMAS-LLM model:")
    print("  - config_weights.pt → Initialize HierarchicalDecoder.config_head.weight")
    print("  - config_lookup.json → Map config_id to resource values at inference")
    print("  - tool_mask_map.json → Mask invalid config_ids during hierarchical decoding")


if __name__ == "__main__":
    main()
