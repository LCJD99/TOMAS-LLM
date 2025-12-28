"""
Pre-training Dataset for Resource Encoder.

Generates (Input, Target) pairs for self-supervised encoder pre-training using
natural language templates of resource configurations.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.offline.text_template import ConfigTextTemplate


class EncoderPretrainDataset(Dataset):
    """
    Dataset for pre-training the resource encoder with self-supervised learning.
    
    Generates (Input, Target) pairs where:
    - Input: Tool ID + Resource Vector (6D)
    - Target: Natural language description (tokenized)
    
    Supports data augmentation via:
    1. Numerical jittering (±5% random noise on resource values)
    2. Linguistic variation (multiple template formats)
    3. Repetition (same config appears multiple times across epochs)
    """
    
    def __init__(
        self,
        tool_registry_path: str = "data/tool_registry/tools.json",
        profiling_data_path: str = "data/profiling/profiling.csv",
        tokenizer_name: str = "Qwen/Qwen2.5-7B",
        max_seq_length: int = 128,
        augmentation_mode: str = "jitter",  # "jitter", "variation", "both", "none"
        jitter_ratio: float = 0.05,  # ±5% noise
        num_augmented_copies: int = 10,  # How many augmented copies per original sample
        use_variation: bool = True,  # Use linguistic variations
        seed: int = 42
    ):
        """
        Initialize the pre-training dataset.
        
        Args:
            tool_registry_path: Path to tools.json
            profiling_data_path: Path to profiling.csv
            tokenizer_name: HuggingFace tokenizer identifier
            max_seq_length: Maximum token sequence length
            augmentation_mode: Type of augmentation ("jitter", "variation", "both", "none")
            jitter_ratio: Noise ratio for numerical jittering
            num_augmented_copies: Number of augmented samples per original
            use_variation: Whether to use template variations
            seed: Random seed for reproducibility
        """
        self.tool_registry_path = Path(tool_registry_path)
        self.profiling_data_path = Path(profiling_data_path)
        self.max_seq_length = max_seq_length
        self.augmentation_mode = augmentation_mode
        self.jitter_ratio = jitter_ratio
        self.num_augmented_copies = num_augmented_copies
        self.use_variation = use_variation
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize template generator
        self.template_gen = ConfigTextTemplate()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            cache_dir="hub"
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.tool_registry = self._load_tool_registry()
        self.profiling_data = self._load_profiling_data()
        
        # Build dataset
        self.samples = self._build_samples()
        
        print(f"Dataset initialized with {len(self.samples)} samples "
              f"(augmentation: {augmentation_mode}, copies: {num_augmented_copies})")
    
    def _load_tool_registry(self) -> List[Dict]:
        """Load and validate tool registry."""
        with open(self.tool_registry_path, 'r') as f:
            registry = json.load(f)
        
        # Build tool name to ID mapping
        self.tool_name_to_id = {
            tool["name"]: idx for idx, tool in enumerate(registry)
        }
        self.tool_id_to_name = {
            idx: tool["name"] for idx, tool in enumerate(registry)
        }
        
        return registry
    
    def _load_profiling_data(self) -> pd.DataFrame:
        """Load and validate profiling data."""
        df = pd.read_csv(self.profiling_data_path)
        
        # Validate required columns (match actual CSV column names)
        required_cols = ["tool", "input_size", "cpu_core", "cpu_mem_gb", 
                        "gpu_sm", "gpu_mem_gb", "latency_ms"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in profiling data: {missing_cols}")
        
        # Rename columns to match expected format
        df = df.rename(columns={
            "tool": "tool_name",
            "cpu_core": "cpu_cores",
            "cpu_mem_gb": "memory_gb",
            "gpu_mem_gb": "gpu_memory_gb"
        })
        
        return df
    
    def _build_samples(self) -> List[Dict]:
        """
        Build the dataset samples with augmentation.
        
        Returns:
            List of sample dictionaries with keys:
                - tool_name: str
                - tool_id: int
                - resource_vector: List[float] (6D)
                - text: str (natural language target)
                - variation_idx: int (for template variation)
        """
        samples = []
        
        # Iterate over profiling data
        for _, row in self.profiling_data.iterrows():
            tool_name = row["tool_name"]
            
            # Skip if tool not in registry
            if tool_name not in self.tool_name_to_id:
                continue
            
            tool_id = self.tool_name_to_id[tool_name]
            
            # Extract resource vector
            base_resource = [
                row["input_size"],  # Keep as string for now
                float(row["cpu_cores"]),
                float(row["memory_gb"]),
                float(row["gpu_sm"]),
                float(row["gpu_memory_gb"]),
                float(row["latency_ms"])
            ]
            
            # Generate augmented copies
            for copy_idx in range(self.num_augmented_copies):
                # Apply jittering if enabled
                if self.augmentation_mode in ["jitter", "both"]:
                    resource_vector = self._apply_jitter(base_resource.copy())
                else:
                    resource_vector = base_resource.copy()
                
                # Select template variation - use placeholder format
                if self.use_variation and self.augmentation_mode in ["variation", "both"]:
                    variation_idx = copy_idx % 3  # Cycle through 3 variations
                else:
                    variation_idx = 0
                
                # Use the new placeholder-based template
                text = self.template_gen.format_with_embedding_placeholder(
                    tool_name, resource_vector, variation_idx
                )
                
                samples.append({
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "resource_vector": resource_vector,
                    "text": text,
                    "variation_idx": variation_idx
                })
        
        return samples
    
    def _apply_jitter(self, resource_vector: List) -> List:
        """
        Apply random jittering to numerical resource values.
        
        Args:
            resource_vector: [input_size, cpu_cores, memory_gb, gpu_sm, gpu_memory_gb, latency_ms]
        
        Returns:
            Jittered resource vector with noise added to numerical values.
        """
        jittered = []
        
        for idx, value in enumerate(resource_vector):
            if idx == 0:
                # input_size is categorical, no jitter
                jittered.append(value)
            else:
                # Add ±jitter_ratio noise to numerical values
                noise = np.random.uniform(-self.jitter_ratio, self.jitter_ratio)
                jittered_value = value * (1 + noise)
                
                # Ensure positive values
                jittered_value = max(jittered_value, 1.0)
                
                # Round to reasonable precision
                if idx in [1]:  # cpu_cores
                    jittered_value = int(round(jittered_value))
                elif idx in [2, 4]:  # memory values in GB
                    jittered_value = round(jittered_value, 1)
                elif idx in [3]:  # gpu_sm
                    jittered_value = int(round(jittered_value))
                else:  # latency
                    jittered_value = round(jittered_value, 0)
                
                jittered.append(jittered_value)
        
        return jittered
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary with:
                - tool_id: Tensor (scalar)
                - resource_vector: Tensor (6D)
                - input_ids: Tensor (token IDs for target text)
                - attention_mask: Tensor (attention mask)
                - labels: Tensor (same as input_ids, for causal LM)
        """
        sample = self.samples[idx]
        
        # Prepare resource vector (convert input_size to numeric if needed)
        resource_vector = sample["resource_vector"].copy()
        if isinstance(resource_vector[0], str):
            size_map = {"small": 0, "medium": 1, "large": 2}
            resource_vector[0] = size_map.get(resource_vector[0], 0)
        
        resource_tensor = torch.tensor(resource_vector, dtype=torch.float32)
        
        # Tokenize target text
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Find the position of [TOOL_RESOURCE] placeholder
        # Tokenize the placeholder to find its token IDs
        placeholder_encoding = self.tokenizer(
            "[TOOL_RESOURCE]",
            add_special_tokens=False
        )
        placeholder_ids = placeholder_encoding["input_ids"]
        
        # Find where the placeholder appears in the tokenized sequence
        input_ids = encoding["input_ids"].squeeze(0)
        placeholder_pos = -1
        
        # Search for the placeholder token sequence
        for i in range(len(input_ids) - len(placeholder_ids) + 1):
            if all(input_ids[i + j] == placeholder_ids[j] for j in range(len(placeholder_ids))):
                placeholder_pos = i
                break
        
        # If placeholder not found (truncated), use position 10 as fallback
        if placeholder_pos == -1:
            placeholder_pos = 10
        
        return {
            "tool_id": torch.tensor(sample["tool_id"], dtype=torch.long),
            "resource_vector": resource_tensor,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),  # For causal LM loss
            "placeholder_pos": torch.tensor(placeholder_pos, dtype=torch.long)
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "unique_configs": len(self.profiling_data),
            "num_tools": len(self.tool_name_to_id),
            "augmentation_factor": self.num_augmented_copies,
            "augmentation_mode": self.augmentation_mode,
            "tool_distribution": {}
        }
        
        # Count samples per tool
        for sample in self.samples:
            tool_name = sample["tool_name"]
            stats["tool_distribution"][tool_name] = \
                stats["tool_distribution"].get(tool_name, 0) + 1
        
        return stats


def test_pretrain_dataset():
    """Test the pre-training dataset."""
    # Create dataset
    dataset = EncoderPretrainDataset(
        augmentation_mode="both",
        num_augmented_copies=5,
        use_variation=True
    )
    
    # Print statistics
    stats = dataset.get_statistics()
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Unique configs: {stats['unique_configs']}")
    print(f"Number of tools: {stats['num_tools']}")
    print(f"Augmentation factor: {stats['augmentation_factor']}")
    print(f"Augmentation mode: {stats['augmentation_mode']}")
    print("\nTool distribution:")
    for tool, count in stats['tool_distribution'].items():
        print(f"  {tool}: {count} samples")
    
    # Sample a few items
    print("\n=== Sample Items ===")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        sample_info = dataset.samples[i]
        
        print(f"\nSample {i}:")
        print(f"  Tool: {sample_info['tool_name']} (ID: {item['tool_id'].item()})")
        print(f"  Resource Vector: {item['resource_vector'].tolist()}")
        print(f"  Text: {sample_info['text']}")
        print(f"  Token length: {item['attention_mask'].sum().item()}")
        
        # Decode tokens
        decoded = dataset.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        print(f"  Decoded: {decoded}")


if __name__ == "__main__":
    test_pretrain_dataset()
