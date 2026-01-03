import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

INSTRUCTION = "Act as a Tool Resource Analysis Specialist." \
    "It have encoded a tool's specifications and its runtime " \
    "performance under specific resource constraints into the following embedding: "

class ConfigTextTemplate:
    """
    Generates natural language descriptions from resource configuration data.
    
    Example output:
        "Tool super_resolution configuration: input size small, requires 4 CPU cores, 
         4 GB memory, 50 GPU SMs, 2 GB GPU memory, and latency is 5800 ms."
    """
    
    def __init__(self, use_variations: bool = True):
        """
        Initialize the template generator.
        
        Args:
            use_variations: Whether to use multiple template variations
        """
        self.use_variations = use_variations
        self.variation_index = 0
    
    def format_qwen(
        self,
        tool_name: str,
        resource_vector: Union[List[float], torch.Tensor, Dict[str, float]],
        variation_idx: int = 0
    ) -> Dict[str, str]:
        """
        Docstring for format_qwen
        
        :param self: Description
        :param tool_name: Description
        :type tool_name: str
        :param resource_vector: Description
        :type resource_vector: Union[List[float], torch.Tensor, Dict[str, float]]
        :param variation_idx: Description
        :type variation_idx: int
        :return: Description
        :rtype: Dict[str, str]
        """
        if isinstance(resource_vector, dict):
            input_size = resource_vector["input_size"]
            cpu_cores = resource_vector["cpu_cores"]
            memory_gb = resource_vector["memory_gb"]
            gpu_sm = resource_vector["gpu_sm"]
            gpu_memory_gb = resource_vector["gpu_memory_gb"]
            latency_ms = resource_vector["latency_ms"]
        else:
            if isinstance(resource_vector, torch.Tensor):
                resource_vector = resource_vector.tolist()
            input_size, cpu_cores, memory_gb, gpu_sm, gpu_memory_gb, latency_ms = resource_vector
        
        # Format input size
        if isinstance(input_size, str):
            input_size_str = input_size
        else:
            size_map = {0: "small", 1: "medium", 2: "large"}
            input_size_str = size_map.get(int(input_size), "small")

        queries = [
            "Please describe the tool's resource allocation and performance.",
            "What are the resource requirements and execution characteristics?",
            "Explain the current configuration and latency metrics.",
        ]
        
        # Assistant response variations
        responses = [
            # Variation 0: Detailed format
            (
                f"The tool {tool_name} is configured with input size {input_size_str}, "
                f"requiring {int(cpu_cores)} CPU cores, {int(memory_gb)} GB memory, "
                f"{int(gpu_sm)} GPU SMs, {int(gpu_memory_gb)} GB GPU memory, "
                f"and achieves {int(latency_ms)} ms execution latency."
            ),
            # Variation 1: Resource-first format
            (
                f"This configuration allocates {int(cpu_cores)} CPU cores and {int(memory_gb)} GB RAM, "
                f"with {int(gpu_sm)} GPU SMs and {int(gpu_memory_gb)} GB GPU memory "
                f"for {tool_name} to process {input_size_str} inputs, "
                f"resulting in {int(latency_ms)} ms latency."
            ),
            # Variation 2: Performance-focused format
            (
                f"{tool_name} processes {input_size_str} data with {int(latency_ms)} ms latency, "
                f"utilizing {int(cpu_cores)} cores, {int(memory_gb)} GB memory, "
                f"{int(gpu_sm)} SMs and {int(gpu_memory_gb)} GB GPU memory."
            ),
        ]
        
        user_query = queries[variation_idx % len(queries)]
        assistant_response = responses[variation_idx % len(responses)]

        return {
            "prefix": f"{INSTRUCTION}",
            "suffix": f"Question:{user_query} Answer:{assistant_response}",
            "question": f"Question:{user_query} Answer:",
            "answer": assistant_response
        }

    
    def format_qwen_instruct(
        self,
        tool_name: str,
        resource_vector: Union[List[float], torch.Tensor, Dict[str, float]],
        variation_idx: int = 0
    ) -> Dict[str, str]:
        """
        Generate Qwen2.5-Instruct style conversation with <tool_feat> placeholder.
        
        Format:
            User: "<tool_feat> Please describe the tool's resource allocation and performance."
            Assistant: "The tool {tool_name} is configured with input size {size}, ..."
        
        Args:
            tool_name: Name of the tool
            resource_vector: Resource configuration
            variation_idx: Index to select template variation (0-2)
        
        Returns:
            Dictionary with 'user_query', 'assistant_response', and 'full_text' keys.
        """
        # Parse resource vector (same as format_single)
        if isinstance(resource_vector, dict):
            input_size = resource_vector["input_size"]
            cpu_cores = resource_vector["cpu_cores"]
            memory_gb = resource_vector["memory_gb"]
            gpu_sm = resource_vector["gpu_sm"]
            gpu_memory_gb = resource_vector["gpu_memory_gb"]
            latency_ms = resource_vector["latency_ms"]
        else:
            if isinstance(resource_vector, torch.Tensor):
                resource_vector = resource_vector.tolist()
            input_size, cpu_cores, memory_gb, gpu_sm, gpu_memory_gb, latency_ms = resource_vector
        
        # Format input size
        if isinstance(input_size, str):
            input_size_str = input_size
        else:
            size_map = {0: "small", 1: "medium", 2: "large"}
            input_size_str = size_map.get(int(input_size), "small")
        
        queries = [
            "Please describe the tool's resource allocation and performance.",
            "What are the resource requirements and execution characteristics?",
            "Explain the current configuration and latency metrics.",
        ]
        
        # Assistant response variations
        responses = [
            # Variation 0: Detailed format
            (
                f"The tool {tool_name} is configured with input size {input_size_str}, "
                f"requiring {int(cpu_cores)} CPU cores, {int(memory_gb)} GB memory, "
                f"{int(gpu_sm)} GPU SMs, {int(gpu_memory_gb)} GB GPU memory, "
                f"and achieves {int(latency_ms)} ms execution latency."
            ),
            # Variation 1: Resource-first format
            (
                f"This configuration allocates {int(cpu_cores)} CPU cores and {int(memory_gb)} GB RAM, "
                f"with {int(gpu_sm)} GPU SMs and {int(gpu_memory_gb)} GB GPU memory "
                f"for {tool_name} to process {input_size_str} inputs, "
                f"resulting in {int(latency_ms)} ms latency."
            ),
            # Variation 2: Performance-focused format
            (
                f"{tool_name} processes {input_size_str} data with {int(latency_ms)} ms latency, "
                f"utilizing {int(cpu_cores)} cores, {int(memory_gb)} GB memory, "
                f"{int(gpu_sm)} SMs and {int(gpu_memory_gb)} GB GPU memory."
            ),
        ]
        
        user_query = queries[variation_idx % len(queries)]
        assistant_response = responses[variation_idx % len(responses)]
        
        full_text = (
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
        )

        
        return {
            "user_query": user_query,
            "assistant_response": assistant_response,
            "full_text": full_text
        }
    

def test_template_generator():
    """Test the template generator with sample data."""
    generator = ConfigTextTemplate()
    
    # Test with list format
    tool_name = "super_resolution"
    resource_vector = ["small", 4, 4, 50, 2, 5800]
    
    text = generator.format_single(tool_name, resource_vector)
    print("List Format Test:")
    print(text)
    print()
    
    # Test with dict format
    resource_dict = {
        "input_size": "medium",
        "cpu_cores": 8,
        "memory_gb": 8,
        "gpu_sm": 80,
        "gpu_memory_gb": 4,
        "latency_ms": 3200
    }
    
    text = generator.format_single("object_detection", resource_dict)
    print("Dict Format Test:")
    print(text)
    print()
    
    # Test batch format
    batch = [
        {"tool_name": "super_resolution", "resource_vector": ["small", 4, 4, 50, 2, 5800]},
        {"tool_name": "object_detection", "resource_vector": resource_dict}
    ]
    
    texts = generator.format_batch(batch)
    print("Batch Format Test:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
    print()
    
    # Test variations
    print("Variation Test:")
    for i in range(3):
        text = generator.format_with_variation(tool_name, resource_vector, variation_idx=i)
        print(f"Variation {i}: {text}")


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
        tokenizer_name: str = "Qwen2.5-7B-Instruct",
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
                
                # Select template variation - use Qwen instruct format
                if self.use_variation and self.augmentation_mode in ["variation", "both"]:
                    variation_idx = copy_idx % 3  # Cycle through 3 variations
                else:
                    variation_idx = 0
                
                # Use the new Qwen instruct format (returns dict)
                text = self.template_gen.format_qwen(
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
        Get a single training sample for Qwen-Instruct format.
        
        Returns:
            Dictionary with:
                - tool_id: Tensor (scalar)
                - resource_vector: Tensor (6D)
                - prefix_input_ids: Tensor (tokens before <tool_feat>)
                - prefix_attention_mask: Tensor
                - suffix_input_ids: Tensor (tokens after <tool_feat>)
                - suffix_attention_mask: Tensor
                - assistant_start_pos: int (position where assistant response starts)
                - prefix_labels: Tensor (-100 for all)
                - suffix_labels: Tensor (-100 before assistant, real tokens after)
        """
        sample = self.samples[idx]
        
        # Prepare resource vector (convert input_size to numeric if needed)
        resource_vector = sample["resource_vector"].copy()
        if isinstance(resource_vector[0], str):
            size_map = {"small": 0, "medium": 1, "large": 2}
            resource_vector[0] = size_map.get(resource_vector[0], 0)
        
        resource_tensor = torch.tensor(resource_vector, dtype=torch.float32)
        
        # Get Qwen format text with <tool_feat> placeholder
        text_data = sample["text"]  # This is a dict from format_qwen_instruct
        prefix_text = text_data["prefix"]  # Contains "<tool_feat> Please describe..."
        suffix_text = text_data["suffix"]
        question_text = text_data["question"]
        answer_text = text_data["answer"]
        
        # Tokenize prefix (before embedding)
        prefix_encoding = self.tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Tokenize question (after embedding)
        question_encoding = self.tokenizer(
            question_text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        # Tokenize answer (after embedding)
        answer_encoding = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        prefix_labels = torch.full_like(prefix_encoding["input_ids"], -100)
        question_labels = torch.full_like(question_encoding["input_ids"], -100)
        answer_labels = answer_encoding["input_ids"].clone()

        suffix_input_ids = torch.cat([question_encoding["input_ids"], answer_encoding["input_ids"]], dim=1)
        suffix_attention_mask = torch.cat([question_encoding["attention_mask"], answer_encoding["attention_mask"]], dim=1).squeeze()
        suffix_labels = torch.cat([question_labels, answer_labels], dim=1)
        
        return {
            "tool_id": torch.tensor(sample["tool_id"], dtype=torch.long),
            "resource_vector": resource_tensor,
            "prefix_input_ids": prefix_encoding["input_ids"].squeeze(0),
            "prefix_attention_mask": prefix_encoding["attention_mask"].squeeze(0),
            "suffix_input_ids": suffix_input_ids.squeeze(),
            "suffix_attention_mask": suffix_attention_mask.squeeze(),
            "prefix_labels": prefix_labels.squeeze(0),
            "suffix_labels": suffix_labels.squeeze(0)
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