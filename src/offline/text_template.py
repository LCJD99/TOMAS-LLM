"""
Natural Language Template Generator for Resource Configurations.

Converts (Tool, Resource) pairs into natural language descriptions for encoder pre-training.
"""

from typing import List, Dict, Union
import torch


class ConfigTextTemplate:
    """
    Generates natural language descriptions from resource configuration data.
    
    Example output:
        "Tool super_resolution configuration: input size small, requires 4 CPU cores, 
         4 GB memory, 50 GPU SMs, 2 GB GPU memory, and latency is 5800 ms."
    """
    
    # Mapping for input size encoding
    INPUT_SIZE_MAP = {
        "small": "small",
        "medium": "medium", 
        "large": "large"
    }
    
    def __init__(self, use_variations: bool = True):
        """
        Initialize the template generator.
        
        Args:
            use_variations: Whether to use multiple template variations
        """
        self.use_variations = use_variations
        self.variation_index = 0
    
    def format_single(
        self, 
        tool_name: str, 
        resource_vector: Union[List[float], torch.Tensor, Dict[str, float]],
        variation_idx: int = 0
    ) -> str:
        """
        Convert a single (Tool, Resource) pair to natural language.
        
        Args:
            tool_name: Name of the tool (e.g., "super_resolution")
            resource_vector: Either:
                - List/Tensor: [input_size, cpu_cores, memory_gb, gpu_sm, gpu_memory_gb, latency_ms]
                - Dict: {"input_size": ..., "cpu_cores": ..., ...}
            variation_idx: Template variation index (0-4)
        
        Returns:
            Natural language description string.
        """
        # Parse resource vector
        if isinstance(resource_vector, dict):
            input_size = resource_vector["input_size"]
            cpu_cores = resource_vector["cpu_cores"]
            memory_gb = resource_vector["memory_gb"]
            gpu_sm = resource_vector["gpu_sm"]
            gpu_memory_gb = resource_vector["gpu_memory_gb"]
            latency_ms = resource_vector["latency_ms"]
        else:
            # List or Tensor format
            if isinstance(resource_vector, torch.Tensor):
                resource_vector = resource_vector.tolist()
            
            input_size, cpu_cores, memory_gb, gpu_sm, gpu_memory_gb, latency_ms = resource_vector
        
        # Format input size (handle both string and numeric encoding)
        if isinstance(input_size, str):
            input_size_str = input_size
        else:
            # If numeric, map to string (assuming small=0, medium=1, large=2)
            size_map = {0: "small", 1: "medium", 2: "large"}
            input_size_str = size_map.get(int(input_size), "small")
        
        # Convert to int for cleaner output
        cpu_cores = int(cpu_cores)
        memory_gb = int(memory_gb)
        gpu_sm = int(gpu_sm)
        gpu_memory_gb = int(gpu_memory_gb)
        latency_ms = int(latency_ms)
        
        # Multiple template variations to avoid all starting with "Tool"
        if not self.use_variations:
            variation_idx = 0
        
        templates = [
            # Variation 0: Original format
            (f"Tool {tool_name} configuration: "
             f"input size {input_size_str}, "
             f"requires {cpu_cores} CPU cores, "
             f"{memory_gb} GB memory, "
             f"{gpu_sm} GPU SMs, "
             f"{gpu_memory_gb} GB GPU memory, "
             f"and latency is {latency_ms} ms."),
            
            # Variation 1: Resource-first format
            (f"Resource allocation for {tool_name}: "
             f"{cpu_cores} CPU cores, {memory_gb} GB RAM, "
             f"{gpu_sm} GPU SMs with {gpu_memory_gb} GB VRAM, "
             f"processes {input_size_str} inputs in {latency_ms} ms."),
            
            # Variation 2: Performance-focused format
            (f"The {tool_name} tool processes {input_size_str} inputs with {latency_ms} ms latency, "
             f"using {cpu_cores} CPU cores, {memory_gb} GB memory, "
             f"{gpu_sm} GPU SMs and {gpu_memory_gb} GB GPU memory."),
            
            # Variation 3: Specification format
            (f"Configuration: {tool_name} | Input: {input_size_str} | "
             f"CPU: {cpu_cores} cores, {memory_gb} GB | "
             f"GPU: {gpu_sm} SMs, {gpu_memory_gb} GB | "
             f"Latency: {latency_ms} ms"),
            
            # Variation 4: Natural description
            (f"Running {tool_name} on {input_size_str} data needs "
             f"{cpu_cores} cores and {memory_gb} GB of memory, "
             f"plus {gpu_sm} SMs with {gpu_memory_gb} GB GPU memory, "
             f"achieving {latency_ms} ms latency."),
        ]
        
        return templates[variation_idx % len(templates)]
    
    def format_batch(
        self, 
        data_list: List[Dict[str, Union[str, List, torch.Tensor, Dict]]]
    ) -> List[str]:
        """
        Convert a batch of (Tool, Resource) pairs to natural language.
        
        Args:
            data_list: List of dictionaries, each containing:
                - "tool_name": str
                - "resource_vector": List/Tensor/Dict
        
        Returns:
            List of natural language descriptions.
        """
        return [
            self.format_single(item["tool_name"], item["resource_vector"])
            for item in data_list
        ]
    
    def format_with_embedding_placeholder(
        self,
        tool_name: str,
        resource_vector: Union[List[float], torch.Tensor, Dict[str, float]],
        variation_idx: int = 0
    ) -> str:
        """
        Generate template with [TOOL_RESOURCE] placeholder for encoder embedding injection.
        
        Format: "The information and resource constraint of the tool is expressed as [TOOL_RESOURCE], 
                 that means Tool {tool_name} processes..."
        
        Args:
            tool_name: Name of the tool
            resource_vector: Resource configuration
            variation_idx: Index to select template variation (0-2)
        
        Returns:
            Natural language description with [TOOL_RESOURCE] placeholder.
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
        
        # Template variations with placeholder
        templates = [
            # Variation 0: Standard format with placeholder
            (
                f"The information and resource constraint of the tool is expressed as [TOOL_RESOURCE], "
                f"that means Tool {tool_name} processes {input_size_str} input with "
                f"{int(cpu_cores)} CPU cores, {int(memory_gb)} GB memory, "
                f"{int(gpu_sm)} GPU SMs, {int(gpu_memory_gb)} GB GPU memory, "
                f"and achieves {int(latency_ms)} ms latency."
            ),
            # Variation 1: Alternative phrasing
            (
                f"The tool configuration is represented as [TOOL_RESOURCE], "
                f"which indicates {tool_name} operates on {input_size_str} data using "
                f"{int(cpu_cores)} cores, {int(memory_gb)} GB RAM, "
                f"{int(gpu_sm)} SMs with {int(gpu_memory_gb)} GB GPU memory, "
                f"completing in {int(latency_ms)} ms."
            ),
            # Variation 2: Concise format
            (
                f"As [TOOL_RESOURCE] describes, {tool_name} requires "
                f"{int(cpu_cores)} CPU cores and {int(memory_gb)} GB memory, "
                f"with {int(gpu_sm)} GPU SMs and {int(gpu_memory_gb)} GB GPU memory "
                f"to process {input_size_str} inputs in {int(latency_ms)} ms."
            ),
        ]
        
        return templates[variation_idx % len(templates)]
    
    def format_with_variation(
        self,
        tool_name: str,
        resource_vector: Union[List[float], torch.Tensor, Dict[str, float]],
        variation_idx: int = 0
    ) -> str:
        """
        Generate template with slight linguistic variations for data augmentation.
        
        Args:
            tool_name: Name of the tool
            resource_vector: Resource configuration
            variation_idx: Index to select template variation (0-2)
        
        Returns:
            Natural language description with variation.
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
        
        # Template variations
        templates = [
            # Variation 0: Standard format
            (
                f"Tool {tool_name} configuration: "
                f"input size {input_size_str}, "
                f"requires {int(cpu_cores)} CPU cores, "
                f"{int(memory_gb)} GB memory, "
                f"{int(gpu_sm)} GPU SMs, "
                f"{int(gpu_memory_gb)} GB GPU memory, "
                f"and latency is {int(latency_ms)} ms."
            ),
            # Variation 1: Reordered format
            (
                f"The {tool_name} tool with {input_size_str} input needs "
                f"{int(cpu_cores)} CPU cores and {int(memory_gb)} GB RAM, "
                f"plus {int(gpu_sm)} GPU SMs with {int(gpu_memory_gb)} GB GPU memory, "
                f"achieving {int(latency_ms)} ms latency."
            ),
            # Variation 2: Concise format
            (
                f"{tool_name} ({input_size_str}): "
                f"CPU={int(cpu_cores)} cores, RAM={int(memory_gb)}GB, "
                f"GPU={int(gpu_sm)} SMs/{int(gpu_memory_gb)}GB, "
                f"latency={int(latency_ms)}ms."
            )
        ]
        
        return templates[variation_idx % len(templates)]


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


if __name__ == "__main__":
    test_template_generator()
