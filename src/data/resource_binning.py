"""
Resource binning configuration for TOMAS-LLM.

Defines the mapping from actual resource values to semantic levels (low/medium/high).
"""

# Resource binning mappings
RESOURCE_BINS = {
    'cpu_core': {
        'low': 2,
        'medium': 4,
        'high': 8
    },
    'cpu_mem_gb': {
        'low': 4.0,
        'medium': 8.0,
        'high': 16.0
    },
    'gpu_sm': {
        'low': 20,
        'medium': 40,
        'high': 60
    },
    'gpu_mem_gb': {
        'low': 2.0,
        'medium': 4.0,
        'high': 8.0
    }
}

# Input size categories (already semantic)
INPUT_SIZES = ['small', 'medium', 'large']


def value_to_level(resource_type: str, value: float) -> str:
    """
    Convert a resource value to its semantic level.
    
    Args:
        resource_type: Type of resource (cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb)
        value: Actual resource value
    
    Returns:
        Semantic level: 'low', 'medium', or 'high'
    
    Examples:
        >>> value_to_level('cpu_core', 2)
        'low'
        >>> value_to_level('gpu_sm', 60)
        'high'
    """
    if resource_type not in RESOURCE_BINS:
        raise ValueError(f"Unknown resource type: {resource_type}")
    
    bins = RESOURCE_BINS[resource_type]
    
    # Find matching bin
    for level, bin_value in bins.items():
        if value == bin_value:
            return level
    
    # If exact match not found, find closest
    closest_level = min(bins.items(), key=lambda x: abs(x[1] - value))[0]
    return closest_level


def level_to_value(resource_type: str, level: str) -> float:
    """
    Convert a semantic level to its resource value.
    
    Args:
        resource_type: Type of resource
        level: Semantic level ('low', 'medium', 'high')
    
    Returns:
        Actual resource value
    
    Examples:
        >>> level_to_value('cpu_core', 'low')
        2
        >>> level_to_value('gpu_mem_gb', 'high')
        8.0
    """
    if resource_type not in RESOURCE_BINS:
        raise ValueError(f"Unknown resource type: {resource_type}")
    
    if level not in RESOURCE_BINS[resource_type]:
        raise ValueError(f"Unknown level '{level}' for resource '{resource_type}'")
    
    return RESOURCE_BINS[resource_type][level]


if __name__ == "__main__":
    # Test binning functions
    print("Resource Binning Tests:")
    print(f"cpu_core=4 -> {value_to_level('cpu_core', 4)}")
    print(f"gpu_sm=20 -> {value_to_level('gpu_sm', 20)}")
    print(f"level='high' for cpu_mem_gb -> {level_to_value('cpu_mem_gb', 'high')}")
