"""
Token schema and naming conventions for TOMAS-LLM.

Defines the virtual token naming format and tool abbreviations.
"""

from typing import Dict, Tuple

try:
    from .resource_binning import RESOURCE_BINS, INPUT_SIZES
except ImportError:
    from resource_binning import RESOURCE_BINS, INPUT_SIZES


# Tool name abbreviations
TOOL_ABBREV = {
    'image_classification': 'IMG_CLS',
    'text_summarization': 'TXT_SUM',
    'image_captioning': 'IMG_CAP',
    'object_detection': 'OBJ_DET',
    'machine_translation': 'MACH_TRANS',
    'super_resolution': 'SUPER_RES',
    'visual_question_answering': 'VQA'
}

# Reverse mapping
ABBREV_TO_TOOL = {v: k for k, v in TOOL_ABBREV.items()}

# Level abbreviations for token names
LEVEL_ABBREV = {
    'low': 'LOW',
    'medium': 'MED',
    'high': 'HIGH'
}

# Input size abbreviations
SIZE_ABBREV = {
    'small': 'SMALL',
    'medium': 'MEDIUM',
    'large': 'LARGE'
}


def generate_token_name(
    tool_name: str,
    input_size: str,
    cpu_core_level: str,
    cpu_mem_level: str,
    gpu_sm_level: str,
    gpu_mem_level: str
) -> str:
    """
    Generate virtual token name from configuration.
    
    Format: <TOOL>_<INPUT_SIZE>_<CPU_CORE>_<CPU_MEM>_<GPU_SM>_<GPU_MEM>
    
    Args:
        tool_name: Full tool name (e.g., 'image_classification')
        input_size: Input size ('small', 'medium', 'large')
        cpu_core_level: CPU core level ('low', 'medium', 'high')
        cpu_mem_level: CPU memory level ('low', 'medium', 'high')
        gpu_sm_level: GPU SM level ('low', 'medium', 'high')
        gpu_mem_level: GPU memory level ('low', 'medium', 'high')
    
    Returns:
        Virtual token name (e.g., 'IMG_CLS_SMALL_LOW_MED_HIGH_LOW')
    
    Examples:
        >>> generate_token_name('image_classification', 'small', 'low', 'medium', 'high', 'low')
        'IMG_CLS_SMALL_LOW_MED_HIGH_LOW'
    """
    if tool_name not in TOOL_ABBREV:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    if input_size not in SIZE_ABBREV:
        raise ValueError(f"Unknown input size: {input_size}")
    
    for level in [cpu_core_level, cpu_mem_level, gpu_sm_level, gpu_mem_level]:
        if level not in LEVEL_ABBREV:
            raise ValueError(f"Unknown level: {level}")
    
    token = f"{TOOL_ABBREV[tool_name]}_{SIZE_ABBREV[input_size]}_" \
            f"{LEVEL_ABBREV[cpu_core_level]}_{LEVEL_ABBREV[cpu_mem_level]}_" \
            f"{LEVEL_ABBREV[gpu_sm_level]}_{LEVEL_ABBREV[gpu_mem_level]}"
    
    return token


def parse_token_name(token: str) -> Dict[str, str]:
    """
    Parse virtual token name into components.
    
    Args:
        token: Virtual token name (e.g., 'IMG_CLS_SMALL_LOW_MED_HIGH_LOW')
    
    Returns:
        Dictionary with components: tool_name, input_size, cpu_core_level, etc.
    
    Examples:
        >>> parse_token_name('IMG_CLS_SMALL_LOW_MED_HIGH_LOW')
        {
            'tool_name': 'image_classification',
            'input_size': 'small',
            'cpu_core_level': 'low',
            'cpu_mem_level': 'medium',
            'gpu_sm_level': 'high',
            'gpu_mem_level': 'low'
        }
    """
    parts = token.split('_')
    
    # Handle multi-part tool abbreviations (e.g., MACH_TRANS)
    if len(parts) == 7:
        tool_abbrev = f"{parts[0]}_{parts[1]}"
        input_size_abbrev = parts[2]
        levels = parts[3:]
    elif len(parts) == 6:
        tool_abbrev = parts[0]
        input_size_abbrev = parts[1]
        levels = parts[2:]
    else:
        raise ValueError(f"Invalid token format: {token}")
    
    # Reverse lookups
    tool_name = ABBREV_TO_TOOL.get(tool_abbrev)
    if not tool_name:
        raise ValueError(f"Unknown tool abbreviation: {tool_abbrev}")
    
    input_size = input_size_abbrev.lower()
    if input_size not in INPUT_SIZES:
        raise ValueError(f"Unknown input size: {input_size}")
    
    # Reverse level abbreviations
    level_reverse = {v: k for k, v in LEVEL_ABBREV.items()}
    cpu_core_level = level_reverse[levels[0]]
    cpu_mem_level = level_reverse[levels[1]]
    gpu_sm_level = level_reverse[levels[2]]
    gpu_mem_level = level_reverse[levels[3]]
    
    return {
        'tool_name': tool_name,
        'input_size': input_size,
        'cpu_core_level': cpu_core_level,
        'cpu_mem_level': cpu_mem_level,
        'gpu_sm_level': gpu_sm_level,
        'gpu_mem_level': gpu_mem_level
    }


def format_token_for_model(token: str) -> str:
    """
    Format token for model use (add angle brackets).
    
    Args:
        token: Token name
    
    Returns:
        Formatted token (e.g., '<IMG_CLS_SMALL_LOW_MED_HIGH_LOW>')
    """
    return f"<{token}>"


def strip_token_brackets(token: str) -> str:
    """
    Remove angle brackets from token.
    
    Args:
        token: Formatted token (e.g., '<IMG_CLS_SMALL_LOW_MED_HIGH_LOW>')
    
    Returns:
        Token name without brackets
    """
    return token.strip('<>')


if __name__ == "__main__":
    # Test token generation and parsing
    print("Token Schema Tests:")
    
    token = generate_token_name(
        'image_classification', 'small', 'low', 'medium', 'high', 'low'
    )
    print(f"Generated token: {token}")
    print(f"Formatted: {format_token_for_model(token)}")
    
    parsed = parse_token_name(token)
    print(f"Parsed: {parsed}")
    
    # Test multi-word tool
    token2 = generate_token_name(
        'machine_translation', 'large', 'high', 'high', 'medium', 'high'
    )
    print(f"\nMulti-word tool token: {token2}")
    print(f"Parsed: {parse_token_name(token2)}")
