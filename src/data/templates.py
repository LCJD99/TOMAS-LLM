"""
数据增强模板库
包含三种类型的模板：全量描述型、资源偏重型、性能偏重型
"""

import random

# 全量描述模板 - 完整的工具配置信息
FULL_DESCRIPTION_TEMPLATES = [
    "Define the tool profile for {tool_display} with the following specs:\n- Input: {input_size}\n- CPU: {cpu_core} Cores, {cpu_mem}GB\n- GPU: {gpu_sm_pct}% SM, {gpu_mem}GB\n- Latency: {latency}ms",
    
    "Tool configuration for {tool_display}:\nInput size: {input_size}\nCPU: {cpu_core} cores, {cpu_mem}GB memory\nGPU: {gpu_sm_pct}% compute, {gpu_mem}GB VRAM\nExpected latency: {latency}ms",
    
    "Configure {tool_display} tool:\n• Input: {input_size} batch\n• CPU Resources: {cpu_core} cores / {cpu_mem}GB RAM\n• GPU Resources: {gpu_sm_pct}% SM / {gpu_mem}GB VRAM\n• Performance: ~{latency}ms",
    
    "Setup {tool_display}:\nInput scale: {input_size}\nCPU allocation: {cpu_core} cores, {cpu_mem}GB memory\nGPU allocation: {gpu_sm_pct}% streaming multiprocessors, {gpu_mem}GB video memory\nTarget latency: {latency}ms",
    
    "Tool: {tool_display}\nConfiguration details:\n- Input batch size: {input_size}\n- CPU: {cpu_core} cores with {cpu_mem}GB RAM\n- GPU: {gpu_sm_pct}% SM utilization, {gpu_mem}GB VRAM\n- Expected execution time: {latency}ms",
]

# 资源偏重模板 - 强调资源配置
RESOURCE_FOCUSED_TEMPLATES = [
    "Tool: {tool_display}. Profile: {cpu_core} CPU cores, {cpu_mem}GB RAM, {gpu_sm_pct}% GPU SM, {gpu_mem}GB VRAM. What is the tool token?",
    
    "{tool_display} with {cpu_core} cores, {cpu_mem}GB memory, {gpu_sm_pct}% GPU, {gpu_mem}GB VRAM. Input: {input_size}.",
    
    "Resource config for {tool_display}: CPU={cpu_core}c/{cpu_mem}GB, GPU={gpu_sm_pct}%/{gpu_mem}GB, size={input_size}",
    
    "Hardware allocation for {tool_display}: {cpu_core} CPU cores, {cpu_mem}GB system memory, {gpu_sm_pct}% GPU compute units, {gpu_mem}GB graphics memory, processing {input_size} inputs",
    
    "{tool_display} - Resources: CPU({cpu_core} cores, {cpu_mem}GB), GPU({gpu_sm_pct}% SM, {gpu_mem}GB), Input({input_size})",
    
    "Allocate {cpu_core} cores and {cpu_mem}GB RAM for CPU, {gpu_sm_pct}% SM and {gpu_mem}GB VRAM for GPU to run {tool_display} on {input_size} data",
]

# 性能偏重模板 - 强调延迟和性能需求
PERFORMANCE_FOCUSED_TEMPLATES = [
    "I need a {tool_display} tool that runs in about {latency}ms using {gpu_mem_level} GPU memory with {input_size} inputs.",
    
    "Find {tool_display} configuration with ~{latency}ms latency, {cpu_mem_level} CPU memory, {input_size} batch size.",
    
    "Looking for {tool_display} that completes in {latency}ms with {cpu_core_level} CPU and {gpu_sm_level} GPU compute for {input_size} data.",
    
    "Need {tool_display} optimized for {latency}ms execution time, using {gpu_mem_level} GPU memory and {cpu_core_level} CPU cores for {input_size} inputs",
    
    "Target performance: {latency}ms for {tool_display} on {input_size} batch. Requirements: {cpu_mem_level} RAM, {gpu_sm_level} GPU utilization",
    
    "{tool_display} with approximately {latency}ms latency, {input_size} input scale, {cpu_core_level} CPU power, {gpu_mem_level} VRAM",
    
    "Performance requirement: complete {tool_display} in {latency}ms using {cpu_core_level} CPU cores and {gpu_sm_level} GPU compute on {input_size} data",
]

# 工具功能同义词
TOOL_FUNCTIONS = {
    "image_classification": [
        "classify images",
        "categorize images", 
        "image recognition",
        "identify objects in images",
        "image categorization",
        "visual classification"
    ],
    "text_summarization": [
        "summarize text",
        "generate text summary",
        "condense documents",
        "extract key information",
        "text condensation",
        "create text summary"
    ],
    "image_captioning": [
        "caption images",
        "describe images",
        "generate image descriptions",
        "image description generation",
        "visual captioning",
        "create image captions"
    ],
    "object_detection": [
        "detect objects",
        "locate objects in images",
        "find and identify objects",
        "object localization",
        "identify object positions",
        "visual object detection"
    ],
    "machine_translation": [
        "translate text",
        "convert between languages",
        "language translation",
        "text translation",
        "translate languages",
        "convert text to another language"
    ],
    "super_resolution": [
        "enhance image resolution",
        "upscale images",
        "improve image quality",
        "increase image resolution",
        "image upscaling",
        "resolution enhancement"
    ],
    "visual_question_answering": [
        "answer questions about images",
        "visual Q&A",
        "image understanding",
        "answer visual questions",
        "image-based question answering",
        "visual reasoning"
    ]
}

# 工具名称显示格式映射
TOOL_DISPLAY_NAMES = {
    "image_classification": "Image Classification",
    "text_summarization": "Text Summarization",
    "image_captioning": "Image Captioning",
    "object_detection": "Object Detection",
    "machine_translation": "Machine Translation",
    "super_resolution": "Super Resolution",
    "visual_question_answering": "Visual Question Answering"
}

# 资源等级描述词
LEVEL_DESCRIPTIONS = {
    "low": ["minimal", "low", "basic", "small", "limited"],
    "medium": ["moderate", "medium", "standard", "average", "balanced"],
    "high": ["high", "large", "substantial", "generous", "ample"]
}

# Input size 描述词
INPUT_SIZE_VARIATIONS = {
    "small": ["small", "small-scale", "compact", "limited"],
    "medium": ["medium", "moderate", "standard", "typical"],
    "large": ["large", "large-scale", "extensive", "substantial"]
}


def get_level_description(level: str, vary: bool = True) -> str:
    """获取资源等级的描述词"""
    if vary:
        return random.choice(LEVEL_DESCRIPTIONS.get(level, [level]))
    return level


def get_input_size_description(size: str, vary: bool = True) -> str:
    """获取输入规模的描述词"""
    if vary:
        return random.choice(INPUT_SIZE_VARIATIONS.get(size, [size]))
    return size


def get_tool_function_description(tool_name: str, vary: bool = True) -> str:
    """获取工具功能的描述"""
    if vary and tool_name in TOOL_FUNCTIONS:
        return random.choice(TOOL_FUNCTIONS[tool_name])
    return TOOL_DISPLAY_NAMES.get(tool_name, tool_name)


def format_template(template: str, token_info: dict, vary: bool = True) -> str:
    """
    使用 token 信息填充模板
    
    Args:
        template: 模板字符串
        token_info: token 的配置信息
        vary: 是否使用同义词变化
    
    Returns:
        填充后的文本
    """
    tool_name = token_info['tool_name']
    resources = token_info['resources']
    resource_levels = token_info['resource_levels']
    input_size = token_info['input_size']
    latency = int(token_info['latency_ms'])
    
    # 计算 GPU SM 百分比 (假设最大 80 SM)
    gpu_sm = resources['gpu_sm']
    gpu_sm_pct = int((gpu_sm / 80) * 100)
    
    params = {
        'tool_display': TOOL_DISPLAY_NAMES.get(tool_name, tool_name),
        'tool_function': get_tool_function_description(tool_name, vary),
        'input_size': get_input_size_description(input_size, vary),
        'cpu_core': resources['cpu_core'],
        'cpu_mem': int(resources['cpu_mem_gb']),
        'gpu_sm_pct': gpu_sm_pct,
        'gpu_mem': int(resources['gpu_mem_gb']),
        'latency': latency,
        'cpu_core_level': get_level_description(resource_levels['cpu_core_level'], vary),
        'cpu_mem_level': get_level_description(resource_levels['cpu_mem_level'], vary),
        'gpu_sm_level': get_level_description(resource_levels['gpu_sm_level'], vary),
        'gpu_mem_level': get_level_description(resource_levels['gpu_mem_level'], vary),
    }
    
    return template.format(**params)


def generate_full_description(token_info: dict) -> str:
    """生成全量描述型输入"""
    template = random.choice(FULL_DESCRIPTION_TEMPLATES)
    return format_template(template, token_info, vary=True)


def generate_resource_focused(token_info: dict) -> str:
    """生成资源偏重型输入"""
    template = random.choice(RESOURCE_FOCUSED_TEMPLATES)
    return format_template(template, token_info, vary=True)


def generate_performance_focused(token_info: dict) -> str:
    """生成性能偏重型输入"""
    template = random.choice(PERFORMANCE_FOCUSED_TEMPLATES)
    return format_template(template, token_info, vary=True)


# 模板生成器映射
AUGMENTATION_GENERATORS = {
    'full_description': generate_full_description,
    'resource_focused': generate_resource_focused,
    'performance_focused': generate_performance_focused,
}
