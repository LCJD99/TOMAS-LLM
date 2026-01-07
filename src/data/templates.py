"""
数据增强模板库
包含三种类型的模板：全量描述型、资源偏重型、性能偏重型
"""

import random

# 全量描述模板 - 完整的工具配置信息（同时包含数值和语义级别）
FULL_DESCRIPTION_TEMPLATES = [
    "Define the tool profile for {tool_display} with the following specs:\n- Input: {input_size} ({input_size_desc})\n- CPU: {cpu_core} cores ({cpu_core_level}), {cpu_mem}GB memory ({cpu_mem_level})\n- GPU: {gpu_sm} SM units ({gpu_sm_level}), {gpu_mem}GB VRAM ({gpu_mem_level})\n- Latency: {latency}ms",
    
    "Tool configuration for {tool_display}:\nInput size: {input_size} batch ({input_size_desc} scale)\nCPU: {cpu_core} cores ({cpu_core_level} level), {cpu_mem}GB memory ({cpu_mem_level} capacity)\nGPU: {gpu_sm} SM units ({gpu_sm_level} compute), {gpu_mem}GB VRAM ({gpu_mem_level} memory)\nExpected latency: {latency}ms",
    
    "Configure {tool_display} tool:\n• Input: {input_size} batch ({input_size_desc})\n• CPU Resources: {cpu_core} cores ({cpu_core_level}) / {cpu_mem}GB RAM ({cpu_mem_level})\n• GPU Resources: {gpu_sm} SM ({gpu_sm_level}) / {gpu_mem}GB VRAM ({gpu_mem_level})\n• Performance: ~{latency}ms",
    
    "Setup {tool_display}:\nInput scale: {input_size} ({input_size_desc} size)\nCPU allocation: {cpu_core} cores with {cpu_mem}GB memory ({cpu_core_level} cores, {cpu_mem_level} RAM)\nGPU allocation: {gpu_sm} streaming multiprocessors with {gpu_mem}GB video memory ({gpu_sm_level} compute, {gpu_mem_level} VRAM)\nTarget latency: {latency}ms",
    
    "Tool: {tool_display}\nConfiguration details:\n- Input batch size: {input_size} ({input_size_desc})\n- CPU: {cpu_core} cores ({cpu_core_level} compute power) with {cpu_mem}GB RAM ({cpu_mem_level} capacity)\n- GPU: {gpu_sm} SM units ({gpu_sm_level} utilization), {gpu_mem}GB VRAM ({gpu_mem_level} allocation)\n- Expected execution time: {latency}ms",
]

# 资源偏重模板 - 强调资源配置（同时包含数值和语义级别）
RESOURCE_FOCUSED_TEMPLATES = [
    "Tool: {tool_display}. Profile: {cpu_core} CPU cores ({cpu_core_level}), {cpu_mem}GB RAM ({cpu_mem_level}), {gpu_sm} GPU SM units ({gpu_sm_level}), {gpu_mem}GB VRAM ({gpu_mem_level}). What is the tool token?",
    
    "{tool_display} with {cpu_core} cores ({cpu_core_level} CPU), {cpu_mem}GB memory ({cpu_mem_level}), {gpu_sm} SM units ({gpu_sm_level} GPU), {gpu_mem}GB VRAM ({gpu_mem_level}). Input: {input_size} ({input_size_desc}).",
    
    "Resource config for {tool_display}: CPU={cpu_core}c ({cpu_core_level})/{cpu_mem}GB ({cpu_mem_level}), GPU={gpu_sm}SM ({gpu_sm_level})/{gpu_mem}GB ({gpu_mem_level}), size={input_size} ({input_size_desc})",
    
    "Hardware allocation for {tool_display}: {cpu_core} CPU cores ({cpu_core_level} level), {cpu_mem}GB system memory ({cpu_mem_level}), {gpu_sm} GPU SM units ({gpu_sm_level} compute), {gpu_mem}GB graphics memory ({gpu_mem_level}), processing {input_size} ({input_size_desc}) inputs",
    
    "{tool_display} - Resources: CPU({cpu_core} cores at {cpu_core_level} level, {cpu_mem}GB at {cpu_mem_level} capacity), GPU({gpu_sm} SM at {gpu_sm_level} level, {gpu_mem}GB at {gpu_mem_level} capacity), Input({input_size}, {input_size_desc} batch)",
    
    "Allocate {cpu_core} cores ({cpu_core_level}) and {cpu_mem}GB RAM ({cpu_mem_level}) for CPU, {gpu_sm} SM units ({gpu_sm_level}) and {gpu_mem}GB VRAM ({gpu_mem_level}) for GPU to run {tool_display} on {input_size} ({input_size_desc}) data",
]

# 性能偏重模板 - 强调延迟和性能需求（同时包含数值和语义级别）
PERFORMANCE_FOCUSED_TEMPLATES = [
    "I need a {tool_display} tool that runs in about {latency}ms using {gpu_mem}GB ({gpu_mem_level}) GPU memory with {input_size} ({input_size_desc}) inputs, {cpu_core} ({cpu_core_level}) CPU cores.",
    
    "Find {tool_display} configuration with ~{latency}ms latency, {cpu_mem}GB ({cpu_mem_level}) CPU memory, {input_size} ({input_size_desc}) batch size, {gpu_sm} ({gpu_sm_level}) SM units.",
    
    "Looking for {tool_display} that completes in {latency}ms with {cpu_core} cores ({cpu_core_level} CPU) and {gpu_sm} SM units ({gpu_sm_level} GPU compute) for {input_size} ({input_size_desc}) data.",
    
    "Need {tool_display} optimized for {latency}ms execution time, using {gpu_mem}GB ({gpu_mem_level}) GPU memory and {cpu_core} ({cpu_core_level}) CPU cores for {input_size} ({input_size_desc}) inputs",
    
    "Target performance: {latency}ms for {tool_display} on {input_size} ({input_size_desc}) batch. Requirements: {cpu_mem}GB ({cpu_mem_level}) RAM, {gpu_sm} ({gpu_sm_level}) SM units",
    
    "{tool_display} with approximately {latency}ms latency, {input_size} ({input_size_desc}) input scale, {cpu_core} cores ({cpu_core_level} CPU power), {gpu_mem}GB ({gpu_mem_level}) VRAM",
    
    "Performance requirement: complete {tool_display} in {latency}ms using {cpu_core} ({cpu_core_level}) CPU cores, {cpu_mem}GB ({cpu_mem_level}) RAM, and {gpu_sm} ({gpu_sm_level}) GPU SM units on {input_size} ({input_size_desc}) data",
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
    
    # 提取资源数值
    cpu_core = resources['cpu_core']
    cpu_mem = int(resources['cpu_mem_gb'])
    gpu_sm = resources['gpu_sm']
    gpu_mem = int(resources['gpu_mem_gb'])
    
    # 计算 GPU SM 百分比 (假设最大 80 SM)
    gpu_sm_pct = int((gpu_sm / 80) * 100)
    
    # 获取语义级别描述
    cpu_core_level = get_level_description(resource_levels['cpu_core_level'], vary)
    cpu_mem_level = get_level_description(resource_levels['cpu_mem_level'], vary)
    gpu_sm_level = get_level_description(resource_levels['gpu_sm_level'], vary)
    gpu_mem_level = get_level_description(resource_levels['gpu_mem_level'], vary)
    input_size_desc = get_input_size_description(input_size, vary)
    
    params = {
        'tool_display': TOOL_DISPLAY_NAMES.get(tool_name, tool_name),
        'tool_function': get_tool_function_description(tool_name, vary),
        # Input size - 数值和描述
        'input_size': input_size,
        'input_size_desc': input_size_desc,
        # CPU资源 - 数值和语义级别
        'cpu_core': cpu_core,
        'cpu_core_level': cpu_core_level,
        'cpu_mem': cpu_mem,
        'cpu_mem_level': cpu_mem_level,
        # GPU资源 - 数值和语义级别
        'gpu_sm': gpu_sm,
        'gpu_sm_pct': gpu_sm_pct,
        'gpu_sm_level': gpu_sm_level,
        'gpu_mem': gpu_mem,
        'gpu_mem_level': gpu_mem_level,
        # 性能指标
        'latency': latency,
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
