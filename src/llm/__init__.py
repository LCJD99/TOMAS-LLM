"""
LLM module - Qwen2.5 integration and context injection
"""

from .qwen_backbone import QwenBackbone, ContextProjector
from .model_wrapper import TOMASSLLMModel

__all__ = [
    "QwenBackbone",
    "ContextProjector",
    "TOMASSLLMModel"
]
