"""
Offline processing modules for TOMAS-LLM.

This package contains modules for Phase 0 (Encoder Pre-training):
- ConfigTextTemplate: Converts resource configs to natural language templates
- ConfigEmbeddingGenerator: [DEPRECATED] Will be replaced by pretrain_encoder
- ConfigLookupBuilder: Creates config ID to resource mapping tables

These modules run offline to pre-train the encoder and generate assets.
"""

from .text_template import ConfigTextTemplate
from .embedding_generator import ConfigEmbeddingGenerator
from .lookup_builder import ConfigLookupBuilder

__all__ = ['ConfigTextTemplate', 'ConfigEmbeddingGenerator', 'ConfigLookupBuilder']