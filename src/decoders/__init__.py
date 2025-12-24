"""
Output Generation & Parsing Module

This module implements the Right Panel of TOMAS-LLM:
- TokenTypeGate: Route LLM outputs to appropriate heads
- ToolClassifier: Select tools from registry
- ResourceRegressor: Allocate CPU/GPU resources
- OutputParser: Convert raw outputs to executable plans
"""

from .token_gate import TokenTypeGate, TOOL_PLAN_TOKEN_OFFSET
from .tool_classifier import ToolClassifier
from .resource_regressor import ResourceRegressor
from .output_parser import OutputParser, ToolPlan

__all__ = [
    'TokenTypeGate',
    'TOOL_PLAN_TOKEN_OFFSET',
    'ToolClassifier',
    'ResourceRegressor',
    'OutputParser',
    'ToolPlan',
]
