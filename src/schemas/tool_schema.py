"""
Tool and Profiling Data Schemas using Pydantic
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ToolSchema(BaseModel):
    """
    Schema for tool semantic description in the tool registry.
    
    Attributes:
        name: Unique identifier for the tool
        desc: Functional description of what the tool does
    """
    name: str = Field(..., min_length=1, max_length=100, description="Tool name (unique identifier)")
    desc: str = Field(..., min_length=10, max_length=1000, description="Tool description")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure tool name contains no whitespace and is lowercase"""
        if ' ' in v:
            raise ValueError("Tool name cannot contain whitespace")
        return v.lower()
    
    @field_validator('desc')
    @classmethod
    def validate_desc(cls, v: str) -> str:
        """Ensure description is non-empty after stripping"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Tool description cannot be empty")
        return stripped


class ProfilingSchema(BaseModel):
    """
    Schema for tool profiling data (resource consumption metrics).
    
    The input_size is bucketed into three levels: small, medium, large.
    
    Attributes:
        tool: Tool name (must match a name in tool registry)
        input_size: Input size bucket (small/medium/large)
        cpu_core: Number of CPU cores required
        cpu_mem_gb: CPU memory in gigabytes
        gpu_sm: GPU streaming multiprocessors (SM) percentage (0-100)
        gpu_mem_gb: GPU memory in gigabytes
        latency_ms: Expected latency in milliseconds
    """
    tool: str = Field(..., description="Tool name")
    input_size: Literal["small", "medium", "large"] = Field(..., description="Input size bucket")
    cpu_core: int = Field(..., ge=0, description="CPU cores required")
    cpu_mem_gb: float = Field(..., ge=0.0, description="CPU memory in GB")
    gpu_sm: int = Field(..., ge=0, le=100, description="GPU SM percentage (0-100)")
    gpu_mem_gb: float = Field(..., ge=0.0, description="GPU memory in GB")
    latency_ms: float = Field(..., ge=0.0, description="Expected latency in milliseconds")
    
    @field_validator('tool')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is lowercase and no whitespace"""
        return v.lower().strip()


class ResourceConfig(BaseModel):
    """
    Schema for resource allocation configuration (output from model).
    
    Attributes:
        cpu_core: Allocated CPU cores
        cpu_mem_gb: Allocated CPU memory in GB
        gpu_sm: Allocated GPU SM percentage
        gpu_mem_gb: Allocated GPU memory in GB
    """
    cpu_core: int = Field(..., ge=0, description="Allocated CPU cores")
    cpu_mem_gb: float = Field(..., ge=0.0, description="Allocated CPU memory in GB")
    gpu_sm: int = Field(..., ge=0, le=100, description="Allocated GPU SM percentage")
    gpu_mem_gb: float = Field(..., ge=0.0, description="Allocated GPU memory in GB")


class ToolPlanSchema(BaseModel):
    """
    Schema for the final executable tool plan (model output).
    
    Attributes:
        task: Original user task description
        tool_id: Selected tool identifier
        tool_name: Selected tool name
        resource_config: Resource allocation configuration
        expected_latency_ms: Predicted execution latency
        status: Plan status (optional)
    """
    task: str = Field(..., description="User task description")
    tool_id: str = Field(..., description="Tool identifier")
    tool_name: str = Field(..., description="Tool name")
    resource_config: ResourceConfig = Field(..., description="Resource allocation")
    expected_latency_ms: Optional[float] = Field(None, ge=0.0, description="Expected latency")
    status: Optional[str] = Field(None, description="Plan execution status")
