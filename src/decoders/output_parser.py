"""
Output Parser

Converts model outputs to executable tool plans.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class ToolPlan:
    """
    Executable tool plan with resource allocation.
    
    This is the final output format from TOMAS-LLM.
    """
    # Tool selection
    tool_id: int
    tool_name: str
    confidence: float
    
    # Resource allocation
    cpu_core: float
    cpu_mem_gb: float
    gpu_sm: float
    gpu_mem_gb: float
    
    # Optional fields
    expected_latency_ms: Optional[float] = None
    explanation: Optional[str] = None
    input_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolPlan':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ToolPlan':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> bool:
        """Validate resource allocation."""
        # Check non-negative resources
        if self.cpu_core < 0 or self.cpu_mem_gb < 0:
            return False
        if self.gpu_sm < 0 or self.gpu_mem_gb < 0:
            return False
        
        # Check confidence
        if not (0.0 <= self.confidence <= 1.0):
            return False
        
        # Check GPU consistency
        if self.gpu_sm > 0 and self.gpu_mem_gb == 0:
            return False
        if self.gpu_sm == 0 and self.gpu_mem_gb > 0:
            return False
        
        return True
    
    def __repr__(self):
        return (
            f"ToolPlan(\n"
            f"  tool={self.tool_name} (id={self.tool_id}, conf={self.confidence:.3f}),\n"
            f"  cpu={self.cpu_core:.1f} cores, {self.cpu_mem_gb:.1f} GB,\n"
            f"  gpu={self.gpu_sm:.0f} SM, {self.gpu_mem_gb:.1f} GB\n"
            f")"
        )


class OutputParser(nn.Module):
    """
    Output Parser
    
    Parses model outputs (LLM hidden states + special token) into executable tool plans.
    
    Flow:
        1. Detect TOOL_PLAN token in generated sequence
        2. Extract hidden state at TOOL_PLAN position
        3. Feed to BOTH heads simultaneously (single hidden → dual outputs):
           - ToolClassifier(hidden) → tool_id
           - ResourceRegressor(hidden, tool_emb, temporal) → resources
        4. Combine into ToolPlan
    
    Key Design:
        - ONE special token encodes ALL information
        - Like encoder: single representation, multiple decodings
        - Both heads share the same input hidden state
    
    Usage:
        parser = OutputParser(
            tool_classifier=tool_classifier,
            resource_regressor=resource_regressor,
            token_gate=token_gate
        )
        
        # Parse generation output
        plan = parser.parse(
            generated_ids=generated_ids,        # (batch, seq)
            hidden_states=hidden_states,        # (batch, seq, hidden_dim)
            tool_embeddings=tool_embeddings,    # (num_tools, tool_dim)
            temporal_emb=temporal_emb,          # (temporal_dim,)
            available_resources=available_res   # (4,)
        )
    """
    
    def __init__(
        self,
        tool_classifier,
        resource_regressor,
        token_gate,
        tool_id_to_name: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            tool_classifier: ToolClassifier instance
            resource_regressor: ResourceRegressor instance
            token_gate: TokenTypeGate instance
            tool_id_to_name: Mapping from tool ID to tool name
        """
        super().__init__()
        
        self.tool_classifier = tool_classifier
        self.resource_regressor = resource_regressor
        self.token_gate = token_gate
        
        # Tool name mapping
        if tool_id_to_name is not None:
            self.tool_id_to_name = tool_id_to_name
        else:
            # Default: tool_0, tool_1, ...
            num_tools = tool_classifier.num_tools
            self.tool_id_to_name = {i: f"tool_{i}" for i in range(num_tools)}
    
    def parse(
        self,
        generated_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
    ) -> List[ToolPlan]:
        """
        Parse generation output into tool plans.
        
        Args:
            generated_ids: (batch, seq) - Generated token IDs
            hidden_states: (batch, seq, hidden_dim) - LLM hidden states
            tool_embeddings: (num_tools, tool_dim) - Tool embeddings
            available_resources: (4,) or (batch, 4) - Available resources [optional]
        
        Returns:
            plans: List of ToolPlan objects (length = batch_size)
        """
        batch_size = generated_ids.size(0)
        plans = []
        
        for i in range(batch_size):
            # Extract single sequence
            seq_ids = generated_ids[i]  # (seq,)
            seq_hidden = hidden_states[i]  # (seq, hidden_dim)
            
            # Find TOOL_PLAN token positions
            masks = self.token_gate.get_routing_mask(seq_ids.unsqueeze(0))
            tool_plan_mask = masks['special_mask'][0]  # (seq,)
            
            # Check if TOOL_PLAN token exists
            has_tool_plan = tool_plan_mask.any()
            
            if not has_tool_plan:
                # No valid tool plan in this sequence
                plans.append(self._create_default_plan())
                continue
            
            # Extract hidden state at TOOL_PLAN position (use first occurrence)
            tool_plan_pos = torch.nonzero(tool_plan_mask, as_tuple=False)[0].item()
            plan_hidden = seq_hidden[tool_plan_pos].unsqueeze(0)  # (1, hidden_dim)
            
            # ========================================
            # Dual Decoding from Single Hidden State
            # ========================================
            
            # 1. Predict tool (ToolClassifier)
            tool_pred = self.tool_classifier.predict(
                plan_hidden,
                tool_embeddings,
                return_probs=False
            )
            
            tool_id = tool_pred['tool_id'][0].item()
            tool_name = self.tool_id_to_name.get(tool_id, f"unknown_{tool_id}")
            confidence = tool_pred['confidence'][0].item()
            
            # 2. Predict resources (ResourceRegressor)
            # Note: Uses SAME hidden state as ToolClassifier
            # LLM hidden already contains tool + temporal information!
            resource_alloc = self.resource_regressor.predict(
                plan_hidden,  # Same input! Single-source dual-decoding
                available_resources=available_resources
            )
            
            cpu_core = resource_alloc['cpu_core'][0].item()
            cpu_mem_gb = resource_alloc['cpu_mem_gb'][0].item()
            gpu_sm = resource_alloc['gpu_sm'][0].item()
            gpu_mem_gb = resource_alloc['gpu_mem_gb'][0].item()
            
            # Create plan
            plan = ToolPlan(
                tool_id=tool_id,
                tool_name=tool_name,
                confidence=confidence,
                cpu_core=cpu_core,
                cpu_mem_gb=cpu_mem_gb,
                gpu_sm=gpu_sm,
                gpu_mem_gb=gpu_mem_gb,
            )
            
            plans.append(plan)
        
        return plans
    
    def parse_with_text(
        self,
        generated_ids: torch.Tensor,
        generated_text: List[str],
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        temporal_emb: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
    ) -> List[ToolPlan]:
        """
        Parse with explanatory text.
        
        Args:
            generated_ids: (batch, seq)
            generated_text: List of generated text strings
            hidden_states: (batch, seq, hidden_dim)
            tool_embeddings: (num_tools, tool_dim)
            temporal_emb: (temporal_dim,)
            available_resources: (4,) or (batch, 4)
        
        Returns:
            plans: List of ToolPlan with explanation field filled
        """
        plans = self.parse(
            generated_ids, hidden_states,
            tool_embeddings, temporal_emb,
            available_resources
        )
        
        # Add explanations
        for plan, text in zip(plans, generated_text):
            plan.explanation = text
        
        return plans
    
    def _create_default_plan(self) -> ToolPlan:
        """Create a default plan when parsing fails."""
        return ToolPlan(
            tool_id=-1,
            tool_name="none",
            confidence=0.0,
            cpu_core=0.0,
            cpu_mem_gb=0.0,
            gpu_sm=0.0,
            gpu_mem_gb=0.0,
        )
    
    def forward(self, *args, **kwargs):
        """Alias for parse()."""
        return self.parse(*args, **kwargs)


class BatchOutputParser:
    """
    Batch output parser for efficient multi-sample parsing.
    
    Processes entire batches at once using vectorized operations.
    Key: All samples use SAME hidden state for both tool and resource prediction.
    """
    
    def __init__(
        self,
        tool_classifier,
        resource_regressor,
        token_gate,
        tool_id_to_name: Optional[Dict[int, str]] = None,
    ):
        self.tool_classifier = tool_classifier
        self.resource_regressor = resource_regressor
        self.token_gate = token_gate
        self.tool_id_to_name = tool_id_to_name or {}
    
    def parse_batch(
        self,
        generated_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        available_resources: Optional[torch.Tensor] = None,
    ) -> List[ToolPlan]:
        """
        Parse entire batch at once (vectorized).
        
        Key optimization: Extract all TOOL_PLAN hidden states, then batch process.
        """
        # Extract all TOOL_PLAN positions
        plan_hidden, plan_indices = self.token_gate.extract_special_positions(
            generated_ids, hidden_states
        )
        
        if plan_hidden.size(0) == 0:
            # No valid plans
            batch_size = generated_ids.size(0)
            return [self._create_default_plan() for _ in range(batch_size)]
        
        # Batch predict tools
        tool_pred = self.tool_classifier.predict(
            plan_hidden,
            tool_embeddings,
            return_probs=False
        )
        
        tool_ids = tool_pred['tool_id']  # (num_special,)
        confidences = tool_pred['confidence']  # (num_special,)
        
        # Batch predict resources (using SAME hidden states)
        # No need for tool embeddings - hidden state already knows the tool!
        resource_alloc = self.resource_regressor.predict(
            plan_hidden,  # Same as tool prediction! Single-source dual-decoding
            available_resources
        )
        
        # Create plans
        plans = []
        for i in range(tool_ids.size(0)):
            tool_id = tool_ids[i].item()
            tool_name = self.tool_id_to_name.get(tool_id, f"tool_{tool_id}")
            
            plan = ToolPlan(
                tool_id=tool_id,
                tool_name=tool_name,
                confidence=confidences[i].item(),
                cpu_core=resource_alloc['cpu_core'][i].item(),
                cpu_mem_gb=resource_alloc['cpu_mem_gb'][i].item(),
                gpu_sm=resource_alloc['gpu_sm'][i].item(),
                gpu_mem_gb=resource_alloc['gpu_mem_gb'][i].item(),
            )
            plans.append(plan)
        
        return plans
    
    def _create_default_plan(self) -> ToolPlan:
        """Create default plan."""
        return ToolPlan(
            tool_id=-1,
            tool_name="none",
            confidence=0.0,
            cpu_core=0.0,
            cpu_mem_gb=0.0,
            gpu_sm=0.0,
            gpu_mem_gb=0.0,
        )
