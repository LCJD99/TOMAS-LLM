"""
Tool Classifier Head

Softmax classifier for tool selection from registry.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import numpy as np


class ToolClassifier(nn.Module):
    """
    Tool Classifier Head
    
    Selects a tool from the registry based on:
    - LLM hidden states (task understanding)
    - Tool embeddings (tool capabilities)
    - Optional: temporal embedding (resource constraints)
    
    Architecture:
        ┌─────────────────────────────────────────────────────────┐
        │  Inputs:                                                 │
        │  - hidden_states: (batch, hidden_dim)                   │
        │  - tool_embeddings: (num_tools, tool_dim)               │
        │  - temporal_emb: (temporal_dim) [optional]              │
        └──────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Attention Pooling  │
                    │  (Query: hidden)    │
                    │  (Keys: tools)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  MLP Classifier     │
                    │  hidden → hidden/2  │
                    │  → num_tools        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Softmax            │
                    │  → tool_probs       │
                    └─────────────────────┘
    
    Training:
        - Loss: CrossEntropyLoss(predicted_tool_id, target_tool_id)
        - Data: (task_description, target_tool_id) pairs
    
    Inference:
        - tool_id = argmax(tool_probs)
        - confidence = max(tool_probs)
    
    Usage:
        classifier = ToolClassifier(
            hidden_dim=3584,
            tool_dim=1024,
            num_tools=50
        )
        
        # During inference at <TOOL_ID> position
        tool_logits = classifier(
            hidden_states,      # (batch, 3584)
            tool_embeddings     # (50, 1024)
        )
        tool_probs = torch.softmax(tool_logits, dim=-1)  # (batch, 50)
        tool_id = tool_probs.argmax(-1)  # (batch,)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        tool_dim: int,
        num_tools: int,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: LLM hidden dimension (896 for 0.5B, 3584 for 7B)
            tool_dim: Tool embedding dimension (1024 from Section 1.2)
            num_tools: Number of tools in registry
            use_attention: Whether to use attention pooling over tools
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.tool_dim = tool_dim
        self.num_tools = num_tools
        self.use_attention = use_attention
        
        # Project tool embeddings to hidden dimension
        self.tool_proj = nn.Linear(tool_dim, hidden_dim)
        
        if use_attention:
            # Attention-based tool selection
            # Query: LLM hidden state
            # Keys/Values: Tool embeddings
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            
            # Post-attention MLP
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_tools),
            )
        else:
            # Simple MLP classifier
            # Concatenate: [hidden_state, avg_tool_embedding]
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_tools),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        tool_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, hidden_dim) - LLM hidden at <TOOL_ID> position
            tool_embeddings: (num_tools, tool_dim) - Tool embeddings from registry
            temporal_emb: (temporal_dim,) - Temporal resource embedding [optional]
            tool_mask: (num_tools,) - Boolean mask for valid tools [optional]
        
        Returns:
            logits: (batch, num_tools) - Logits for each tool
        """
        batch_size = hidden_states.size(0)
        
        # Project tool embeddings to hidden dimension
        tool_hidden = self.tool_proj(tool_embeddings)  # (num_tools, hidden_dim)
        
        if self.use_attention:
            # Attention pooling
            # Query: hidden_states (batch, 1, hidden_dim)
            # Key/Value: tool_hidden (batch, num_tools, hidden_dim)
            
            query = hidden_states.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # Expand tools to batch
            key_value = tool_hidden.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_tools, hidden_dim)
            
            # Attention
            attn_out, attn_weights = self.attention(
                query, key_value, key_value,
                key_padding_mask=None if tool_mask is None else ~tool_mask.unsqueeze(0).expand(batch_size, -1)
            )  # attn_out: (batch, 1, hidden_dim)
            
            attn_out = attn_out.squeeze(1)  # (batch, hidden_dim)
            
            # Classify
            logits = self.classifier(attn_out)  # (batch, num_tools)
        
        else:
            # Simple concatenation
            # Average tool embeddings
            avg_tool = tool_hidden.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            avg_tool = avg_tool.expand(batch_size, -1)  # (batch, hidden_dim)
            
            # Concatenate
            combined = torch.cat([hidden_states, avg_tool], dim=-1)  # (batch, 2*hidden_dim)
            
            # Classify
            logits = self.classifier(combined)  # (batch, num_tools)
        
        # Apply tool mask (set invalid tools to -inf)
        if tool_mask is not None:
            logits = logits.masked_fill(~tool_mask.unsqueeze(0), float('-inf'))
        
        return logits
    
    def predict(
        self,
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        temporal_emb: Optional[torch.Tensor] = None,  # Kept for backward compatibility, not used
        tool_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict tool ID with confidence.
        
        Args:
            hidden_states: (batch, hidden_dim)
            tool_embeddings: (num_tools, tool_dim)
            temporal_emb: (temporal_dim,) [optional, for backward compatibility, not used]
            tool_mask: (num_tools,) [optional]
            return_probs: Whether to return full probability distribution
        
        Returns:
            predictions: Dict containing:
                - 'tool_id': (batch,) - Predicted tool indices
                - 'confidence': (batch,) - Confidence scores (max probability)
                - 'probs': (batch, num_tools) - Full probability distribution [if return_probs=True]
        """
        # Note: temporal_emb is ignored (kept for backward compatibility)
        logits = self.forward(hidden_states, tool_embeddings, tool_mask)
        probs = torch.softmax(logits, dim=-1)  # (batch, num_tools)
        
        confidence, tool_id = probs.max(dim=-1)  # (batch,)
        
        predictions = {
            'tool_id': tool_id,
            'confidence': confidence,
        }
        
        if return_probs:
            predictions['probs'] = probs
        
        return predictions
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        tool_embeddings: torch.Tensor,
        target_tool_ids: torch.Tensor,
        tool_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for tool classification.
        
        Args:
            hidden_states: (batch, hidden_dim)
            tool_embeddings: (num_tools, tool_dim)
            target_tool_ids: (batch,) - Ground truth tool indices
            temporal_emb: (temporal_dim,) [optional]
            tool_mask: (num_tools,) [optional]
        
        Returns:
            loss: Scalar tensor
        """
        logits = self.forward(hidden_states, tool_embeddings, tool_mask)
        loss = nn.functional.cross_entropy(logits, target_tool_ids)
        return loss
    
    def __repr__(self):
        return (
            f"ToolClassifier(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  tool_dim={self.tool_dim},\n"
            f"  num_tools={self.num_tools},\n"
            f"  use_attention={self.use_attention}\n"
            f")"
        )


class ToolClassifierWithRegistry(nn.Module):
    """
    Tool Classifier with integrated tool registry.
    
    Loads tool embeddings from registry and maintains tool metadata.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        tool_dim: int,
        tool_registry_path: Optional[str] = None,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: LLM hidden dimension
            tool_dim: Tool embedding dimension
            tool_registry_path: Path to tool registry JSON (optional)
            use_attention: Whether to use attention pooling
            dropout: Dropout probability
        """
        super().__init__()
        
        # Load tool registry
        if tool_registry_path is not None:
            self.tool_names, self.tool_embeddings = self._load_tool_registry(tool_registry_path)
            num_tools = len(self.tool_names)
        else:
            # Dummy registry for testing
            num_tools = 10
            self.tool_names = [f"tool_{i}" for i in range(num_tools)]
            self.tool_embeddings = nn.Parameter(
                torch.randn(num_tools, tool_dim) * 0.02,
                requires_grad=False  # Fixed embeddings (loaded from registry)
            )
        
        # Classifier
        self.classifier = ToolClassifier(
            hidden_dim=hidden_dim,
            tool_dim=tool_dim,
            num_tools=num_tools,
            use_attention=use_attention,
            dropout=dropout,
        )
        
        # Tool ID to name mapping
        self.id_to_name = {i: name for i, name in enumerate(self.tool_names)}
        self.name_to_id = {name: i for i, name in enumerate(self.tool_names)}
    
    def _load_tool_registry(self, registry_path: str) -> Tuple[List[str], torch.Tensor]:
        """Load tool registry from JSON file."""
        import json
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        tool_names = [tool['name'] for tool in registry]
        
        # Tool embeddings should be precomputed (from Section 1.2)
        # For now, use random embeddings
        tool_embeddings = torch.randn(len(tool_names), self.classifier.tool_dim) * 0.02
        
        return tool_names, tool_embeddings
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temporal_emb: Optional[torch.Tensor] = None,
        tool_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using registry embeddings."""
        return self.classifier(
            hidden_states,
            self.tool_embeddings,
            temporal_emb,
            tool_mask
        )
    
    def predict_with_names(
        self,
        hidden_states: torch.Tensor,
        temporal_emb: Optional[torch.Tensor] = None,
        tool_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """Predict tool with name instead of ID."""
        predictions = self.classifier.predict(
            hidden_states,
            self.tool_embeddings,
            temporal_emb,
            tool_mask,
            return_probs=True
        )
        
        # Convert IDs to names
        tool_ids = predictions['tool_id'].cpu().numpy()
        tool_names = [self.id_to_name[int(tid)] for tid in tool_ids]
        
        return {
            'tool_name': tool_names,
            'tool_id': predictions['tool_id'],
            'confidence': predictions['confidence'],
            'probs': predictions['probs'],
        }
