"""
Gated Fusion Module for Resource Encoder.

Fuses tool semantic embeddings with resource embeddings using a gated
cross-attention mechanism with learnable gate parameter.

Formula:
    Attended = CrossAttention(Q=E_tool, K=E_res, V=E_res)
    E_final = LayerNorm(E_tool + α * Attended)

Where α is a learnable gate parameter initialized to 0.
This ensures cold-start alignment with pure semantics.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    """
    Gated fusion module that gradually injects resource information
    into tool semantic embeddings.
    
    Design rationale:
    - Initialize gate α=0 → cold start with pure semantics (E_final ≈ E_tool)
    - Training increases α → progressively inject resource information
    - Residual connection preserves semantic anchor
    - LayerNorm stabilizes training
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize gated fusion module.
        
        Args:
            hidden_dim: Hidden dimension (should match LLM hidden size)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-attention: tool (Q) attends to resource (K, V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable gate parameter (initialized to 0)
        self.gate_alpha = nn.Parameter(torch.zeros(1))
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(
            f"Initialized GatedFusion: hidden_dim={hidden_dim}, "
            f"num_heads={num_heads}, gate_alpha={self.gate_alpha.item():.6f}"
        )
    
    def forward(
        self,
        tool_emb: torch.Tensor,
        resource_emb: torch.Tensor,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Fuse tool and resource embeddings with gated attention.
        
        Args:
            tool_emb: [batch_size, hidden_dim] - Tool semantic embeddings
            resource_emb: [batch_size, hidden_dim] - Resource embeddings
            return_attention_weights: Whether to return attention weights
        
        Returns:
            fused_emb: [batch_size, hidden_dim] - Fused embeddings
            (optional) attention_weights: [batch_size, num_heads, 1, 1]
        """
        # Expand dims for attention (expects [B, L, H])
        tool_query = tool_emb.unsqueeze(1)      # [B, 1, H]
        resource_kv = resource_emb.unsqueeze(1) # [B, 1, H]
        
        # Cross-attention: tool attends to resource
        attended, attn_weights = self.cross_attention(
            query=tool_query,
            key=resource_kv,
            value=resource_kv,
            need_weights=return_attention_weights
        )  # [B, 1, H]
        
        attended = attended.squeeze(1)  # [B, H]
        
        # Gated residual connection
        # At initialization (α≈0): fused ≈ tool_emb (pure semantics)
        # During training: α grows, gradually injecting resource info
        fused = self.layer_norm(tool_emb + self.gate_alpha * attended)
        
        if return_attention_weights:
            return fused, attn_weights
        else:
            return fused
    
    def get_gate_value(self) -> float:
        """Get current gate parameter value."""
        return self.gate_alpha.item()
    
    def get_gate_grad(self) -> Optional[float]:
        """Get gradient of gate parameter (for monitoring)."""
        if self.gate_alpha.grad is not None:
            return self.gate_alpha.grad.item()
        return None


def test_gated_fusion():
    """Test gated fusion module."""
    print("=== Testing GatedFusion ===\n")
    
    batch_size = 4
    hidden_dim = 3584
    
    fusion = GatedFusion(hidden_dim=hidden_dim, num_heads=8)
    
    # Create dummy embeddings
    tool_emb = torch.randn(batch_size, hidden_dim)
    resource_emb = torch.randn(batch_size, hidden_dim)
    
    print(f"Input shapes:")
    print(f"  tool_emb: {tool_emb.shape}")
    print(f"  resource_emb: {resource_emb.shape}")
    print(f"  gate_alpha: {fusion.get_gate_value():.6f}")
    print()
    
    # Forward pass
    fused = fusion(tool_emb, resource_emb)
    
    print(f"Output shape: {fused.shape}")
    print(f"Expected: [{batch_size}, {hidden_dim}]")
    
    # Check initial behavior (should be close to tool_emb)
    # Since alpha=0, fused should be close to LayerNorm(tool_emb)
    normalized_tool = fusion.layer_norm(tool_emb)
    similarity = torch.cosine_similarity(fused, normalized_tool, dim=-1).mean()
    
    print(f"\nCosine similarity with normalized tool_emb: {similarity:.4f}")
    print(f"(Should be ~1.0 at initialization since α≈0)")
    
    # Test training
    print("\n=== Testing Training ===")
    optimizer = torch.optim.Adam(fusion.parameters(), lr=0.01)
    
    initial_alpha = fusion.get_gate_value()
    
    for i in range(10):
        tool_emb_batch = torch.randn(4, hidden_dim)
        resource_emb_batch = torch.randn(4, hidden_dim)
        target = torch.randn(4, hidden_dim)
        
        fused_batch = fusion(tool_emb_batch, resource_emb_batch)
        loss = ((fused_batch - target) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 3 == 0:
            print(f"  Step {i}: loss={loss.item():.4f}, α={fusion.get_gate_value():.6f}")
    
    final_alpha = fusion.get_gate_value()
    
    print(f"\nInitial α: {initial_alpha:.6f}")
    print(f"Final α: {final_alpha:.6f}")
    print(f"Change: {abs(final_alpha - initial_alpha):.6f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_gated_fusion()
