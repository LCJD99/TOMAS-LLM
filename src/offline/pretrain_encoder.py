"""
Resource Encoder for Pre-training (Redesigned).

This encoder learns to compress 6D resource vectors into LLM-understandable embeddings
through self-supervised language modeling.

Architecture (NEW):
1. Stream A (Semantic): Deep LLM forward pass with EOS token extraction (frozen)
2. Stream B (Resource): Trainable MLP for resource vectors
3. Fusion: Gated cross-attention with learnable gate parameter

Key improvements:
- Stream A uses full transformer layers (not just embedding)
- Gated fusion ensures cold-start semantic alignment
- Precomputed semantic embeddings for efficient training
"""

import torch
import torch.nn as nn
from typing import Optional

from src.encoders.semantic_encoder import SemanticEncoder
from src.encoders.resource_mlp import ResourceMLP
from src.encoders.gated_fusion import GatedFusion


class ResourceEncoderForPretraining(nn.Module):
    """
    Encoder for pre-training with self-supervised task (REDESIGNED).
    
    Learns to map (Tool ID + Resource Vector) to an embedding that enables
    the LLM to generate natural language descriptions of the configuration.
    
    New architecture:
    - Stream A: Deep semantic encoding via LLM forward pass
    - Stream B: Trainable resource MLP
    - Fusion: Gated cross-attention (α initialized to 0)
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B",
        tool_registry_path: str = "data/tool_registry/tools.json",
        d_resource: Optional[int] = None,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        cache_dir: str = "hub"
    ):
        """
        Initialize the pre-training encoder.
        
        Args:
            llm_model_name: HuggingFace model identifier for semantic stream
            tool_registry_path: Path to tools.json with descriptions
            d_resource: Output dimension for resource MLP (default: match LLM hidden dim)
            num_attention_heads: Number of heads in fusion attention
            dropout: Dropout probability
            cache_dir: Cache directory for models
        """
        super().__init__()
        
        # ===== Stream A: Deep Semantic Encoding (Frozen) =====
        self.semantic_encoder = SemanticEncoder(
            llm_model_name=llm_model_name,
            tool_registry_path=tool_registry_path,
            cache_dir=cache_dir
        )
        
        # Get hidden dimension from semantic encoder
        self.llm_hidden_dim = self.semantic_encoder.get_embedding_dim()
        
        # Set d_resource to match hidden_dim if not specified
        if d_resource is None:
            d_resource = self.llm_hidden_dim
        
        self.d_resource = d_resource
        
        # ===== Stream B: Resource Encoding (Trainable) =====
        self.resource_mlp = ResourceMLP(
            input_dim=6,
            hidden_dim=512,
            d_resource=d_resource,
            dropout=dropout
        )
        
        # ===== Fusion: Gated Cross-Attention (Trainable) =====
        self.fusion = GatedFusion(
            hidden_dim=self.llm_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Store metadata
        self.tool_names = self.semantic_encoder.tool_names
        self.num_tools = self.semantic_encoder.get_num_tools()
    
    def encode_semantic(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode tool semantics using precomputed deep LLM embeddings.
        
        Args:
            tool_ids: Tensor of shape [batch_size]
        
        Returns:
            Semantic embeddings of shape [batch_size, llm_hidden_dim]
        """
        return self.semantic_encoder(tool_ids)
    
    def encode_resource(self, resource_vectors: torch.Tensor) -> torch.Tensor:
        """
        Encode resource vectors using trainable MLP.
        
        Args:
            resource_vectors: Tensor of shape [batch_size, 6]
        
        Returns:
            Resource embeddings of shape [batch_size, d_resource]
        """
        return self.resource_mlp(resource_vectors)
    
    def fuse_streams(
        self, 
        semantic_emb: torch.Tensor, 
        resource_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse semantic and resource streams using gated cross-attention.
        
        Args:
            semantic_emb: [batch_size, hidden_dim]
            resource_emb: [batch_size, hidden_dim]
        
        Returns:
            Fused embedding [batch_size, hidden_dim]
        """
        return self.fusion(semantic_emb, resource_emb)
    
    def forward(
        self, 
        tool_ids: torch.Tensor, 
        resource_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: encode tool + resource into unified embedding.
        
        Args:
            tool_ids: [batch_size] - Tool indices (0-6)
            resource_vectors: [batch_size, 6] - Resource features
        
        Returns:
            Fused embeddings [batch_size, llm_hidden_dim]
        """
        # Stream A: Semantic encoding
        semantic_emb = self.encode_semantic(tool_ids)  # [batch, hidden]
        
        # Stream B: Resource encoding
        resource_emb = self.encode_resource(resource_vectors)  # [batch, hidden]
        
        # Fusion
        fused_emb = self.fuse_streams(semantic_emb, resource_emb)  # [batch, hidden]
        
        return fused_emb
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (Stream B + Fusion)."""
        trainable_params = []
        
        # Resource MLP parameters (Stream B)
        trainable_params.extend(self.resource_mlp.parameters())
        
        # Gated fusion parameters
        trainable_params.extend(self.fusion.parameters())
        
        return trainable_params
    
    def get_gate_value(self) -> float:
        """Get current gate parameter value from fusion module."""
        return self.fusion.get_gate_value()


def test_pretrain_encoder():
    """Test the pre-training encoder."""
    print("=== Testing ResourceEncoderForPretraining (Redesigned) ===\n")
    
    # Initialize encoder with small model for testing
    encoder = ResourceEncoderForPretraining(
        llm_model_name="Qwen/Qwen2.5-0.5B",
        tool_registry_path="data/tool_registry/tools.json"
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    
    print(f"Device: {device}")
    print(f"Number of tools: {encoder.num_tools}")
    print(f"Hidden dimension: {encoder.llm_hidden_dim}")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in encoder.get_trainable_parameters()):,}")
    print(f"Gate alpha: {encoder.get_gate_value():.6f}")
    print()
    
    # Create dummy batch
    batch_size = 4
    tool_ids = torch.tensor([0, 1, 2, 3], device=device)  # Different tools
    resource_vectors = torch.randn(batch_size, 6, device=device)
    
    print(f"Input shapes:")
    print(f"  tool_ids: {tool_ids.shape}")
    print(f"  resource_vectors: {resource_vectors.shape}")
    print()
    
    # Forward pass
    with torch.no_grad():
        embeddings = encoder(tool_ids, resource_vectors)
    
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: [batch_size={batch_size}, hidden_dim={encoder.llm_hidden_dim}]")
    print()
    
    # Test individual components
    print("=== Testing Individual Components ===")
    
    with torch.no_grad():
        semantic_emb = encoder.encode_semantic(tool_ids)
        resource_emb = encoder.encode_resource(resource_vectors)
        
        print(f"Semantic embedding shape: {semantic_emb.shape}")
        print(f"Resource embedding shape: {resource_emb.shape}")
        
        # Test semantic quality
        cos_sim = torch.cosine_similarity(semantic_emb[0], semantic_emb[1], dim=0)
        print(f"Semantic similarity (tool 0 vs 1): {cos_sim:.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_pretrain_encoder()
