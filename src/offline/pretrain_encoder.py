"""
Resource Encoder for Pre-training.

This encoder learns to compress 6D resource vectors into LLM-understandable embeddings
through self-supervised language modeling.

Architecture:
1. Stream A (Semantic): Frozen Qwen2.5 embedding layer for tool names
2. Stream B (Resource): Trainable MLP for resource vectors
3. Fusion: Trainable self-attention to combine streams
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from src.encoders.resource_mlp import ResourceMLP


class ResourceEncoderForPretraining(nn.Module):
    """
    Encoder for pre-training with self-supervised task.
    
    Learns to map (Tool ID + Resource Vector) to an embedding that enables
    the LLM to generate natural language descriptions of the configuration.
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B",
        llm_hidden_dim: int = 3584,
        d_resource: int = 3584,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        num_tools: int = 7,
        freeze_semantic: bool = True,
        cache_dir: str = "hub"
    ):
        """
        Initialize the pre-training encoder.
        
        Args:
            llm_model_name: HuggingFace model identifier for semantic stream
            llm_hidden_dim: Hidden dimension of LLM (3584 for Qwen2.5-7B)
            d_resource: Output dimension for resource MLP
            num_attention_heads: Number of heads in fusion attention
            dropout: Dropout probability
            num_tools: Number of tools (for tool embedding)
            freeze_semantic: Whether to freeze Stream A (semantic)
            cache_dir: Cache directory for models
        """
        super().__init__()
        
        self.llm_hidden_dim = llm_hidden_dim
        self.d_resource = d_resource
        self.freeze_semantic = freeze_semantic
        
        # Determine if loading from local path or HuggingFace Hub
        # If llm_model_name is a local directory, don't use cache_dir
        import os
        is_local_path = os.path.isdir(llm_model_name)
        load_kwargs = {
            "trust_remote_code": True,
        }
        if not is_local_path and cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        
        # ===== Stream A: Tool Semantic Encoding (Frozen) =====
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            **load_kwargs
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load embedding layer from pretrained LLM
        llm_model = AutoModel.from_pretrained(
            llm_model_name,
            **load_kwargs
        )
        
        # Extract and freeze embedding layer
        self.semantic_embedding = llm_model.get_input_embeddings()
        
        if freeze_semantic:
            for param in self.semantic_embedding.parameters():
                param.requires_grad = False
        
        # Create tool name lookup (tool_id -> tool_name)
        self.tool_names = [
            "image_classification",
            "text_summarization", 
            "image_captioning",
            "object_detection",
            "machine_translation",
            "super_resolution",
            "visual_question_answering"
        ]
        
        # Pre-tokenize tool names for efficiency
        self._precompute_tool_tokens()
        
        # ===== Stream B: Resource Encoding (Trainable) =====
        self.resource_mlp = ResourceMLP(
            d_resource=d_resource,
            dropout=dropout
        )
        
        # ===== Fusion: Self-Attention (Trainable) =====
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=llm_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(llm_hidden_dim)
        
        # Initialize trainable parameters
        self._init_weights()
    
    def _precompute_tool_tokens(self):
        """Pre-tokenize all tool names and store as buffers."""
        tool_token_ids = []
        
        for tool_name in self.tool_names:
            # Tokenize tool name
            tokens = self.tokenizer(
                tool_name,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"]
            tool_token_ids.append(tokens.squeeze(0))
        
        # Store as padded tensor
        max_len = max(len(t) for t in tool_token_ids)
        padded_tokens = torch.zeros(len(self.tool_names), max_len, dtype=torch.long)
        
        for i, tokens in enumerate(tool_token_ids):
            padded_tokens[i, :len(tokens)] = tokens
        
        # Register as buffer (moves to device automatically)
        self.register_buffer("tool_token_ids", padded_tokens)
    
    def _init_weights(self):
        """Initialize trainable parameters (Xavier/He init)."""
        # ResourceMLP is already initialized in its __init__
        
        # Initialize fusion attention (already done by PyTorch)
        # Just ensure layer norm is properly initialized
        nn.init.ones_(self.fusion_norm.weight)
        nn.init.zeros_(self.fusion_norm.bias)
    
    def encode_semantic(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode tool semantics using frozen LLM embeddings.
        
        Args:
            tool_ids: Tensor of shape [batch_size]
        
        Returns:
            Semantic embeddings of shape [batch_size, llm_hidden_dim]
        """
        batch_size = tool_ids.size(0)
        device = tool_ids.device
        
        # Get token IDs for each tool
        tool_tokens = self.tool_token_ids[tool_ids]  # [batch, max_token_len]
        
        # Get embeddings
        with torch.set_grad_enabled(not self.freeze_semantic):
            token_embeddings = self.semantic_embedding(tool_tokens)  # [batch, max_token_len, hidden]
        
        # Mean pooling over token dimension
        # Note: This is simple mean pooling. For better quality, could use attention mask.
        semantic_emb = token_embeddings.mean(dim=1)  # [batch, hidden]
        
        return semantic_emb
    
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
        Fuse semantic and resource streams using self-attention.
        
        Args:
            semantic_emb: [batch_size, hidden_dim]
            resource_emb: [batch_size, hidden_dim]
        
        Returns:
            Fused embedding [batch_size, hidden_dim]
        """
        # Stack to create sequence: [semantic, resource]
        stacked = torch.stack([semantic_emb, resource_emb], dim=1)  # [batch, 2, hidden]
        
        # Self-attention
        attended, _ = self.fusion_attention(
            stacked, stacked, stacked,
            need_weights=False
        )  # [batch, 2, hidden]
        
        # Residual connection + LayerNorm
        fused = self.fusion_norm(attended + stacked)  # [batch, 2, hidden]
        
        # Take first token as final embedding
        final_emb = fused[:, 0, :]  # [batch, hidden]
        
        return final_emb
    
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
        
        # Resource MLP parameters
        trainable_params.extend(self.resource_mlp.parameters())
        
        # Fusion attention parameters
        trainable_params.extend(self.fusion_attention.parameters())
        trainable_params.extend(self.fusion_norm.parameters())
        
        return trainable_params


def test_pretrain_encoder():
    """Test the pre-training encoder."""
    print("=== Testing ResourceEncoderForPretraining ===\n")
    
    # Initialize encoder
    encoder = ResourceEncoderForPretraining(
        llm_model_name="Qwen/Qwen2.5-7B-Instruct",
        freeze_semantic=True
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    
    print(f"Device: {device}")
    print(f"Model initialized with {sum(p.numel() for p in encoder.parameters()):,} total parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in encoder.get_trainable_parameters()):,}")
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
    print(f"Expected: [batch_size={batch_size}, hidden_dim=3584]")
    print()
    
    # Test individual components
    print("=== Testing Individual Components ===")
    
    with torch.no_grad():
        semantic_emb = encoder.encode_semantic(tool_ids)
        resource_emb = encoder.encode_resource(resource_vectors)
        
        print(f"Semantic embedding shape: {semantic_emb.shape}")
        print(f"Resource embedding shape: {resource_emb.shape}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_pretrain_encoder()
