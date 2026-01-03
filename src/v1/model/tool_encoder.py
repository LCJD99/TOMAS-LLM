import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class SemanticEncoder(nn.Module):
    """
    Deep semantic encoder using LLM forward pass (Stream A)
    
    Architecture:
    1. Construct prompt: "Tool: {name}\nDescription: {description}\n"
    2. Run full LLM forward pass with output_hidden_states=True
    3. Extract last layer's EOS token hidden state
    4. Store as precomputed buffer for efficient inference
    
    The encoder is frozen during training and serves as a semantic anchor.
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen2.5-7B",
        tool_registry_path: Optional[str] = None,
        cache_dir: Optional[str] = "hub",
        device: Optional[str] = None
    ):
        """
        Initialize semantic encoder.
        
        Args:
            llm_model_name: HuggingFace model identifier or local path
            tool_registry_path: Path to tools.json with tool descriptions
            cache_dir: Cache directory for model weights
            device: Device to use for LLM forward pass (default: auto-detect)
        """
        super().__init__()
        
        self.llm_model_name = llm_model_name
        self.tool_registry_path = tool_registry_path
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load kwargs
        import os
        is_local_path = os.path.isdir(llm_model_name)
        load_kwargs = {"trust_remote_code": True}
        if not is_local_path and cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        
        logger.info(f"Loading LLM for semantic encoding: {llm_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            **load_kwargs
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LLM model (frozen, only for precomputation)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            **load_kwargs
        )
        self.llm_model.eval()  # Always in eval mode
        self.llm_model.to(self.device)
        
        # Freeze all parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Get hidden dimension
        self.hidden_dim = self.llm_model.config.hidden_size
        
        logger.info(f"Semantic encoder initialized: hidden_dim={self.hidden_dim}")
        
        # Load tool registry and precompute embeddings
        if tool_registry_path:
            self._load_and_precompute(tool_registry_path)
        else:
            logger.warning("No tool_registry_path provided. Call load_and_precompute() manually.")
    
    def _load_tool_registry(self, registry_path: str) -> List[Dict]:
        """Load tool registry from JSON file."""
        registry_path = Path(registry_path)
        
        if not registry_path.exists():
            raise FileNotFoundError(f"Tool registry not found: {registry_path}")
        
        with open(registry_path, 'r') as f:
            tools = json.load(f)
        
        logger.info(f"Loaded {len(tools)} tools from {registry_path}")
        
        # Validate required fields
        for i, tool in enumerate(tools):
            if "name" not in tool:
                raise ValueError(f"Tool {i} missing 'name' field")
            if "description" not in tool:
                raise ValueError(f"Tool '{tool['name']}' missing 'description' field")
        
        return tools
    
    def _encode_single_tool(self, tool_name: str, tool_desc: str) -> torch.Tensor:
        """
        Encode a single tool using LLM forward pass.
        
        Args:
            tool_name: Tool name
            tool_desc: Tool description
        
        Returns:
            semantic_emb: [hidden_dim] - EOS token hidden state
        """
        # Construct prompt
        prompt = f"Tool: {tool_name}\nDescription: {tool_desc}\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Forward pass through LLM
        with torch.no_grad():
            outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Extract last layer hidden states
        last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        
        # Get EOS position (last non-padding token)
        seq_lengths = attention_mask.sum(dim=1) - 1  # [1]
        eos_position = seq_lengths[0].item()
        
        # Extract EOS token hidden state
        semantic_emb = last_hidden_state[0, eos_position, :]  # [hidden_dim]
        
        # Move to CPU for storage
        semantic_emb = semantic_emb.cpu()
        
        return semantic_emb
    
    def _load_and_precompute(self, registry_path: str):
        """
        Load tool registry and precompute all semantic embeddings.
        
        This is called during initialization and stores embeddings as a buffer.
        """
        logger.info("Precomputing tool semantic embeddings...")
        
        # Load tools
        tools = self._load_tool_registry(registry_path)
        
        # Store tool names for reference
        self.tool_names = [tool["name"] for tool in tools]
        self.num_tools = len(self.tool_names)
        
        # Precompute embeddings
        embeddings = []
        for tool in tools:
            semantic_emb = self._encode_single_tool(tool["name"], tool["description"])
            embeddings.append(semantic_emb)
            logger.info(f"  ✓ Encoded: {tool['name']}")
        
        # Stack and register as buffer
        embeddings_tensor = torch.stack(embeddings)  # [num_tools, hidden_dim]
        self.register_buffer("tool_semantic_embeddings", embeddings_tensor)

        self.llm_model.cpu()  # Free up GPU memory
        
        logger.info(f"Semantic embeddings precomputed: {embeddings_tensor.shape}")
        logger.info("Stream A (Semantic Encoder) ready for training.")
    
    def forward(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """
        Get precomputed semantic embeddings for given tool IDs.
        
        Args:
            tool_ids: [batch_size] - Tool indices
        
        Returns:
            semantic_emb: [batch_size, hidden_dim] - Precomputed embeddings
        """
        if not hasattr(self, "tool_semantic_embeddings"):
            raise RuntimeError(
                "Semantic embeddings not precomputed. "
                "Call load_and_precompute() or provide tool_registry_path during init."
            )
        
        return self.tool_semantic_embeddings[tool_ids]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of semantic embeddings."""
        return self.hidden_dim
    
    def get_num_tools(self) -> int:
        """Get the number of tools."""
        return self.num_tools
    
    def state_dict_without_llm(self) -> Dict:
        """
        Get state dict without LLM weights (only precomputed embeddings).
        
        This is useful for saving/loading without the large LLM model.
        """
        return {
            "tool_semantic_embeddings": self.tool_semantic_embeddings,
            "tool_names": self.tool_names,
            "num_tools": self.num_tools,
            "hidden_dim": self.hidden_dim,
            "llm_model_name": self.llm_model_name
        }
    
    def load_state_dict_without_llm(self, state_dict: Dict):
        """Load state dict (only precomputed embeddings)."""
        self.register_buffer(
            "tool_semantic_embeddings",
            state_dict["tool_semantic_embeddings"]
        )
        self.tool_names = state_dict["tool_names"]
        self.num_tools = state_dict["num_tools"]
        self.hidden_dim = state_dict["hidden_dim"]
        
        logger.info(f"Loaded semantic embeddings: {self.num_tools} tools")


class ResourceMLP(nn.Module):
    """
    MLP for projecting resource profiling features to high-dimensional latent space.
    
    Takes normalized 6D resource vectors and projects them to d_resource dimensions
    via a two-layer MLP with ReLU activation.
    
    Architecture:
        Input (6D) → Linear(6, hidden_dim) → ReLU → Linear(hidden_dim, d_resource) → Output (d_resource)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 512,
        d_resource: int = 256,
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        """
        Initialize Resource MLP.
        
        Args:
            input_dim: Input feature dimension (default: 6)
            hidden_dim: Hidden layer dimension (default: 512)
            d_resource: Output dimension for resource embeddings (default: 256)
            dropout: Dropout probability (default: 0.0, no dropout)
            use_batch_norm: Whether to use batch normalization (default: False)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_resource = d_resource
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # First layer: input_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Optional batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        
        # Activation
        self.activation = nn.ReLU()
        
        # Optional dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Second layer: hidden_dim -> d_resource
        self.fc2 = nn.Linear(hidden_dim, d_resource)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized ResourceMLP: {input_dim} -> {hidden_dim} -> {d_resource}, "
                   f"dropout={dropout}, batch_norm={use_batch_norm}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
               Expected to be normalized resource features
        
        Returns:
            Output tensor of shape (batch_size, d_resource) or (d_resource,)
        """
        # Handle both batched and unbatched inputs
        input_is_1d = x.dim() == 1
        if input_is_1d:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Validate input dimension
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(-1)}"
            )
        
        # First layer
        out = self.fc1(x)
        
        # Batch normalization (if enabled)
        if self.bn1 is not None:
            out = self.bn1(out)
        
        # Activation
        out = self.activation(out)
        
        # Dropout (if enabled)
        if self.dropout_layer is not None:
            out = self.dropout_layer(out)
        
        # Second layer
        out = self.fc2(out)
        
        # Remove batch dimension if input was 1D
        if input_is_1d:
            out = out.squeeze(0)
        
        return out
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ResourceMLP':
        """
        Create ResourceMLP from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'model.resource_mlp' section
        
        Returns:
            Initialized ResourceMLP instance
        """
        mlp_config = config['model']['resource_mlp']
        
        return cls(
            input_dim=mlp_config.get('input_features', 6),
            hidden_dim=mlp_config.get('hidden_dim', 512),
            d_resource=mlp_config.get('d_resource', 256),
            dropout=mlp_config.get('dropout', 0.0),
            use_batch_norm=mlp_config.get('use_batch_norm', False)
        )


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


class ResourceEncoderForPretraining(nn.Module):
    """
    Encoder for pre-training with self-supervised task.
    
    Learns to map (Tool ID + Resource Vector) to an embedding that enables
    the LLM to generate natural language descriptions of the configuration.
    
    New architecture:
    - Stream A: Deep semantic encoding via LLM forward pass
    - Stream B: Trainable resource MLP
    - Fusion: Gated cross-attention (α initialized to 0)
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen2.5-7B-Instruct",
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
        semantic_emb = self.semantic_encoder(tool_ids)
        
        # Stream B: Resource encoding
        resource_emb = self.resource_mlp(resource_vectors)
        
        # Fusion
        fused_emb = self.fusion(semantic_emb, resource_emb)  # [batch, hidden]
        
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
        llm_model_name="Qwen2.5-7B-Instruct",
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
        semantic_emb = encoder.semantic_encoder(tool_ids)
        
        print(f"Semantic embedding shape: {semantic_emb.shape}")
        
        # Test semantic quality
        cos_sim = torch.cosine_similarity(semantic_emb[0], semantic_emb[1], dim=0)
        print(f"Semantic similarity (tool 0 vs 1): {cos_sim:.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_pretrain_encoder()
