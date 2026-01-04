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

class NaiveEncoderForPretraining(nn.Module):
    """
    Naive encoder that extends LLM vocabulary with tool configuration tokens.
    
    Architecture:
    - Base LLM: Qwen2.5-7B (frozen, unchanged)
    - New Token Embeddings: Separate embedding layer for new tokens
    - New Token LM Head: Separate lm_head for new tokens
    
    Training Strategy:
    - Keep base LLM completely frozen and unmodified
    - Only train new token embeddings and lm_head modules
    - Combine outputs via concatenation
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen2.5-7B",
        extended_tokenizer_path: Optional[str] = None,
        combined_tokens_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the NaiveEncoderForPretraining model.
        
        Args:
            llm_model_name: Name or path of the base LLM model
            extended_tokenizer_path: Path to the extended tokenizer directory
            combined_tokens_path: Path to the combined_tokens.json file
            device: Device to load the model on
        """
        super().__init__()
        
        self.device = device
        self.llm_model_name = llm_model_name
        
        # Load the base LLM model (keep frozen and unmodified)
        logger.info(f"Loading base LLM model: {llm_model_name}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        # Freeze all base model parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Load extended tokenizer
        if extended_tokenizer_path is None:
            extended_tokenizer_path = "data/generated/extended_tokenizer"
        
        logger.info(f"Loading extended tokenizer from: {extended_tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # Load combined tokens metadata
        if combined_tokens_path is None:
            combined_tokens_path = "data/generated/combined_tokens.json"
        
        logger.info(f"Loading combined tokens from: {combined_tokens_path}")
        with open(combined_tokens_path, "r") as f:
            token_data = json.load(f)
            self.combined_tokens = token_data["tokens"]
            self.num_new_tokens = token_data["num_new_tokens"]
        
        # Record vocabulary sizes
        self.original_vocab_size = self.llm_model.config.vocab_size
        self.extended_vocab_size = self.original_vocab_size + self.num_new_tokens
        
        logger.info(f"Original vocabulary size: {self.original_vocab_size}")
        logger.info(f"Extended vocabulary size: {self.extended_vocab_size}")
        logger.info(f"Number of new tokens: {self.num_new_tokens}")
        
        # Get hidden dimension from base model
        self.hidden_dim = self.llm_model.config.hidden_size
        
        # Create separate modules for new tokens
        self._create_new_token_modules()
        
        # Log trainable parameters
        self._log_trainable_parameters()
    
    def _create_new_token_modules(self):
        """
        Create separate embedding and lm_head modules for new tokens.
        
        Strategy:
        - Create embedding layer with num_new_tokens rows
        - Initialize from constituent token embeddings (warm start)
        - Create lm_head layer for new tokens
        """
        logger.info("Creating new token modules...")
        
        # Get original embedding layer for initialization
        original_embeddings = self.llm_model.get_input_embeddings()
        original_embedding_weight = original_embeddings.weight.data
        
        # Initialize new token embeddings
        new_embeddings_init = []
        for combined_token in self.combined_tokens:
            # Tokenize the combined token using the original vocabulary
            token_ids = self.tokenizer.encode(
                combined_token,
                add_special_tokens=False,
            )
            
            # Get embeddings for constituent tokens
            constituent_embeddings = original_embedding_weight[token_ids]
            
            # Initialize as mean of constituent embeddings
            new_embedding = constituent_embeddings.mean(dim=0)
            new_embeddings_init.append(new_embedding)
        
        # Stack into initialization tensor
        new_embeddings_init = torch.stack(new_embeddings_init, dim=0)  # [num_new_tokens, hidden_dim]
        
        # Create new token embedding layer
        self.new_token_embeddings = nn.Embedding(
            num_embeddings=self.num_new_tokens,
            embedding_dim=self.hidden_dim,
        )
        # Initialize with computed values
        self.new_token_embeddings.weight.data = new_embeddings_init.to(self.new_token_embeddings.weight.device)
        
        # Create new token lm_head
        self.new_token_lm_head = nn.Linear(
            self.hidden_dim,
            self.num_new_tokens,
            bias=self.llm_model.lm_head.bias is not None,
        )
        # Initialize from new embeddings (weight tying)
        self.new_token_lm_head.weight.data = new_embeddings_init.clone().to(self.new_token_lm_head.weight.device)
        if self.new_token_lm_head.bias is not None:
            nn.init.zeros_(self.new_token_lm_head.bias)
        
        logger.info(f"New token embedding created: {self.new_token_embeddings.weight.shape}")
        logger.info(f"New token lm_head created: {self.new_token_lm_head.weight.shape}")
    
    def _log_trainable_parameters(self):
        """
        Log the number of trainable parameters in the model.
        """
        # Base model parameters (all frozen)
        base_params = sum(p.numel() for p in self.llm_model.parameters())
        
        # New token module parameters (all trainable)
        new_emb_params = sum(p.numel() for p in self.new_token_embeddings.parameters())
        new_lm_params = sum(p.numel() for p in self.new_token_lm_head.parameters())
        trainable_params = new_emb_params + new_lm_params
        
        total_params = base_params + trainable_params
        
        logger.info(f"Base LLM parameters (frozen): {base_params:,}")
        logger.info(f"New token embeddings: {new_emb_params:,}")
        logger.info(f"New token lm_head: {new_lm_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass through the model.
        
        Strategy:
        - Separate input_ids into original tokens and new tokens
        - Process original tokens through base LLM
        - Process new tokens through new token embeddings
        - Combine embeddings and pass through transformer
        - Combine logits from base lm_head and new token lm_head
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs for language modeling [batch_size, seq_len]
                   Use -100 for tokens that should not contribute to loss
            **kwargs: Additional arguments passed to the LLM model
        
        Returns:
            ModelOutput containing loss, logits, etc.
        """
        batch_size, seq_len = input_ids.shape
        
        # Create mask for new tokens
        new_token_mask = input_ids >= self.original_vocab_size  # [batch_size, seq_len]
        
        # Get embeddings
        # For original tokens: use base model embeddings
        # For new tokens: use new token embeddings
        base_embeddings = self.llm_model.get_input_embeddings()
        
        # Initialize combined embeddings
        combined_embeds = torch.zeros(
            batch_size, seq_len, self.hidden_dim,
            dtype=base_embeddings.weight.dtype,
            device=input_ids.device
        )
        
        # Get original token embeddings (clamp to avoid index errors)
        original_input_ids = torch.clamp(input_ids, max=self.original_vocab_size - 1)
        original_embeds = base_embeddings(original_input_ids)
        
        # Get new token embeddings
        new_token_indices = input_ids - self.original_vocab_size  # Offset to 0-based indexing
        new_token_indices = torch.clamp(new_token_indices, min=0)  # Avoid negative indices
        new_embeds = self.new_token_embeddings(new_token_indices)
        
        # Combine embeddings based on mask
        # Use weighted sum instead of torch.where to maintain gradient flow
        mask_float = new_token_mask.unsqueeze(-1).float()
        combined_embeds = mask_float * new_embeds + (1 - mask_float) * original_embeds
        
        # Pass through transformer (without lm_head)
        outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Get logits from both heads and concatenate
        base_logits = self.llm_model.lm_head(hidden_states)  # [batch_size, seq_len, original_vocab_size]
        new_logits = self.new_token_lm_head(hidden_states)   # [batch_size, seq_len, num_new_tokens]
        
        # Concatenate logits
        logits = torch.cat([base_logits, new_logits], dim=-1)  # [batch_size, seq_len, extended_vocab_size]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.extended_vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = loss_fct(shift_logits, shift_labels)
        
        # Return in HuggingFace format
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states if kwargs.get("output_hidden_states") else None,
            attentions=outputs.attentions if kwargs.get("output_attentions") else None,
        )
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            **kwargs: Additional arguments passed to the generate method
        
        Returns:
            Generated token IDs
        """
        # Override the forward method temporarily for generation
        original_forward = self.llm_model.forward
        
        def custom_forward(input_ids=None, attention_mask=None, inputs_embeds=None, **gen_kwargs):
            if inputs_embeds is None:
                # Get embeddings using our custom logic
                batch_size, seq_len = input_ids.shape
                new_token_mask = input_ids >= self.original_vocab_size
                
                base_embeddings = self.llm_model.get_input_embeddings()
                combined_embeds = torch.zeros(
                    batch_size, seq_len, self.hidden_dim,
                    dtype=base_embeddings.weight.dtype,
                    device=input_ids.device
                )
                
                original_input_ids = torch.clamp(input_ids, max=self.original_vocab_size - 1)
                original_embeds = base_embeddings(original_input_ids)
                
                new_token_indices = input_ids - self.original_vocab_size
                new_token_indices = torch.clamp(new_token_indices, min=0)
                new_embeds = self.new_token_embeddings(new_token_indices)
                
                # Use weighted sum to maintain gradient flow
                mask_float = new_token_mask.unsqueeze(-1).float()
                inputs_embeds = mask_float * new_embeds + (1 - mask_float) * original_embeds
            
            # Forward through transformer
            outputs = original_forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **gen_kwargs
            )
            
            # Combine logits
            hidden_states = outputs.hidden_states[-1]
            base_logits = self.llm_model.lm_head(hidden_states)
            new_logits = self.new_token_lm_head(hidden_states)
            outputs.logits = torch.cat([base_logits, new_logits], dim=-1)
            
            return outputs
        
        # Temporarily replace forward
        self.llm_model.forward = custom_forward
        
        try:
            generated = self.llm_model.generate(input_ids=input_ids, **kwargs)
        finally:
            # Restore original forward
            self.llm_model.forward = original_forward
        
        return generated
    
    def save_new_parameters(self, save_path: str):
        """
        Save only the new parameters (embeddings and LM head extensions).
        This creates a lightweight checkpoint.
        
        Args:
            save_path: Path to save the new parameters
        """
        import os
        logger.info(f"Saving new parameters to: {save_path}")
        
        # Save new token modules
        new_params = {
            "new_token_embeddings": self.new_token_embeddings.state_dict(),
            "new_token_lm_head": self.new_token_lm_head.state_dict(),
            "num_new_tokens": self.num_new_tokens,
            "original_vocab_size": self.original_vocab_size,
            "hidden_dim": self.hidden_dim,
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save
        torch.save(new_params, save_path)
        
        # Log file size
        file_size_mb = os.path.getsize(save_path) / 1024 / 1024
        logger.info(f"Saved new parameters: {file_size_mb:.2f} MB")
    
    def load_new_parameters(self, load_path: str):
        """
        Load previously saved new parameters.
        
        Args:
            load_path: Path to the saved new parameters
        """
        logger.info(f"Loading new parameters from: {load_path}")
        
        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Verify compatibility
        assert checkpoint["num_new_tokens"] == self.num_new_tokens, \
            f"Token count mismatch: {checkpoint['num_new_tokens']} vs {self.num_new_tokens}"
        assert checkpoint["original_vocab_size"] == self.original_vocab_size, \
            f"Vocab size mismatch: {checkpoint['original_vocab_size']} vs {self.original_vocab_size}"
        
        # Load module states
        self.new_token_embeddings.load_state_dict(checkpoint["new_token_embeddings"])
        self.new_token_lm_head.load_state_dict(checkpoint["new_token_lm_head"])
        
        logger.info("Successfully loaded new parameters")
    



def test_naive_pretrain_encoder():
    """Test the NaiveEncoderForPretraining implementation."""
    print("=" * 80)
    print("Testing NaiveEncoderForPretraining")
    print("=" * 80)
    print()
    
    # Initialize model
    print("Initializing NaiveEncoderForPretraining...")
    encoder = NaiveEncoderForPretraining(
        llm_model_name="Qwen/Qwen2.5-7B",
        extended_tokenizer_path="data/generated/extended_tokenizer",
        combined_tokens_path="data/generated/combined_tokens.json",
        device="cpu",  # Use CPU for testing
    )
    print()
    
    # Test tokenization
    print("=" * 80)
    print("Test 1: Tokenization")
    print("=" * 80)
    
    test_prompt = "This is the image_classification tool with small input size. It uses 2 CPU cores, 4 GB CPU memory, 20 GPU SMs, and 2 GB GPU memory. The expected latency is 1252 ms. <TOOL_image_classification_small_2_4_20_2>"
    
    encoding = encoder.tokenizer(
        test_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    
    print(f"Input text: {test_prompt[:100]}...")
    print(f"Token IDs shape: {encoding['input_ids'].shape}")
    print(f"First 10 token IDs: {encoding['input_ids'][0][:10].tolist()}")
    print()
    
    # Test forward pass
    print("=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()
    
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Expected vocab size: {encoder.extended_vocab_size}")
    print()
    
    # Test parameter saving and loading
    print("=" * 80)
    print("Test 3: Parameter Saving and Loading")
    print("=" * 80)
    
    import os
    import tempfile
    
    # Create temp file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pt') as f:
        save_path = f.name
    
    try:
        # Save
        encoder.save_new_parameters(save_path)
        print(f"Saved parameters to: {save_path}")
        
        # Modify parameters
        original_embedding = encoder.llm_model.get_input_embeddings().weight.data[encoder.original_vocab_size].clone()
        encoder.llm_model.get_input_embeddings().weight.data[encoder.original_vocab_size] *= 2.0
        modified_embedding = encoder.llm_model.get_input_embeddings().weight.data[encoder.original_vocab_size].clone()
        
        print(f"Original embedding norm: {original_embedding.norm().item():.4f}")
        print(f"Modified embedding norm: {modified_embedding.norm().item():.4f}")
        
        # Load
        encoder.load_new_parameters(save_path)
        restored_embedding = encoder.llm_model.get_input_embeddings().weight.data[encoder.original_vocab_size].clone()
        
        print(f"Restored embedding norm: {restored_embedding.norm().item():.4f}")
        
        # Verify restoration
        if torch.allclose(original_embedding, restored_embedding, atol=1e-6):
            print("✓ Parameter save/load test passed!")
        else:
            print("✗ Parameter restoration failed!")
    finally:
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)
    
    print()
    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)


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
    # Test both encoders
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "naive":
        test_naive_pretrain_encoder()
    else:
        test_pretrain_encoder()

