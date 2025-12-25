"""
Configuration Embedding Generator for TOMAS-LLM Phase 0.

Implements "Weights as Memory" architecture by converting static tool configurations
into embedding vectors that will be used to initialize the Config Pointer Head.

Architecture:
    Stream A (Semantic): Tool Name + Description → Frozen LLM → Mean Pooling → [3584]
    Stream B (Resource): 6D Resource Data → MLP → [3584]  
    Fusion: Concat[A,B] → Self-Attention → Take First Token → [3584]

This replaces runtime tool encoding with offline weight precomputation.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import existing ResourceMLP for Stream B
from ..encoders.resource_mlp import ResourceMLP

logger = logging.getLogger(__name__)


class ToolSemanticEncoder(nn.Module):
    """
    Stream A: Encodes tool semantics using frozen LLM.
    
    Converts (tool_name, description) pairs into dense semantic vectors
    by leveraging pretrained language understanding.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 128
    ):
        """
        Initialize semantic encoder with frozen LLM.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computation device
            dtype: Model precision
            max_length: Maximum sequence length for tool descriptions
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Determine torch dtype
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        logger.info(f"Loading frozen LLM for semantic encoding: {model_name}")
        logger.info(f"Device: {device}, dtype: {dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Load and freeze model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map=device if device != "cpu" else None
        )
        
        # Freeze all parameters - no gradients needed
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()  # Permanent eval mode
        
        # Get hidden dimension
        self.hidden_dim = self.model.config.hidden_size
        logger.info(f"LLM loaded: hidden_dim={self.hidden_dim}")
        
    def construct_prompt(self, tool_name: str, description: str) -> str:
        """
        Construct standardized prompt for tool encoding.
        
        Format: "Tool Name: {name}. Description: {desc}."
        """
        return f"Tool Name: {tool_name}. Description: {description}."
    
    def forward(self, tool_infos: List[Dict[str, str]]) -> torch.Tensor:
        """
        Encode tool semantics using frozen LLM.
        
        Args:
            tool_infos: List of {"name": str, "desc": str} dictionaries
            
        Returns:
            Semantic embeddings: [batch_size, hidden_dim]
        """
        batch_size = len(tool_infos)
        
        # Construct prompts
        prompts = [
            self.construct_prompt(info["name"], info["desc"]) 
            for info in tool_infos
        ]
        
        logger.debug(f"Encoding {batch_size} tool semantics")
        logger.debug(f"Sample prompt: {prompts[0]}")
        
        # Tokenize batch
        encoded = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Forward through frozen LLM
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Extract last layer hidden states
        last_hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        # Mean pooling with attention mask
        # Expand mask to match hidden states
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_states)
        
        # Zero out padded positions
        masked_hidden = last_hidden_states * attention_mask_expanded
        
        # Sum and divide by sequence length (excluding padding)
        sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)  # [batch, 1]
        
        # Mean pooling
        semantic_embeddings = sum_hidden / seq_lengths.clamp(min=1)  # [batch, hidden_dim]
        
        logger.debug(f"Semantic embeddings shape: {semantic_embeddings.shape}")
        return semantic_embeddings


class SemanticResourceFusion(nn.Module):
    """
    Fusion Module: Combines semantic and resource embeddings.
    
    Uses self-attention to allow interaction between tool semantics
    and resource characteristics, then extracts a unified representation.
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize fusion module.
        
        Args:
            hidden_dim: Embedding dimension (should match LLM)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        logger.info(f"Fusion module: {num_heads}-head attention, dim={hidden_dim}")
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: [batch, seq, dim]
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding for sequence order
        self.pos_encoding = nn.Parameter(
            torch.randn(2, hidden_dim) * 0.02  # 2 positions: semantic, resource
        )
    
    def forward(
        self,
        semantic_embeds: torch.Tensor,
        resource_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse semantic and resource embeddings.
        
        Args:
            semantic_embeds: [batch_size, hidden_dim] - Stream A output
            resource_embeds: [batch_size, hidden_dim] - Stream B output
            
        Returns:
            Fused embeddings: [batch_size, hidden_dim] - Final config embeddings
        """
        batch_size = semantic_embeds.size(0)
        
        logger.debug(f"Fusing {batch_size} semantic-resource pairs")
        
        # Stack embeddings to create sequence [semantic, resource]
        sequence = torch.stack([semantic_embeds, resource_embeds], dim=1)  # [batch, 2, hidden_dim]
        
        # Add positional encoding
        sequence = sequence + self.pos_encoding.unsqueeze(0)  # Broadcast to batch
        
        # Apply self-attention
        attended, attention_weights = self.self_attention(
            query=sequence,
            key=sequence,
            value=sequence
        )
        
        # Residual connection and layer norm
        attended = self.layer_norm(attended + sequence)
        
        # Extract first token (semantic-focused) as final embedding
        fused_embeddings = attended[:, 0, :]  # [batch_size, hidden_dim]
        
        logger.debug(f"Fused embeddings shape: {fused_embeddings.shape}")
        return fused_embeddings


class ConfigEmbeddingGenerator(nn.Module):
    """
    Complete Phase 0 configuration embedding generator.
    
    Implements the "Weights as Memory" approach by converting 1701 static
    tool configurations into embedding vectors that initialize the Config
    Pointer Head weights.
    
    Pipeline:
        Tool Info → Stream A (Semantic) → [batch, 3584]
        Resource Data → Stream B (Resource) → [batch, 3584]
        [A, B] → Fusion (Self-Attention) → [batch, 3584]
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        resource_input_dim: int = 6,
        resource_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 128
    ):
        """
        Initialize complete embedding generator.
        
        Args:
            llm_model_name: LLM for semantic encoding
            device: Computation device
            dtype: Model precision
            resource_input_dim: Resource feature dimension
            resource_hidden_dim: Resource MLP hidden size
            fusion_num_heads: Number of fusion attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for tool descriptions
        """
        super().__init__()
        
        self.device = device
        
        logger.info("=" * 60)
        logger.info("Initializing ConfigEmbeddingGenerator")
        logger.info("=" * 60)
        
        # Initialize Stream A: Semantic Encoder
        self.semantic_encoder = ToolSemanticEncoder(
            model_name=llm_model_name,
            device=device,
            dtype=dtype,
            max_length=max_seq_length
        )
        
        # Get LLM hidden dimension for other components
        llm_hidden_dim = self.semantic_encoder.hidden_dim
        
        # Initialize Stream B: Resource Encoder (using existing ResourceMLP)
        self.resource_encoder = ResourceMLP(
            input_dim=resource_input_dim,
            hidden_dim=resource_hidden_dim,
            d_resource=llm_hidden_dim,  # Set to 3584 to match LLM hidden dim
            dropout=dropout
        )
        
        # Initialize Fusion Module
        self.fusion = SemanticResourceFusion(
            hidden_dim=llm_hidden_dim,
            num_heads=fusion_num_heads,
            dropout=dropout
        )
        
        # Move trainable components to device
        self.resource_encoder = self.resource_encoder.to(device)
        self.fusion = self.fusion.to(device)
        
        logger.info(f"ConfigEmbeddingGenerator initialized successfully")
        logger.info(f"Output embedding dimension: {llm_hidden_dim}")
    
    def forward(
        self,
        tool_infos: List[Dict[str, str]],
        resource_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate configuration embeddings for tool-resource pairs.
        
        Args:
            tool_infos: List of {"name": str, "desc": str} tool information
            resource_data: [batch_size, 6] resource profiles
            
        Returns:
            Config embeddings: [batch_size, hidden_dim] ready for weight initialization
        """
        batch_size = len(tool_infos)
        assert resource_data.size(0) == batch_size, \
            f"Tool info count ({batch_size}) must match resource data batch size ({resource_data.size(0)})"
        
        logger.info(f"Generating embeddings for {batch_size} configurations")
        
        # Move resource data to correct device
        resource_data = resource_data.to(self.device)
        
        # Stream A: Encode tool semantics
        logger.debug("Stream A: Encoding tool semantics...")
        semantic_embeddings = self.semantic_encoder(tool_infos)
        
        # Stream B: Encode resource profiles
        logger.debug("Stream B: Encoding resource profiles...")
        resource_embeddings = self.resource_encoder(resource_data)
        
        # Fusion: Combine semantic and resource information
        logger.debug("Fusion: Combining semantic and resource embeddings...")
        config_embeddings = self.fusion(semantic_embeddings, resource_embeddings)
        
        logger.info(f"Generated config embeddings shape: {config_embeddings.shape}")
        return config_embeddings
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ConfigEmbeddingGenerator':
        """
        Create generator from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Initialized ConfigEmbeddingGenerator
        """
        offline_config = config.get('offline', {})
        embedding_config = offline_config.get('embedding_generator', {})
        llm_config = config.get('llm', {}).get('model', {})
        
        return cls(
            llm_model_name=llm_config.get('name', 'Qwen/Qwen2.5-7B'),
            device=llm_config.get('device', 'cuda'),
            dtype=llm_config.get('dtype', 'bfloat16'),
            resource_input_dim=embedding_config.get('resource_input_dim', 6),
            resource_hidden_dim=embedding_config.get('resource_hidden_dim', 512),
            fusion_num_heads=embedding_config.get('fusion_num_heads', 8),
            dropout=embedding_config.get('dropout', 0.1),
            max_seq_length=embedding_config.get('max_seq_length', 128)
        )
    
    def generate_batch(
        self,
        tool_infos: List[Dict[str, str]],
        resource_data: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate embeddings in batches to handle large datasets.
        
        Args:
            tool_infos: List of tool information dictionaries
            resource_data: [N, 6] resource data tensor
            batch_size: Processing batch size
            
        Returns:
            All config embeddings: [N, hidden_dim]
        """
        total_configs = len(tool_infos)
        logger.info(f"Processing {total_configs} configs in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, total_configs, batch_size):
            end_idx = min(i + batch_size, total_configs)
            
            batch_tools = tool_infos[i:end_idx]
            batch_resources = resource_data[i:end_idx]
            
            logger.debug(f"Processing batch {i//batch_size + 1}: configs {i}-{end_idx-1}")
            
            with torch.no_grad():  # Save memory during inference
                batch_embeddings = self.forward(batch_tools, batch_resources)
                all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory
        
        # Concatenate all batches
        final_embeddings = torch.cat(all_embeddings, dim=0)
        logger.info(f"Generated all embeddings: {final_embeddings.shape}")
        
        return final_embeddings