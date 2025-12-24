"""
Tool Encoder (Stream A) - Semantic encoding for tool descriptions.

This module implements two approaches for encoding tool semantic information:
1. Direct tool name mapping to learnable embedding table
2. Qwen tokenizer + embedding layer for full text encoding
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ToolNameEncoder(nn.Module):
    """
    Approach 1: Direct tool name mapping to learnable embeddings.
    
    Maps each unique tool name to a learnable embedding vector.
    This is a simple lookup table approach that assigns each tool
    a continuous embedding outside the standard vocabulary.
    
    Advantages:
    - Fast and memory efficient
    - Each tool gets a dedicated learnable representation
    - No dependency on external models
    
    Disadvantages:
    - Cannot generalize to unseen tools
    - Doesn't leverage semantic information in descriptions
    """
    
    def __init__(self, tool_names: List[str], d_tool: int):
        """
        Initialize tool name encoder.
        
        Args:
            tool_names: List of unique tool names
            d_tool: Dimension of tool embedding vectors
        """
        super().__init__()
        
        self.tool_names = sorted(tool_names)  # Ensure consistent ordering
        self.d_tool = d_tool
        
        # Create name to index mapping
        self.name_to_idx = {name: idx for idx, name in enumerate(self.tool_names)}
        self.num_tools = len(self.tool_names)
        
        # Learnable embedding table
        self.embedding = nn.Embedding(self.num_tools, d_tool)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        
        logger.info(f"Initialized ToolNameEncoder with {self.num_tools} tools, d_tool={d_tool}")
    
    def forward(self, tool_names: List[str]) -> torch.Tensor:
        """
        Encode tool names to embedding vectors.
        
        Args:
            tool_names: List of tool names to encode
        
        Returns:
            Tensor of shape (batch_size, d_tool)
        
        Raises:
            ValueError: If any tool name is not in the vocabulary
        """
        # Convert names to indices
        indices = []
        for name in tool_names:
            if name not in self.name_to_idx:
                raise ValueError(f"Unknown tool name: {name}")
            indices.append(self.name_to_idx[name])
        
        # Convert to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)
        
        # Lookup embeddings
        embeddings = self.embedding(indices_tensor)
        
        return embeddings
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get embeddings for all tools.
        
        Returns:
            Tensor of shape (num_tools, d_tool)
        """
        all_indices = torch.arange(self.num_tools, device=self.embedding.weight.device)
        return self.embedding(all_indices)
    
    def get_tool_embedding(self, tool_name: str) -> torch.Tensor:
        """
        Get embedding for a single tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Tensor of shape (d_tool,)
        """
        return self.forward([tool_name])[0]


class ToolTextEncoder(nn.Module):
    """
    Approach 2: Full text encoding using Qwen tokenizer + embedding layer.
    
    Encodes the full tool description (name + desc) using a pretrained
    language model tokenizer and embedding layer. Optionally uses pooling
    to aggregate token embeddings.
    
    Advantages:
    - Leverages semantic information from descriptions
    - Can potentially generalize to unseen tools with similar descriptions
    - Uses pretrained knowledge
    
    Disadvantages:
    - More memory and computation intensive
    - May require caching for efficiency
    """
    
    def __init__(
        self,
        model_name: str,
        d_tool: int,
        max_length: int = 256,
        pooling: str = "mean",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize tool text encoder.
        
        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B")
            d_tool: Output dimension for tool embeddings
            max_length: Maximum sequence length for tokenization
            pooling: Pooling strategy ("mean", "max", "cls")
            cache_dir: Directory to cache model files
        """
        super().__init__()
        
        self.model_name = model_name
        self.d_tool = d_tool
        self.max_length = max_length
        self.pooling = pooling
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Load embedding layer (just the embedding, not full model for efficiency)
        # For now, we'll use a simple approach: tokenizer vocab size -> embedding
        self.vocab_size = len(self.tokenizer)
        
        # Get embedding dimension from pretrained model
        # For Qwen2.5-7B, it's typically 3584 or 4096
        # We'll use a configurable hidden dimension
        try:
            config = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).config
            self.hidden_dim = config.hidden_size
        except:
            # Fallback if model is too large to load
            self.hidden_dim = 4096
            logger.warning(f"Could not load model config, using default hidden_dim={self.hidden_dim}")
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Projection layer to d_tool dimension
        self.projection = nn.Linear(self.hidden_dim, d_tool)
        
        logger.info(f"Initialized ToolTextEncoder: vocab_size={self.vocab_size}, "
                   f"hidden_dim={self.hidden_dim}, d_tool={d_tool}, pooling={pooling}")
    
    def _pool_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool token embeddings to sentence embedding.
        
        Args:
            embeddings: Token embeddings of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
        
        Returns:
            Pooled embeddings of shape (batch_size, hidden_dim)
        """
        if self.pooling == "mean":
            # Mean pooling (ignore padding tokens)
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return summed / counts
        
        elif self.pooling == "max":
            # Max pooling
            masked_embeddings = embeddings.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            return masked_embeddings.max(dim=1)[0]
        
        elif self.pooling == "cls":
            # Use first token (CLS-style)
            return embeddings[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode tool texts to embedding vectors.
        
        Args:
            texts: List of tool text descriptions (formatted as templates)
        
        Returns:
            Tensor of shape (batch_size, d_tool)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to same device as embeddings
        input_ids = encoded['input_ids'].to(self.embedding.weight.device)
        attention_mask = encoded['attention_mask'].to(self.embedding.weight.device)
        
        # Get embeddings
        token_embeddings = self.embedding(input_ids)
        
        # Pool to sentence embedding
        pooled = self._pool_embeddings(token_embeddings, attention_mask)
        
        # Project to d_tool dimension
        output = self.projection(pooled)
        
        return output


class ToolEncoder(nn.Module):
    """
    Unified tool encoder wrapper that supports both encoding approaches.
    
    Provides a common interface and caching mechanism for tool embeddings.
    """
    
    def __init__(
        self,
        config: Dict,
        tool_names: Optional[List[str]] = None,
        encoder_type: str = "name"
    ):
        """
        Initialize tool encoder.
        
        Args:
            config: Configuration dictionary with encoder settings
            tool_names: List of tool names (required for name encoder)
            encoder_type: Type of encoder ("name" or "text")
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.d_tool = config['model']['tool_encoder']['d_tool']
        
        # Create appropriate encoder
        if encoder_type == "name":
            if tool_names is None:
                raise ValueError("tool_names required for name encoder")
            
            self.encoder = ToolNameEncoder(
                tool_names=tool_names,
                d_tool=self.d_tool
            )
        
        elif encoder_type == "text":
            model_name = config['model']['backbone']['name']
            max_length = config['model']['tool_encoder']['max_desc_length']
            
            self.encoder = ToolTextEncoder(
                model_name=model_name,
                d_tool=self.d_tool,
                max_length=max_length,
                pooling="mean"
            )
        
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Cache for static tool embeddings
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_enabled = True
        
        logger.info(f"Initialized ToolEncoder with type={encoder_type}, d_tool={self.d_tool}")
    
    def enable_cache(self):
        """Enable embedding caching."""
        self.cache_enabled = True
        logger.info("Embedding cache enabled")
    
    def disable_cache(self):
        """Disable embedding caching."""
        self.cache_enabled = False
        logger.info("Embedding cache disabled")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def forward(
        self,
        tool_names: Optional[List[str]] = None,
        tool_texts: Optional[List[str]] = None,
        use_cache: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Encode tools to embedding vectors.
        
        Args:
            tool_names: List of tool names (for name encoder or cache keys)
            tool_texts: List of tool text descriptions (for text encoder)
            use_cache: Override cache setting for this call
        
        Returns:
            Tensor of shape (batch_size, d_tool)
        """
        use_cache = self.cache_enabled if use_cache is None else use_cache
        
        if self.encoder_type == "name":
            if tool_names is None:
                raise ValueError("tool_names required for name encoder")
            
            # Check cache
            if use_cache:
                cached_embeddings = []
                uncached_names = []
                uncached_indices = []
                
                for idx, name in enumerate(tool_names):
                    if name in self.cache:
                        cached_embeddings.append(self.cache[name])
                    else:
                        uncached_names.append(name)
                        uncached_indices.append(idx)
                
                # Encode uncached names
                if uncached_names:
                    new_embeddings = self.encoder(uncached_names)
                    
                    # Update cache
                    for name, emb in zip(uncached_names, new_embeddings):
                        self.cache[name] = emb.detach()
                    
                    # Combine cached and new embeddings
                    all_embeddings = [None] * len(tool_names)
                    
                    # Fill cached
                    cached_idx = 0
                    for idx in range(len(tool_names)):
                        if idx not in uncached_indices:
                            all_embeddings[idx] = cached_embeddings[cached_idx]
                            cached_idx += 1
                    
                    # Fill uncached
                    for list_idx, orig_idx in enumerate(uncached_indices):
                        all_embeddings[orig_idx] = new_embeddings[list_idx]
                    
                    return torch.stack(all_embeddings)
                else:
                    # All cached
                    return torch.stack(cached_embeddings)
            else:
                # No caching
                return self.encoder(tool_names)
        
        elif self.encoder_type == "text":
            if tool_texts is None:
                raise ValueError("tool_texts required for text encoder")
            
            # For text encoder, use tool_names as cache keys if provided
            if use_cache and tool_names is not None:
                # Similar caching logic
                cached_embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                for idx, (name, text) in enumerate(zip(tool_names, tool_texts)):
                    if name in self.cache:
                        cached_embeddings.append(self.cache[name])
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(idx)
                
                if uncached_texts:
                    new_embeddings = self.encoder(uncached_texts)
                    
                    # Update cache
                    for idx, name in enumerate([tool_names[i] for i in uncached_indices]):
                        self.cache[name] = new_embeddings[idx].detach()
                    
                    # Combine
                    all_embeddings = [None] * len(tool_names)
                    cached_idx = 0
                    for idx in range(len(tool_names)):
                        if idx not in uncached_indices:
                            all_embeddings[idx] = cached_embeddings[cached_idx]
                            cached_idx += 1
                    
                    for list_idx, orig_idx in enumerate(uncached_indices):
                        all_embeddings[orig_idx] = new_embeddings[list_idx]
                    
                    return torch.stack(all_embeddings)
                else:
                    return torch.stack(cached_embeddings)
            else:
                # No caching
                return self.encoder(tool_texts)
        
        else:
            raise RuntimeError(f"Invalid encoder_type: {self.encoder_type}")
    
    def precompute_embeddings(
        self,
        tool_names: List[str],
        tool_texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute and cache embeddings for all tools.
        
        Args:
            tool_names: List of all tool names
            tool_texts: List of tool text descriptions (for text encoder)
        
        Returns:
            Dictionary mapping tool names to embeddings
        """
        logger.info(f"Precomputing embeddings for {len(tool_names)} tools...")
        
        if self.encoder_type == "name":
            embeddings = self.encoder(tool_names)
        else:
            if tool_texts is None:
                raise ValueError("tool_texts required for text encoder")
            embeddings = self.encoder(tool_texts)
        
        # Update cache
        for name, emb in zip(tool_names, embeddings):
            self.cache[name] = emb.detach()
        
        logger.info(f"Cached {len(self.cache)} tool embeddings")
        
        return self.cache
