"""
User Task Encoding for LLM Input.

This module handles encoding user task descriptions into embeddings suitable
for the LLM backbone. It uses the same tokenizer/embedding as the LLM to ensure
compatibility.

Input: User task text (e.g., "Generate an image of a sunset over mountains")
Output: Task embeddings compatible with LLM input (seq_len, d_model)
"""

import logging
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class TaskEmbedding(nn.Module):
    """
    Converts task text to embeddings using pretrained LLM tokenizer and embeddings.
    
    This ensures the task representation is compatible with the LLM backbone
    (Qwen2.5-7B) and can be processed alongside tool embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        max_length: int = 512,
        device: str = "cpu",
        use_pretrained_embeddings: bool = True
    ):
        """
        Initialize task embedding module.
        
        Args:
            model_name: HuggingFace model name (should match LLM backbone)
            max_length: Maximum sequence length for task description
            device: Device for embeddings ("cpu" or "cuda")
            use_pretrained_embeddings: Whether to use pretrained embeddings
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Load embeddings
        if use_pretrained_embeddings:
            logger.info(f"Loading pretrained embeddings from {model_name}")
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for embeddings
            )
            
            # Extract embedding layer
            if hasattr(model, 'embed_tokens'):
                self.embeddings = model.embed_tokens
            elif hasattr(model, 'embeddings'):
                self.embeddings = model.embeddings
            elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                self.embeddings = model.model.embed_tokens
            else:
                raise ValueError(f"Cannot find embedding layer in {model_name}")
            
            # Freeze pretrained embeddings
            for param in self.embeddings.parameters():
                param.requires_grad = False
            
            self.d_model = self.embeddings.embedding_dim
            logger.info(f"Loaded pretrained embeddings: vocab_size={self.embeddings.num_embeddings}, "
                       f"d_model={self.d_model}")
        else:
            # Create trainable embeddings from scratch
            vocab_size = len(self.tokenizer)
            self.d_model = 1024  # Default dimension
            self.embeddings = nn.Embedding(vocab_size, self.d_model)
            logger.info(f"Created trainable embeddings: vocab_size={vocab_size}, d_model={self.d_model}")
        
        self.embeddings = self.embeddings.to(device)
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: Single text or list of texts
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: Return format ("pt" for PyTorch)
        
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        encoding = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        
        # Move to device
        for key in encoding:
            if isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(self.device)
        
        return encoding
    
    def forward(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert task texts to embeddings.
        
        Args:
            texts: Task description(s) as text
            input_ids: Pre-tokenized input IDs (if texts not provided)
            attention_mask: Attention mask (if texts not provided)
        
        Returns:
            Tuple of (embeddings, attention_mask)
                - embeddings: (batch_size, seq_len, d_model)
                - attention_mask: (batch_size, seq_len)
        """
        # Tokenize if texts provided
        if texts is not None:
            encoding = self.tokenize(texts)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
        elif input_ids is None:
            raise ValueError("Either texts or input_ids must be provided")
        
        # Ensure attention_mask exists
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        return embeddings, attention_mask
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        return len(self.tokenizer)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.d_model
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TaskEmbedding':
        """
        Create TaskEmbedding from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized TaskEmbedding instance
        """
        backbone_config = config['model']['backbone']
        task_config = config['model'].get('task_encoder', {})
        
        return cls(
            model_name=backbone_config['name'],
            max_length=task_config.get('max_length', 512),
            device=backbone_config.get('device', 'cpu'),
            use_pretrained_embeddings=task_config.get('use_pretrained_embeddings', True)
        )


class UserTaskEncoder(nn.Module):
    """
    Complete user task encoder with optional projection layer.
    
    This module:
    1. Tokenizes user task descriptions
    2. Converts to embeddings (pretrained or trainable)
    3. Optionally projects to match tool embedding dimension
    4. Returns both full sequence and pooled representations
    """
    
    def __init__(
        self,
        task_embedding: TaskEmbedding,
        project_to_tool_dim: bool = False,
        d_tool: Optional[int] = None,
        pooling_method: str = "mean"
    ):
        """
        Initialize user task encoder.
        
        Args:
            task_embedding: TaskEmbedding instance
            project_to_tool_dim: Whether to project to tool embedding dimension
            d_tool: Tool embedding dimension (required if project_to_tool_dim=True)
            pooling_method: How to pool sequence ("mean", "max", "cls", "last")
        """
        super().__init__()
        
        self.task_embedding = task_embedding
        self.project_to_tool_dim = project_to_tool_dim
        self.pooling_method = pooling_method
        
        self.d_model = task_embedding.d_model
        
        # Optional projection layer
        if project_to_tool_dim:
            if d_tool is None:
                raise ValueError("d_tool must be provided when project_to_tool_dim=True")
            
            self.projection = nn.Linear(self.d_model, d_tool)
            self.output_dim = d_tool
            logger.info(f"Created projection layer: {self.d_model} → {d_tool}")
        else:
            self.projection = None
            self.output_dim = self.d_model
        
        logger.info(f"Initialized UserTaskEncoder: d_model={self.d_model}, "
                   f"output_dim={self.output_dim}, pooling={pooling_method}")
    
    def pool_sequence(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence embeddings to single vector.
        
        Args:
            embeddings: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            Pooled embeddings (batch_size, d_model)
        """
        if self.pooling_method == "mean":
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        elif self.pooling_method == "max":
            # Max pooling with mask
            embeddings = embeddings.clone()
            embeddings[attention_mask == 0] = -1e9
            pooled = torch.max(embeddings, dim=1)[0]
        
        elif self.pooling_method == "cls":
            # Use first token (CLS-style)
            pooled = embeddings[:, 0, :]
        
        elif self.pooling_method == "last":
            # Use last non-padded token
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_size = embeddings.size(0)
            pooled = embeddings[torch.arange(batch_size), seq_lengths, :]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        return pooled
    
    def forward(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = True
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encode user task descriptions.
        
        Args:
            texts: Task description(s)
            input_ids: Pre-tokenized IDs
            attention_mask: Attention mask
            return_pooled: Whether to return pooled representation
        
        Returns:
            If return_pooled=True:
                (sequence_embeddings, pooled_embeddings, attention_mask)
            If return_pooled=False:
                (sequence_embeddings, attention_mask)
            
            Where:
                - sequence_embeddings: (batch_size, seq_len, output_dim)
                - pooled_embeddings: (batch_size, output_dim)
                - attention_mask: (batch_size, seq_len)
        """
        # Get base embeddings
        embeddings, attention_mask = self.task_embedding(texts, input_ids, attention_mask)
        
        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Return results
        if return_pooled:
            pooled = self.pool_sequence(embeddings, attention_mask)
            return embeddings, pooled, attention_mask
        else:
            return embeddings, attention_mask
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        project_to_tool_dim: bool = False
    ) -> 'UserTaskEncoder':
        """
        Create UserTaskEncoder from configuration.
        
        Args:
            config: Configuration dictionary
            project_to_tool_dim: Whether to project to tool dimension
        
        Returns:
            Initialized UserTaskEncoder instance
        """
        task_embedding = TaskEmbedding.from_config(config)
        
        task_config = config['model'].get('task_encoder', {})
        pooling_method = task_config.get('pooling_method', 'mean')
        
        d_tool = None
        if project_to_tool_dim:
            # Get tool dimension from config
            d_tool_semantic = config['model']['tool_encoder']['d_tool']
            d_tool_resource = config['model']['resource_mlp']['d_resource']
            d_tool = d_tool_semantic + d_tool_resource  # d_toolaware
        
        return cls(
            task_embedding=task_embedding,
            project_to_tool_dim=project_to_tool_dim,
            d_tool=d_tool,
            pooling_method=pooling_method
        )
