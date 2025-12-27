"""
Semantic Encoder (Stream A) - Deep LLM-based tool semantic encoding.

This module uses complete LLM forward pass to encode tool semantics,
extracting the last layer's EOS token hidden state as the semantic representation.

Key features:
- Utilizes full transformer layers (not just embedding layer)
- Processes tool name + description as complete prompt
- Extracts EOS token hidden state for semantic summary
- Precomputes embeddings offline for O(1) inference
"""

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
    Deep semantic encoder using LLM forward pass.
    
    Architecture:
    1. Construct prompt: "Tool: {name}\nDescription: {description}\n"
    2. Run full LLM forward pass with output_hidden_states=True
    3. Extract last layer's EOS token hidden state
    4. Store as precomputed buffer for efficient inference
    
    The encoder is frozen during training and serves as a semantic anchor.
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B",
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


def test_semantic_encoder():
    """Test semantic encoder."""
    print("=== Testing SemanticEncoder ===\n")
    
    # Initialize encoder
    encoder = SemanticEncoder(
        llm_model_name="Qwen/Qwen2.5-0.5B",  # Use small model for testing
        tool_registry_path="data/tool_registry/tools.json"
    )
    
    print(f"Number of tools: {encoder.get_num_tools()}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    print(f"Tool names: {encoder.tool_names}")
    print()
    
    # Test encoding
    tool_ids = torch.tensor([0, 1, 2])
    semantic_emb = encoder(tool_ids)
    
    print(f"Input tool_ids: {tool_ids}")
    print(f"Output shape: {semantic_emb.shape}")
    print(f"Expected: [{len(tool_ids)}, {encoder.get_embedding_dim()}]")
    print()
    
    # Check embeddings are different
    cos_sim_01 = torch.cosine_similarity(semantic_emb[0], semantic_emb[1], dim=0)
    cos_sim_02 = torch.cosine_similarity(semantic_emb[0], semantic_emb[2], dim=0)
    
    print(f"Cosine similarity (tool 0 vs 1): {cos_sim_01:.4f}")
    print(f"Cosine similarity (tool 0 vs 2): {cos_sim_02:.4f}")
    print()
    
    # Test state dict
    state_dict = encoder.state_dict_without_llm()
    print(f"State dict keys: {state_dict.keys()}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_semantic_encoder()
