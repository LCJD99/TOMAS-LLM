import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

class TomasModel(nn.Module):
    def __init__(
        self,
        llm_model_name: str = "Qwen2.5-7B",
        num_tools: int = 7,
        profiling_num: int = 243,
        llm_hidden_dim: int = 3584,
        device: str = 'cuda',
    ):
        """
        TOMAS Model integrating semantic encoding and tool resource encoding.

        Args:
            llm_model_name (str): Name of the LLM model to use for semantic encoding.
            profiling_file_path (str): Path to the profiling data file.
            device (str): Device to run the model on.
        """
        super().__init__()
        self.llm_model_name = llm_model_name
        self.hidden_dim = llm_hidden_dim

        # ===== Stream A: Tool Encoding =====
        # self.tool_embeddings = nn.Linear(num_tools, llm_hidden_dim).to(device)

        # ===== Stream B: Resource Encoding =====
        self.resource_encoder = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, llm_hidden_dim)
        ).to(device)

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16
        ).to(device)

        # ===== Fusion: MLP =====
        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(llm_hidden_dim * 2, llm_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(llm_hidden_dim, llm_hidden_dim)
        # ).to(device)

    
    def forward(self, resource_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TOMAS model.

        Args:
            tool_ids (torch.Tensor): Tensor of shape (batch_size, num_tools) representing tool usage.
            resource_features (torch.Tensor): Tensor of shape (batch_size, 6) representing resource features.

        Returns:
            torch.Tensor: Fused embedding of shape (batch_size, llm_hidden_dim).
        """
        # Stream A: Tool Encoding
        # tool_embed = self.tool_embeddings(tool_ids)  # shape = (batch_size, llm_hidden_dim)

        # Stream B: Resource Encoding
        resource_embed = self.resource_encoder(resource_features)  # shape = (batch_size, llm_hidden_dim)

        # Fusion
        # combined = torch.cat([tool_embed, resource_embed], dim=-1)  # shape = (batch_size, llm_hidden_dim * 2)
        # fused_embed = self.fusion_mlp(combined)  # shape = (batch_size, llm_hidden_dim)

        # return fused_embed


def load_model(tokenizer: AutoTokenizer,):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
    )

    model.resize_token_embeddings(len(tokenizer) + 2)  # for special tokens
    