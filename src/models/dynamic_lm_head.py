"""
Dynamic LM Head
动态计算虚拟 Token 的输出 logits，融合语义信息和资源配置信息
"""

import torch
import torch.nn as nn
from typing import Optional


class DynamicLMHead(nn.Module):
    """
    动态 LM Head 层
    
    核心公式:
        对于虚拟 token: logit = W_lm_head · (E_tool + ProfileEncoder(Profile_Vector))
        对于普通 token: logit = W_lm_head · E_token (原始方式)
    
    在输出时动态计算虚拟 token 的 logits
    """
    
    def __init__(
        self,
        original_lm_head: nn.Linear,
        profile_encoder: nn.Module,
        tool_semantics: torch.Tensor,
        profiles: torch.Tensor,
        new_token_start_idx: int
    ):
        """
        初始化动态 LM Head
        
        Args:
            original_lm_head: 原始的 lm_head Linear 层
            profile_encoder: Profile Encoder (Hypernetwork)
            tool_semantics: 预计算的工具语义向量 [num_new_tokens, hidden_dim]
            profiles: 对应的资源配置向量 [num_new_tokens, profile_dim]
            new_token_start_idx: 新 Token 在词表中的起始 ID
        """
        super().__init__()
        
        self.original_lm_head = original_lm_head
        self.profile_encoder = profile_encoder
        self.new_token_start_idx = new_token_start_idx
        
        # 注册为 Buffer
        self.register_buffer('tool_semantics', tool_semantics)
        self.register_buffer('profiles', profiles)
        
        # 配置信息
        self.num_new_tokens = tool_semantics.shape[0]
        self.hidden_dim = tool_semantics.shape[1]
        self.vocab_size = original_lm_head.out_features
        
        # 为虚拟 token 预计算动态 embeddings 并缓存其对应的权重
        # 注意：这里我们不直接修改 original_lm_head.weight，而是在前向传播时动态计算
    
    def compute_virtual_token_weights(self) -> torch.Tensor:
        """
        计算所有虚拟 token 对应的 lm_head 权重
        
        Returns:
            virtual_weights: [num_new_tokens, hidden_dim]
        """
        # 计算动态 embeddings
        # E_tool + ProfileEncoder(Profile)
        profile_deltas = self.profile_encoder(
            self.profiles.to(dtype=self.tool_semantics.dtype)
        )  # [num_new_tokens, hidden_dim]
        
        virtual_embeddings = self.tool_semantics + profile_deltas  # [num_new_tokens, hidden_dim]
        
        return virtual_embeddings
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播：动态计算 logits
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 1. 使用原始 lm_head 计算所有 token 的 logits
        # 这包括了虚拟 token，但这些 logits 是基于原始权重的，需要被覆盖
        logits = self.original_lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # 2. 动态计算虚拟 token 的 logits
        # 计算虚拟 token 的动态 embeddings
        virtual_embeddings = self.compute_virtual_token_weights()  # [num_new_tokens, hidden_dim]
        
        # 计算虚拟 token 的 logits
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # virtual_embeddings: [num_new_tokens, hidden_dim]
        # 需要计算: hidden_states @ virtual_embeddings.T
        
        # Reshape for batch matrix multiplication
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # 计算虚拟 token 的 logits
        virtual_logits = torch.matmul(
            hidden_flat, 
            virtual_embeddings.t()
        )  # [batch_size * seq_len, num_new_tokens]
        
        # Reshape back
        virtual_logits = virtual_logits.view(batch_size, seq_len, self.num_new_tokens)
        
        # 3. 覆盖原 logits 中虚拟 token 的部分
        # 虚拟 token 在词表中的范围: [new_token_start_idx, new_token_start_idx + num_new_tokens)
        logits[:, :, self.new_token_start_idx:self.new_token_start_idx + self.num_new_tokens] = virtual_logits
        
        # 添加 bias（如果存在）
        if self.original_lm_head.bias is not None:
            # bias 已经在 original_lm_head 的计算中添加了，但虚拟 token 部分需要重新添加
            bias_virtual = self.original_lm_head.bias[
                self.new_token_start_idx:self.new_token_start_idx + self.num_new_tokens
            ]
            logits[:, :, self.new_token_start_idx:self.new_token_start_idx + self.num_new_tokens] += bias_virtual
        
        return logits
    
    def update_tool_semantics(self, new_semantics: torch.Tensor):
        """
        更新工具的语义向量
        
        Args:
            new_semantics: 新的语义向量 [num_new_tokens, hidden_dim]
        """
        if new_semantics.shape != self.tool_semantics.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.tool_semantics.shape}, "
                f"got {new_semantics.shape}"
            )
        
        self.tool_semantics.copy_(new_semantics)
    
    def update_profiles(self, new_profiles: torch.Tensor):
        """
        更新资源配置向量
        
        Args:
            new_profiles: 新的配置向量 [num_new_tokens, profile_dim]
        """
        if new_profiles.shape != self.profiles.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.profiles.shape}, "
                f"got {new_profiles.shape}"
            )
        
        self.profiles.copy_(new_profiles)


if __name__ == '__main__':
    # 测试代码
    print("Testing DynamicLMHead...")
    
    from src.models.profile_encoder import ProfileHyperNet
    
    # 参数
    vocab_size = 50000
    num_new_tokens = 100
    new_token_start_idx = vocab_size - num_new_tokens
    hidden_dim = 512
    profile_dim = 5
    batch_size = 2
    seq_len = 10
    
    # 创建原始 lm_head
    original_lm_head = nn.Linear(hidden_dim, vocab_size)
    
    # 创建 profile encoder
    profile_encoder = ProfileHyperNet(
        input_dim=profile_dim,
        hidden_dims=[64, 128],
        output_dim=hidden_dim,
        zero_init=True
    )
    
    # 创建工具语义和配置
    tool_semantics = torch.randn(num_new_tokens, hidden_dim)
    profiles = torch.randint(1, 4, (num_new_tokens, profile_dim)).float()
    
    # 创建动态 lm_head
    dynamic_lm_head = DynamicLMHead(
        original_lm_head=original_lm_head,
        profile_encoder=profile_encoder,
        tool_semantics=tool_semantics,
        profiles=profiles,
        new_token_start_idx=new_token_start_idx
    )
    
    # 测试输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 前向传播
    logits = dynamic_lm_head(hidden_states)
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    
    # 验证虚拟 token 的 logits 被正确更新
    print(f"\nVirtual token logits range:")
    virtual_logits = logits[:, :, new_token_start_idx:new_token_start_idx + num_new_tokens]
    print(f"  Min: {virtual_logits.min().item():.4f}")
    print(f"  Max: {virtual_logits.max().item():.4f}")
    print(f"  Mean: {virtual_logits.mean().item():.4f}")
