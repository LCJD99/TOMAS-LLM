"""
Dynamic Tool Embedding Layer
动态计算工具 Token 的 Embedding，融合语义信息和资源配置信息
"""

import torch
import torch.nn as nn
from typing import Optional


class DynamicToolEmbedding(nn.Module):
    """
    动态工具 Embedding 层
    
    核心公式:
        Token_Embedding = E_tool (语义) + ProfileEncoder(Profile_Vector) (资源偏移)
    
    对于虚拟工具 Token，动态计算其 Embedding；
    对于普通 Token，使用原始 Embedding。
    """
    
    def __init__(
        self,
        original_embedding: nn.Embedding,
        profile_encoder: nn.Module,
        tool_semantics: torch.Tensor,
        profiles: torch.Tensor,
        new_token_start_idx: int
    ):
        """
        初始化动态 Embedding 层
        
        Args:
            original_embedding: 原始的 nn.Embedding 层（处理普通 token）
            profile_encoder: Profile Encoder (Hypernetwork)，将配置映射为偏移量
            tool_semantics: 预计算的工具语义向量 [num_new_tokens, hidden_dim]
            profiles: 对应的资源配置向量 [num_new_tokens, profile_dim]
            new_token_start_idx: 新 Token 在词表中的起始 ID
        """
        super().__init__()
        
        self.original_embedding = original_embedding
        self.profile_encoder = profile_encoder
        self.new_token_start_idx = new_token_start_idx
        
        # 注册为 Buffer - 不会被优化器更新，但会随模型移动设备
        self.register_buffer('tool_semantics', tool_semantics)
        self.register_buffer('profiles', profiles)
        
        # 获取配置信息
        self.num_new_tokens = tool_semantics.shape[0]
        self.hidden_dim = tool_semantics.shape[1]
        self.profile_dim = profiles.shape[1]
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播：动态计算 Embedding
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
        
        Returns:
            inputs_embeds: Embedding 表示 [batch_size, seq_len, hidden_dim]
        """
        # 1. 获取基础 Embeddings（对于新 token 这些值是无意义的占位符）
        inputs_embeds = self.original_embedding(input_ids)
        
        # 2. 识别哪些位置是新工具 Token
        is_new_token_mask = input_ids >= self.new_token_start_idx
        
        # 3. 如果当前 batch 中没有新 token，直接返回
        if not is_new_token_mask.any():
            return inputs_embeds
        
        # 4. 计算新 Token 的动态 Embedding
        # 获取新 token 相对于起始索引的偏移（用于索引 buffer）
        relative_indices = input_ids[is_new_token_mask] - self.new_token_start_idx
        
        # 边界检查
        if (relative_indices >= self.num_new_tokens).any() or (relative_indices < 0).any():
            raise ValueError(
                f"Token ID out of range. "
                f"Got indices {relative_indices.tolist()}, "
                f"but num_new_tokens={self.num_new_tokens}"
            )
        
        # 查表获取基础语义和配置
        current_semantics = self.tool_semantics[relative_indices]  # [N, hidden_dim]
        current_profiles = self.profiles[relative_indices]  # [N, profile_dim]
        
        # 5. 使用 Profile Encoder 计算资源配置的偏移量
        # 确保数据类型匹配
        profile_deltas = self.profile_encoder(
            current_profiles.to(dtype=inputs_embeds.dtype)
        )  # [N, hidden_dim]
        
        # 6. 融合：语义 + 资源偏移
        new_token_embeds = current_semantics.to(dtype=inputs_embeds.dtype) + profile_deltas
        
        # 7. 覆盖原 Embedding 中的对应位置
        # 创建副本以避免 in-place 操作影响梯度
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[is_new_token_mask] = new_token_embeds
        
        return inputs_embeds
    
    def get_embedding_for_token(
        self,
        token_id: int,
        include_delta: bool = True
    ) -> torch.Tensor:
        """
        获取单个 token 的 embedding
        
        Args:
            token_id: Token ID
            include_delta: 是否包含资源配置偏移（False 则只返回基础语义）
        
        Returns:
            embedding: [hidden_dim]
        """
        if token_id < self.new_token_start_idx:
            # 普通 token，返回原始 embedding
            return self.original_embedding.weight[token_id]
        
        # 新工具 token
        relative_idx = token_id - self.new_token_start_idx
        
        if relative_idx >= self.num_new_tokens:
            raise ValueError(f"Token ID {token_id} out of range")
        
        base_semantic = self.tool_semantics[relative_idx]
        
        if not include_delta:
            return base_semantic
        
        # 计算资源偏移
        profile = self.profiles[relative_idx:relative_idx+1]  # [1, profile_dim]
        delta = self.profile_encoder(profile.to(dtype=base_semantic.dtype))  # [1, hidden_dim]
        
        return base_semantic + delta.squeeze(0)
    
    def update_tool_semantics(self, new_semantics: torch.Tensor):
        """
        更新工具的语义向量（例如在训练后重新计算）
        
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


class DynamicToolEmbeddingWithCache(DynamicToolEmbedding):
    """
    带缓存的动态 Embedding 层
    
    对于相同的 token ID，缓存其计算结果以提高推理效率
    """
    
    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """带缓存的前向传播"""
        # 训练模式下不使用缓存
        if self.training:
            return super().forward(input_ids)
        
        # 推理模式下使用缓存
        return self._forward_with_cache(input_ids)
    
    def _forward_with_cache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """带缓存的前向传播（仅推理）"""
        inputs_embeds = self.original_embedding(input_ids)
        is_new_token_mask = input_ids >= self.new_token_start_idx
        
        if not is_new_token_mask.any():
            return inputs_embeds
        
        # 获取需要计算的 token IDs
        new_token_ids = input_ids[is_new_token_mask]
        unique_ids = new_token_ids.unique()
        
        # 检查缓存
        embeddings_map = {}
        for token_id in unique_ids.tolist():
            if token_id in self.cache:
                embeddings_map[token_id] = self.cache[token_id]
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                # 计算 embedding
                relative_idx = token_id - self.new_token_start_idx
                semantic = self.tool_semantics[relative_idx]
                profile = self.profiles[relative_idx:relative_idx+1]
                delta = self.profile_encoder(profile.to(dtype=semantic.dtype))
                embedding = semantic + delta.squeeze(0)
                
                # 添加到缓存
                if len(self.cache) >= self.cache_size:
                    # 简单的 FIFO 策略
                    self.cache.pop(next(iter(self.cache)))
                
                self.cache[token_id] = embedding
                embeddings_map[token_id] = embedding
        
        # 填充 embeddings
        inputs_embeds = inputs_embeds.clone()
        for i, (is_new, token_id) in enumerate(zip(is_new_token_mask.view(-1), input_ids.view(-1))):
            if is_new:
                inputs_embeds.view(-1, self.hidden_dim)[i] = embeddings_map[token_id.item()]
        
        return inputs_embeds
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }


if __name__ == '__main__':
    # 测试代码
    print("Testing DynamicToolEmbedding...")
    
    from src.models.profile_encoder import ProfileHyperNet
    
    # 参数
    vocab_size = 50000
    num_new_tokens = 100
    new_token_start_idx = vocab_size - num_new_tokens
    hidden_dim = 512
    profile_dim = 5
    
    # 创建原始 embedding
    original_embedding = nn.Embedding(vocab_size, hidden_dim)
    
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
    
    # 创建动态 embedding 层
    dynamic_embed = DynamicToolEmbedding(
        original_embedding=original_embedding,
        profile_encoder=profile_encoder,
        tool_semantics=tool_semantics,
        profiles=profiles,
        new_token_start_idx=new_token_start_idx
    )
    
    # 测试输入（混合普通 token 和新 token）
    input_ids = torch.tensor([
        [100, 200, new_token_start_idx, 300],  # 包含1个新token
        [new_token_start_idx + 5, new_token_start_idx + 10, 400, 500]  # 包含2个新token
    ])
    
    # 前向传播
    embeds = dynamic_embed(input_ids)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Output embeddings shape: {embeds.shape}")
    print(f"Embedding dtype: {embeds.dtype}")
    
    # 测试单个 token
    single_embed = dynamic_embed.get_embedding_for_token(new_token_start_idx)
    print(f"\nSingle token embedding shape: {single_embed.shape}")
    
    # 测试缓存版本
    print("\nTesting DynamicToolEmbeddingWithCache...")
    cached_embed = DynamicToolEmbeddingWithCache(
        original_embedding=original_embedding,
        profile_encoder=profile_encoder,
        tool_semantics=tool_semantics,
        profiles=profiles,
        new_token_start_idx=new_token_start_idx,
        cache_size=50
    )
    
    cached_embed.eval()
    
    # 多次前向传播测试缓存
    for _ in range(3):
        _ = cached_embed(input_ids)
    
    stats = cached_embed.get_cache_stats()
    print(f"Cache stats: {stats}")
