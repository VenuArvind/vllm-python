import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.transformer import apply_rotary_pos_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor = None, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor,Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward Pass of MultiHeadAttention
        Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        cos: Cosine embeddings of shape (batch_size, seq_len, head_dim)
        sin: Sine embeddings of shape (batch_size, seq_len, head_dim)
        mask: Attention mask of shape (batch_size, seq_len, seq_len)
        kv_cache: Optional KV cache of shape (batch_size, seq_len, hidden_dim)
        Returns:
        Tuple of (output tensor, updated KV cache)
        """
        
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x) # shape [batch_size, seq_len, hidden_dim]
        k = self.k_proj(x) # shape [batch_size, seq_len, hidden_dim]
        v = self.v_proj(x) # shape [batch_size, seq_len, hidden_dim]

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim) # shape [batch_size, seq_len, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim) # shape [batch_size, seq_len, num_heads, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim) # shape [batch_size, seq_len, num_heads, head_dim]

        q = apply_rotary_pos_emb(q, cos, sin) # shape [batch_size, seq_len, num_heads, head_dim]
        k = apply_rotary_pos_emb(k, cos, sin) # shape [batch_size, seq_len, num_heads, head_dim]

        q = q.transpose(1,2) # shape [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1,2) # shape [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1,2) # shape [batch_size, num_heads, seq_len, head_dim]

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim = 2) # shape [batch_size, num_heads, total_kv_len, head_dim]
            v = torch.cat([v_cache, v], dim = 2) # shape [batch_size, num_heads, total_kv_len, head_dim]
        
        current_kv_cache = (k,v)

        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5) # shape [batch_size, num_heads, seq_len, total_kv_len]

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim = -1) # shape [batch_size, num_heads, seq_len, total_kv_len]
        
        output = torch.matmul(attn_weights, v) # shape [batch_size, num_heads, seq_len, head_dim]
        
        output = output.transpose(1,2).contiguous() # shape [batch_size, seq_len, num_heads, head_dim]
        
        output = output.view(batch_size, seq_len, self.hidden_dim) # shape [batch_size, seq_len, hidden_dim]
        
        output = self.o_proj(output) # shape [batch_size, seq_len, hidden_dim]
        
        return output, current_kv_cache