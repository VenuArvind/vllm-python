import torch 
from typing import Tuple, Optional 

class StaticKVCache:
    def __init__(
        self,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048, 
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Pre-allocated KV cache for a single attention layer
        """
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.zeros(
            (max_batch_size, num_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype
        )

        self.v_cache = torch.zeros(
            (max_batch_size, num_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype
        )

        self.current_len = 0
    
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor, start_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache with the new key and value tensors.

        Args:
            new_k: New key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            new_v: New value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            start_pos: Optional starting position
        
        Returns:
            Tuple of (key_cache, value_cache)
        """
        batch_size, num_heads, seq_len, head_dim = new_k.shape

        if start_pos is None:
            start_pos = self.current_len
        
        end_pos = start_pos + seq_len
        assert end_pos <= self.max_seq_len, f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}"

        self.k_cache[:batch_size, :, start_pos:end_pos, :] = new_k
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = new_v

        self.current_len = end_pos 

        active_k = self.k_cache[:batch_size, :, :end_pos, :]
        active_v = self.v_cache[:batch_size, :, :end_pos, :]

        return active_k, active_v

    def reset(self):
        self.current_len = 0

class MultiSeqKVCache:
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Manages KV Cache across multiple layers of the transformer model.
        """

        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.zeros(
            (num_layers, max_batch_size, num_heads, max_seq_len, head_dim),
            device = device,
            dtype = dtype
        )

        self.v_cache = torch.zeros(
            (num_layers, max_batch_size, num_heads, max_seq_len, head_dim),
            device = device,
            dtype = dtype
        )

        self.seq_lens = torch.zeros(max_batch_size, device = device, dtype = torch.long)

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV Cache with new Key and Value tensors.
        """

        batch_size, num_heads, seq_len, head_dim = new_k.shape

        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device = self.device)

        for i, b_idx in enumerate(batch_indices):
            start_pos = self.seq_lens[b_idx].item()
            end_pos = start_pos + seq_len
            assert end_pos <= self.max_seq_len, f"Sequence length {end_pos} exceeds {self.max_seq_len}"

            self.k_cache[layer_idx, b_idx, :, start_pos:end_pos, :] = new_k[i]
            self.v_cache[layer_idx, b_idx, :, start_pos:end_pos, :] = new_v[i]
        
        if layer_idx == self.num_layers - 1:
            self.seq_lens[batch_indices] +=seq_len
        
        max_active_len = self.seq_lens[batch_indices].max().item()
        if layer_idx != self.num_layers - 1:
            max_active_len += seq_len
        
        active_k = self.k_cache[layer_idx, batch_indices, :, :max_active_len, :]
        active_v = self.v_cache[layer_idx, batch_indices, :, :max_active_len, :]
        return active_k, active_v

    def reset_sequence(self, batch_idx: int):
        self.seq_lens[batch_idx] = 0
    
    def reset_all(self):
        self.seq_lens.zero_()

