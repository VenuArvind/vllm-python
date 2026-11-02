import torch 
import torch.nn as nn
from typing import Tuple, Optional

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the tensor by half its dimension. 
    """

    d = x.shape[-1] # shape (Batch, Seq_len, D_head) -> D_head 
    x1 = x[..., :d//2] # (Batch, Seq_len, D_head /2)
    x2 = x[..., d//2:] # (Batch, Seq_len, D_head /2)

    return torch.cat((-x2, x1), dim = -1) # (Batch, Seq_len, D_head)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float=10000.0):
        """ 
        :param dim: (int) dimension of the model
        :param max_seq_len: (int) maximum sequence length
        :param base: (float) base for the rotary embeddings

        We calculate the inverse frequencies for the rotary embeddings, which are 
        used to rotate the query and key vectors in the attention mechanism.
        """
        super().__init__()
        self.dim = dim 
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim)) # shape (dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        
        t = torch.arange(self.max_seq_len, dtype=torch.float32) # shape (max_seq_len,)
        freqs = torch.outer(t, inv_freq) # shape (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1) # shape (max_seq_len, dim)
        self.register_buffer("cos_cache", emb.cos(), persistent = False) # shape (max_seq_len, dim)
        self.register_buffer("sin_cache", emb.sin(), persistent = False) # shape (max_seq_len, dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (torch.Tensor) input tensor
        :param positions: (torch.Tensor) positions of the input tensor
        :return: (Tuple[torch.Tensor, torch.Tensor]) cos and sin caches

        We get the cos and sin values from the cache based on the positions of the input tensor.
        """
        # x shape (Batch, Seq_len, Dim)
        cos = self.cos_cache[positions] # shape (Batch, Seq_len, Dim)
        sin = self.sin_cache[positions] # shape (Batch, Seq_len, Dim)

        cos = cos.unsqueeze(2) # shape (Batch, Seq_len, 1, Dim)
        sin = sin.unsqueeze(2) # shape (Batch, Seq_len, 1, Dim)
        
        return cos.to(x.dtype), sin.to(x.dtype) # (Batch, Seq_len, 1, Dim) for both cos and sin
        
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding to the tensor x.
    """

    return (x*cos) + (rotate_half(x) * sin)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
        """
        :param hidden_dim: (int) dimension of the model
        :param intermediate_dim: (int) dimension of the intermediate layer

        The feed forward network is a two-layer neural network that is used to transform the 
        hidden states of the model. 
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else 4 * hidden_dim

        self.fc1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias = False) # 
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (torch.Tensor) input tensor
        :return: (torch.Tensor) output tensor

        Calculate the output of the feed forward network.
        """

        return self.fc2(self.act(self.fc1(x)))

from model.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.self_attn = MultiHeadAttention(self.hidden_dim, self.num_heads)
        self.ln1 = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param x: (torch.Tensor) input tensor
        :param cos: (torch.Tensor) cos values for RoPE
        :param sin: (torch.Tensor) sin values for RoPE
        :param mask: (torch.Tensor) mask for attention
        :param kv_cache: (torch.Tensor) Key and Value cache for attention
        :return: (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) output tensor and kv cache

        Forward pass through the transformer block.
        """

        norm_x = self.ln1(x) # LayerNorm(x)
        attn_out, new_kv = self.self_attn(norm_x, cos, sin, mask, kv_cache) # MultiHeadAttention(LayerNorm(x))

        x = x + attn_out # Residual Connection

        norm_x = self.ln2(x) # LayerNorm(x)
        ffn_out = self.ffn(norm_x) # FeedForwardNetwork(LayerNorm(x))
        x = x + ffn_out # Residual Connection

        return x, new_kv

class GPTModel(nn.Module):
    def __init__(self,
        vocab_size: int = 50257,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048
    ):
        """
        :param vocab_size: (int) size of the vocabulary
        :param hidden_dim: (int) dimension of the model
        :param num_layers: (int) number of transformer layers
        :param num_heads: (int) number of attention heads
        :param max_seq_len: (int) maximum sequence length
        :param base: (float) base for the rotary embeddings
        :param dropout: (float) dropout rate
        :param bias: (bool) whether to use bias
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = self.hidden_dim // self.num_heads

        self.wte = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=self.max_seq_len)

        self.blocks = nn.ModuleList([TransformerBlock(self.hidden_dim, self.num_heads) for _ in range(self.num_layers)])

        self.ln_f = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None, kv_cache: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        """
        :param input_ids: (torch.Tensor) input tensor
        :param positions: (torch.Tensor) positions of the input tensor
        :param kv_cache: (list) Key and Value cache for attention
        :return: (Tuple[torch.Tensor, list]) output tensor and kv cache

        Forward pass through the GPT model.
        """

        batch_size, seq_len = input_ids.shape

        if positions is None:
            past_len = kv_cache[0][0].shape[2] if (kv_cache is not None and kv_cache[0] is not None) else 0
            positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        
        x_dummy = torch.empty(batch_size, seq_len, self.num_heads, self.head_dim, device=input_ids.device)
        cos, sin = self.rope(x_dummy, positions) # (Batch, Seq_len, 1, Dim) for both cos and sin

        mask = None
        if seq_len > 1:
            past_len = kv_cache[0][0].shape[2] if (kv_cache is not None and kv_cache[0] is not None) else 0
            total_len = past_len + seq_len
            mask = torch.full((seq_len, total_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=past_len + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]  
        
        x = self.wte(input_ids) # (Batch, Seq_len, Dim)

        new_kv = []

        for i, block in enumerate(self.blocks):
            kv_cache_i = kv_cache[i] if (kv_cache is not None and i < len(kv_cache)) else None
            x, new_kv_i = block(x, cos, sin, mask, kv_cache_i)
            new_kv.append(new_kv_i)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_kv
        
    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2") -> "GPTModel":
        """
        :param model_name: (str) name of the model
        :return: (GPTModel) instance of the model

        Load the model from the pre-trained weights.
        """

        from transformers import GPT2LMHeadModel 

        print(f"Loading weights from HuggingFace '{model_name}'...")
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf_model.state_dict()
        config = hf_model.config        

        model = cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.n_embd,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            max_seq_len=config.n_positions,
        )

        with torch.no_grad():
            # Token embeddings
            model.wte.weight.copy_(hf_sd["transformer.wte.weight"])

            # Iterate through layers
            for i in range(config.n_layer):
                prefix = f"transformer.h.{i}"
                block = model.blocks[i]

                #LAyer Norm 1
                block.ln1.weight.copy_(hf_sd[f"{prefix}.ln_1.weight"])
                block.ln1.bias.copy_(hf_sd[f"{prefix}.ln_1.bias"])

                qkv_w = hf_sd[f"{prefix}.attn.c_attn.weight"]
                q_w, k_w, v_w = qkv_w.split(config.n_embd, dim=1)
                block.self_attn.q_proj.weight.copy_(q_w.t())
                block.self_attn.k_proj.weight.copy_(k_w.t())
                block.self_attn.v_proj.weight.copy_(v_w.t())
                # If linear layers have biases, copy them too
                if block.self_attn.q_proj.bias is not None:
                    qkv_b = hf_sd[f"{prefix}.attn.c_attn.bias"]
                    q_b, k_b, v_b = qkv_b.split(config.n_embd, dim=0)
                    block.self_attn.q_proj.bias.copy_(q_b)
                    block.self_attn.k_proj.bias.copy_(k_b)
                    block.self_attn.v_proj.bias.copy_(v_b)
                # Attention: Out projection
                block.self_attn.o_proj.weight.copy_(hf_sd[f"{prefix}.attn.c_proj.weight"].t())
                if block.self_attn.o_proj.bias is not None:
                    block.self_attn.o_proj.bias.copy_(hf_sd[f"{prefix}.attn.c_proj.bias"])
                # LayerNorm 2
                block.ln2.weight.copy_(hf_sd[f"{prefix}.ln_2.weight"])
                block.ln2.bias.copy_(hf_sd[f"{prefix}.ln_2.bias"])
                # FeedForward Network (MLP)
                block.ffn.fc1.weight.copy_(hf_sd[f"{prefix}.mlp.c_fc.weight"].t())
                if block.ffn.fc1.bias is not None:
                    block.ffn.fc1.bias.copy_(hf_sd[f"{prefix}.mlp.c_fc.bias"])
                block.ffn.fc2.weight.copy_(hf_sd[f"{prefix}.mlp.c_proj.weight"].t())
                if block.ffn.fc2.bias is not None:
                    block.ffn.fc2.bias.copy_(hf_sd[f"{prefix}.mlp.c_proj.bias"])
            # 4. Final LayerNorm
            model.ln_f.weight.copy_(hf_sd["transformer.ln_f.weight"])
            model.ln_f.bias.copy_(hf_sd["transformer.ln_f.bias"])
            # 5. LM Head (shares weights with wte)
            model.lm_head.weight = model.wte.weight
        print("All weights copied successfully!")
        return model