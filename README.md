# Fault-Tolerant LLM Serving Engine (from Scratch)

A lightweight, high-performance LLM serving engine built in pure Python and PyTorch from first principles.

---

## 🛠️ Implemented Components (Phase 1: The Transformer Core)

### 1. Rotary Positional Embeddings (RoPE) — [`model/transformer.py`](model/transformer.py)
* **`rotate_half(x)`**: Rotates the last dimension of Query/Key tensors by grouping features into pairs and applying a $90^\circ$ rotation: $[-x_2, x_1]$.
* **`RotaryEmbedding`**: Computes inverse frequency vectors $\theta_i = \text{base}^{-2i/d}$, caches $\cos$ and $\sin$ tables up to `max_seq_len`, and indexes into positions dynamically to support non-zero starting offsets.
* **`apply_rotary_pos_emb(x, cos, sin)`**: Vectorized rotation applying $(x \odot \cos) + (\text{rotate\_half}(x) \odot \sin)$ to queries and keys.

### 2. Causal Multi-Head Attention — [`model/attention.py`](model/attention.py)
* **`MultiHeadAttention`**:
  * Linear projections: $W_q, W_k, W_v \in \mathbb{R}^{H \times H}$ and $W_o \in \mathbb{R}^{H \times H}$.
  * Tensor reshaping and transposition to head-first format: `[batch_size, num_heads, seq_len, head_dim]`.
  * Integrates RoPE on $Q$ and $K$ across all heads.
  * Supports autoregressive KV-caching via sequence-dimension concatenation (`dim=2`).
  * Scaled dot-product attention: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{mask}\right) V$.
  * Merges heads and projects back to `hidden_dim`.

### 3. FeedForward Network (MLP) — [`model/transformer.py`](model/transformer.py)
* **`FeedForwardNetwork`**:
  * Two-layer projection ($H \to 4H \to H$) with `approximate="tanh"` GELU activation.

### 4. Transformer Block — [`model/transformer.py`](model/transformer.py)
* **`TransformerBlock`**:
  * Modern Pre-Layer Normalization architecture with residual skip connections:
    $$x_1 = x + \text{Attention}(\text{LN}_1(x))$$
    $$x_2 = x_1 + \text{FFN}(\text{LN}_2(x_1))$$
  * Propagates and updates layer-specific `(k, v)` cache states.

### 5. GPT Model Assembly — [`model/transformer.py`](model/transformer.py)
* **`GPTModel`**:
  * Token embeddings lookup table: `wte` ($V \times H$).
  * Dynamic RoPE position assignment handling both prefill ($S > 1$) and single-token decode ($S = 1$) modes.
  * Causal lower-triangular attention masking during prompt prefill.
  * Stack of $L$ transformer blocks (`nn.ModuleList`).
  * Final layer normalization (`ln_f`) and un-embedding linear head (`lm_head`).
  * Weight tying between `wte` and `lm_head`.

---

## 📁 Current Project Layout

```text
vllm-python/
├── model/
│   ├── attention.py       # MultiHeadAttention with RoPE and KV-caching
│   └── transformer.py     # RoPE, FFN, TransformerBlock, and GPTModel
├── engine/                # (Next: KV-Cache & BlockManager)
├── serving/               # (Next: FastAPI serving layer)
├── distributed/           # (Next: Supervisor WAL & Redis state sync)
├── tests/                 # Unit and integration test suites
├── requirements.txt       # Environment dependencies
└── README.md
```

---

## 🧪 Verified Capabilities

* **Full Prefill Forward Pass:** Input `[batch_size, seq_len]` $\to$ Logits `[batch_size, seq_len, vocab_size]` verified.
* **Autoregressive Single-Token Step:** Generates next-token logits `[batch_size, 1, vocab_size]` and accumulates KV cache across all layers.
