# Fault-Tolerant LLM Serving Engine

A lightweight LLM serving engine built from scratch in pure Python and PyTorch.

---

## What Has Been Built So Far

### 1. Rotary Positional Embeddings (RoPE) — `model/transformer.py`
* Relative positional embedding mechanism using 2D rotational transformations on query and key vectors.
* Supports arbitrary token offset indexing for continuous batching.

### 2. Causal Multi-Head Attention — `model/attention.py`
* Multi-head attention layer with Q, K, V projections and output mixing.
* Built-in support for RoPE and past key-value (KV) caching for autoregressive decoding.

### 3. Transformer Block & Feed-Forward Network — `model/transformer.py`
* Standard MLP with GELU activation.
* Pre-Layer Normalization architecture with residual connections.

### 4. GPT Model — `model/transformer.py`
* Decoder-only Transformer model assembling token embeddings, stacked Transformer blocks, final LayerNorm, and tied language model head.
* Verified for both prompt prefill and iterative single-token generation with KV-caching.

---

## Project Structure

```text
vllm-python/
├── model/
│   ├── attention.py       # MultiHeadAttention with RoPE and KV-cache support
│   └── transformer.py     # RoPE, FFN, TransformerBlock, and GPTModel
├── engine/                # KV-Cache & BlockManager (In Progress)
├── serving/               # FastAPI serving layer
├── distributed/           # Fault tolerance & state sync
├── requirements.txt
└── README.md
```
