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

### 4. GPT Model & HuggingFace Weight Loading — `model/transformer.py`
* Decoder-only Transformer model assembling token embeddings, stacked Transformer blocks, final LayerNorm, and tied language model head.
* HuggingFace weight importer (`from_pretrained`) translating OpenAI parameters into our custom architecture.
* Verified for both prompt prefill and iterative single-token generation.

### 5. Pre-Allocated KV-Cache — `model/kv_cache.py`
* **`StaticKVCache`**: Eliminates `torch.cat` reallocations by writing keys and values into pre-allocated static tensors in constant $\mathcal{O}(1)$ time.
* **`MultiSeqKVCache`**: Multi-layer, multi-sequence cache tracking independent sequence lengths per request.

### 6. Paged Memory Block Manager — `engine/block_manager.py`
* Operating system-style virtual memory paging dividing KV-cache VRAM into fixed physical blocks (e.g. 16 tokens/block).
* Maps logical sequence tokens to non-contiguous physical blocks via page tables.
* Eliminates internal and external GPU memory fragmentation.

### 7. Dual Schedulers (Static vs. Continuous) — `engine/scheduler.py`
* **`Sequence` State Machine**: Tracks request lifecycles across `WAITING`, `RUNNING`, and `FINISHED`.
* **`StaticBatchingScheduler`**: Traditional batching baseline that locks a batch until all sequences finish.
* **`ContinuousBatchingScheduler`**: Iteration-level continuous batching with instant finished sequence eviction, zero-delay waiting request admission, and memory preemption protection.

### 8. The Inference Engine (`SimpleLLMEngine`) — `engine/inference.py`
* Master runtime orchestrator uniting the custom model, tokenizer, block manager, and continuous batching scheduler.
* Decoupled **prefill** (full prompt processing) and **decode** (single-token generation) execution passes.
* Verified multi-prompt concurrent batching with automatic physical block recycling upon sequence completion.

---

## Project Structure

```text
vllm-python/
├── model/
│   ├── attention.py       # MultiHeadAttention with RoPE and KV-cache support
│   ├── kv_cache.py        # StaticKVCache & MultiSeqKVCache memory managers
│   └── transformer.py     # RoPE, FFN, TransformerBlock, and GPTModel
├── engine/
│   ├── block_manager.py   # Paged memory block allocator & page tables
│   ├── scheduler.py       # Static & Continuous Batching schedulers
│   └── inference.py       # SimpleLLMEngine runtime orchestrator
├── serving/               # FastAPI serving layer (Next)
├── distributed/           # Fault tolerance & state sync
├── tests/                 # Unit & integration test suites
├── requirements.txt
└── README.md
```
