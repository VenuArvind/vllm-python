# Fault-Tolerant LLM Serving Engine (from Scratch)

A lightweight, educational, yet high-performance LLM serving engine built in pure Python and PyTorch from first principles. Inspired by [vLLM](https://github.com/vllm-project/vllm), this project implements PagedAttention, Continuous Batching, KV-Cache memory management, an asynchronous HTTP serving runtime, and an advanced distributed fault-tolerance layer with Redis and Write-Ahead Logging (WAL).

---

## 🌟 Architecture Overview

```text
                               +-----------------------------+
                               |     Client HTTP Request     |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |    FastAPI Serving Gateway   |
                               +-----------------------------+
                                              |
                                              v
    +-----------------------------------------------------------------------+
    |                             SUPERVISOR                                |
    |      * Multiprocessing Process Manager (Heartbeat & Watchdog)         |
    |      * Append-Only Write-Ahead Log (WAL) for Request Durability      |
    +-----------------------------------------------------------------------+
                        |                                   ^
       Spawns / Manages |                                   | Syncs state / WAL
                        v                                   v
    +-----------------------------------------------------------------------+
    |                         ENGINE WORKER PROCESS                         |
    |                                                                       |
    |  +-----------------------------------------------------------------+  |
    |  | Continuous Batching Scheduler                                   |  |
    |  | - Manages WAITING, RUNNING, and FINISHED sequence states        |  |
    |  | - Preempts & reschedules requests dynamically                   |  |
    |  +-----------------------------------------------------------------+  |
    |                                   |                                   |
    |                                   v                                   |
    |  +-----------------------------------------------------------------+  |
    |  | PagedAttention & BlockManager                                   |  |
    |  | - Logical-to-Physical page table mapping                        |  |
    |  | - Non-contiguous VRAM page allocation to eliminate fragmentation|  |
    |  +-----------------------------------------------------------------+  |
    |                                   |                                   |
    |                                   v                                   |
    |  +-----------------------------------------------------------------+  |
    |  | Custom Transformer Execution Engine                             |  |
    |  | - GPT-2 Architecture with Rotary Positional Embeddings (RoPE)   |  |
    |  | - Static & Paged KV-Cache access                                |  |
    |  +-----------------------------------------------------------------+  |
    +-----------------------------------------------------------------------+
                        |                                   ^
            Async state |                                   | Fast snapshot
            snapshots   v                                   | restore
    +-----------------------------------------------------------------------+
    |                         REDIS KV-STATE STORE                          |
    |      * Physical KV-Cache page serializations & logical mappings       |
    |      * Instant failover recovery without prefill recomputation        |
    +-----------------------------------------------------------------------+
```

---

## 📁 Repository Layout

```text
vllm-fault-tolerant/
├── model/
│   ├── attention.py       # MultiHeadAttention & PagedAttention tensor operations
│   ├── kv_cache.py        # Static and dynamic multi-sequence KV-Cache
│   └── transformer.py     # RoPE, FFN, TransformerBlock, and GPTModel definitions
├── engine/
│   ├── block_manager.py   # Logical-to-physical block table allocator
│   ├── scheduler.py       # Continuous batching scheduler & preemption logic
│   └── inference.py       # SimpleLLMEngine step loop orchestration
├── serving/
│   └── api.py             # FastAPI async gateway, endpoints, & background queues
├── distributed/
│   ├── supervisor.py      # Worker watchdog process & Write-Ahead Log (WAL)
│   └── state_sync.py      # Redis client for KV-cache and page table replication
├── tests/                 # Unit and integration test suites for every phase
├── requirements.txt       # Environment dependencies
└── main.py                # Standalone test runner and CLI entrypoint
```

---

## 🚀 Implementation Roadmap (7 Phases)

### 🔹 Phase 1: The Custom Transformer (`model/`)
- **Rotary Positional Embeddings (RoPE):** Complex rotation matrices for relative position encoding.
- **Causal Multi-Head Attention:** Multi-head scaled dot-product attention with support for past KV-caching.
- **GPT Model Assembly:** FeedForward networks, LayerNorm, stacked Transformer blocks, and loading pre-trained weights from HuggingFace.

### 🔹 Phase 2: Static KV-Caching (`model/kv_cache.py`)
- **Eliminating `torch.cat` Overhead:** Pre-allocate static zeroed KV tensors in GPU/CPU memory to avoid constant tensor reallocation.
- **Batch Tracking (`MultiSeqKVCache`):** Manage slot offsets and write new key/value representations in $\mathcal{O}(1)$ time.

### 🔹 Phase 3: PagedAttention & Block Manager (`engine/block_manager.py`)
- **Virtual Memory Paging:** Partition KV-cache memory into fixed-size physical blocks (e.g., 16 tokens/block).
- **Block Allocation & Freeing:** Logical token indices dynamically mapped to arbitrary physical block IDs.
- **Zero-Fragmentation Guarantee:** Prevent GPU Out-Of-Memory (OOM) caused by memory fragmentation.

### 🔹 Phase 4: Continuous Batching Scheduler (`engine/scheduler.py`)
- **Sequence Lifecycle Tracking:** States for `WAITING`, `RUNNING`, and `FINISHED`.
- **Iteration-Level Scheduling:** Re-evaluate the batch on every generated token rather than waiting for an entire batch to finish.
- **Request Preemption:** Gracefully swap or preempt lower-priority/newer requests when physical blocks are exhausted.

### 🔹 Phase 5: The Inference Engine (`engine/inference.py`)
- **Engine Step Loop (`SimpleLLMEngine`):** Tie scheduling, block table lookup, model forward execution, and token sampling together.
- **Generation Control:** Nucleus/temperature sampling, EOS termination, and streaming output buffers.

### 🔹 Phase 6: Asynchronous Serving API (`serving/api.py`)
- **FastAPI Endpoint:** POST `/generate` interface compatible with modern streaming clients.
- **Decoupled Architecture:** Thread/process-isolated queues separating HTTP I/O from compute-heavy inference loops.

### 🔹 Phase 7: Distributed Fault Tolerance & Redis State Sync (`distributed/`)
- **Process Supervisor & WAL:** An append-only Write-Ahead Log to record incoming requests and automatically reboot workers on crashes.
- **Redis KV State Persistence:** Stream active page tables and KV-cache blocks to Redis.
- **Instant Failover:** Rebooted workers load cache blocks from Redis, skipping prefill computations on resumed generations.

---

## 🛠️ Prerequisites & Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch (Apple Silicon MPS, CUDA, or CPU)
- **Web Framework:** FastAPI, Uvicorn
- **Distributed State:** Redis, `redis-py`
- **Tokenizer / Utilities:** Hugging Face `transformers`, `tiktoken`
- **Testing:** `pytest`

