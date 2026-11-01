# Fault-Tolerant LLM Serving Engine in Python (From Scratch)

This document is your line-by-line blueprint for building a complete, fault-tolerant vLLM implementation in pure Python. It is structured around 7 core phases: the 5 foundational phases of the `oms0401/vLLM-from-scratch` repository, plus 2 advanced phases for distributed fault tolerance.

## 1. High-Level Architecture

The system is designed to bypass the Python Global Interpreter Lock (GIL) and handle asynchronous client requests alongside heavy GPU workloads, ensuring that no single node failure drops an active generation.

*   **API Gateway (FastAPI):** Routes HTTP requests to the engine.
*   **The Supervisor (Fault Tolerance):** Monitors worker health and maintains a Write-Ahead Log (WAL) of incoming requests. 
*   **The Scheduler:** Manages `WAITING`, `RUNNING`, and `FINISHED` queues using Continuous Batching.
*   **KV-Cache & Block Manager:** A virtual memory manager that allocates and frees VRAM chunks (blocks) dynamically to prevent fragmentation (PagedAttention).
*   **Model Executor:** A custom GPT-2 style Transformer with Rotary Positional Embeddings (RoPE) that computes the forward passes.
*   **State Sync (Redis):** Streams PagedAttention block states to a central cache so dead workers can be replaced instantly without losing generation progress.

## 2. Low-Level Architecture & File Structure

Create a new directory for your project and set up this exact file structure. You will code each of these from scratch:

```text
vllm-fault-tolerant/
├── model/
│   ├── attention.py       # MultiHeadAttention, PagedAttention logic
│   ├── kv_cache.py        # Static KVCache, MultiSeqKVCache classes
│   └── transformer.py     # GPTModel, RotaryEmbedding, FFN, TransformerBlock
├── engine/
│   ├── block_manager.py   # BlockManager, memory paging, logical-to-physical mapping
│   ├── scheduler.py       # ContinuousBatchingScheduler, sequence state tracking
│   └── inference.py       # SimpleLLMEngine orchestration (the `step()` loop)
├── serving/
│   └── api.py             # FastAPI server, request/response models
├── distributed/           # (NEW for Fault Tolerance)
│   ├── supervisor.py      # Multiprocessing watchdog, request Write-Ahead Log (WAL)
│   └── state_sync.py      # Redis client to backup and restore BlockManager state
└── main.py                # Test suite and entry point
```

## 3. What You Need to Learn Before Starting

1.  **PyTorch Tensor Manipulation:** You must be highly comfortable with `.view()`, `.reshape()`, tensor broadcasting, and matrix multiplication (`torch.matmul`).
2.  **RoPE (Rotary Positional Embeddings):** Understand how to apply complex rotations to query and key vectors instead of adding static absolute positions.
3.  **OS Virtual Memory:** PagedAttention is exactly like OS memory paging. You need to understand logical addresses (token indices) mapping to physical addresses (tensor block indices).
4.  **Python `asyncio` and `multiprocessing`:** To ensure the FastAPI event loop is not blocked by a 500ms matrix multiplication.
5.  **Redis Data Structures:** For Phase 7, you'll need to know how to serialize and stream PyTorch tensors into Redis efficiently.

---

## 4. Step-by-Step Implementation Guide

### Phase 1: The Transformer Model (`model/transformer.py`, `model/attention.py`)
*   **Implement RoPE:** Code the `RotaryPositionalEmbedding` class.
*   **Implement Attention:** Code standard `MultiHeadAttention`. It should accept an optional `kv_cache` argument to avoid recomputing past tokens.
*   **Build the Model:** Assemble the `FeedForwardNetwork`, `TransformerBlock`, and the final `GPTModel` class. Ensure you can load standard GPT-2 weights from HuggingFace into your custom architecture.

### Phase 2: The KV-Cache (`model/kv_cache.py`)
*   **Static Pre-allocation:** Instead of `torch.cat` (which forces a full memory copy every time a token is generated), create pre-allocated zero-tensors based on max context length.
*   **Tracking:** Build a `MultiSeqKVCache` class that tracks the current sequence length for a batch of requests and writes new keys/values directly into the pre-allocated indices in constant time.

### Phase 3: PagedAttention & BlockManager (`engine/block_manager.py`)
*   **The Block Pool:** Divide your KV-Cache into fixed chunks (e.g., 16 tokens per block). Initialize a list of `free_blocks = [0, 1, 2, ..., N]`.
*   **Logical to Physical:** When a prompt of 40 tokens arrives, allocate `ceil(40/16) = 3` blocks. Pop them from `free_blocks` and store them in a `req_to_blocks` dictionary.
*   **Freeing:** Write a `free_sequence(seq_id)` method that returns the blocks to `free_blocks` when a request hits `[EOS]`. 

### Phase 4: Continuous Batching Scheduler (`engine/scheduler.py`)
*   **State Machine:** Create a `Sequence` dataclass with states `WAITING`, `RUNNING`, and `FINISHED`.
*   **The Queue:** Build a `ContinuousBatchingScheduler` class. On every single forward pass, check if `WAITING` requests can fit into the remaining free blocks. If yes, move them to `RUNNING` and allocate blocks.
*   **Preemption:** If you run out of blocks mid-generation, pause the newest request, return it to `WAITING`, and free its blocks so older requests can finish.

### Phase 5: The Inference Engine (`engine/inference.py`)
*   **The Orchestrator:** Build the `SimpleLLMEngine`. This ties it all together. 
*   **The `step()` function:** 
    1. Ask the scheduler for the next batch.
    2. Read from `block_manager` to get physical block locations.
    3. Run a forward pass on `GPTModel`.
    4. Sample the next token (argmax or temperature sampling).
    5. Check for `[EOS]` or `max_tokens` limits. 
    6. Tell the scheduler which sequences just `FINISHED`.

### Phase 6: FastAPI Server (`serving/api.py`)
*   **Endpoints:** Build a POST `/generate` endpoint.
*   **Thread Safety:** Wrap your `SimpleLLMEngine` in a background thread or a separate process. The FastAPI endpoint should push incoming JSON prompts into a Python `Queue`, which the `SimpleLLMEngine` constantly reads from.

### Phase 7: Fault Tolerance & Redis (The Upgrade)
*   **The Supervisor (`distributed/supervisor.py`):** Use `multiprocessing.Process` to spawn your engine. The supervisor writes all incoming requests to an append-only file (WAL). If the engine process dies (OOM, segfault), the supervisor restarts the process and re-injects the WAL.
*   **Redis State Sync (`distributed/state_sync.py`):** As the `SimpleLLMEngine` generates tokens, have it asynchronously serialize its `BlockManager` state (`req_to_blocks` mappings) and physical KV tensors to a local Redis instance.
*   **Recovery:** Modify the engine startup logic: before starting a fresh KV-cache, check Redis. If a crashed session exists, download the tensors directly into GPU memory, skipping the expensive "prefill" phase for all requests that were interrupted mid-generation.