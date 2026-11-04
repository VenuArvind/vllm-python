import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from model.transformer import GPTModel
from engine.inference import SimpleLLMEngine

# Request and Response Schema pydantic models 

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description = "The input text prompt")
    max_new_tokens: int = Field(default=30, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    request_id: str
    prompt: str
    generated_text: str
    num_tokens: int

# Global Serving State - request queue setup

request_queue: asyncio.Queue = asyncio.Queue()
active_futures: Dict[str, Tuple[asyncio.Future, str]] = {}
engine: SimpleLLMEngine = None


# Background loop that runs in the background - like the consumer part of the architecture 
async def background_engine_loop():
    """
    Runs continuously in the background:
    1. Drains new requests from request_queue into the engine.
    2. Steps the engine.
    3. Resolves futures for completed sequences.
    """
    while True:

        while not request_queue.empty():
            req, future = request_queue.get_nowait()
            seq_id = engine.add_request(req.prompt, max_new_tokens=req.max_new_tokens)
            active_futures[seq_id] = (future, req.prompt)

        if engine.scheduler.has_unfinished_sequences():
            engine.step()

            finished_seqs = list(engine.scheduler.finished)
            for seq in finished_seqs:
                if seq.seq_id in active_futures:
                    future, prompt = active_futures.pop(seq.seq_id)
                    if not future.done():
                        full_text = prompt + engine.tokenizer.decode(seq.output_token_ids)
                        future.set_result(
                            GenerateResponse(
                                request_id = seq.seq_id,
                                prompt = prompt,
                                generated_text = full_text,
                                num_tokens = len(seq.output_token_ids)
                            )
                        )
        
        await asyncio.sleep(0.005)
    

# Initialize the engine and start the backgorund loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine 
    print("Initializing engine...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPTModel(
        vocab_size=len(tokenizer),
        hidden_dim=768,
        num_layers=4,
        num_heads=12,
        max_seq_len=1024,   
    )

    engine = SimpleLLMEngine(
        model=model,
        tokenizer=tokenizer,
        num_blocks=128,
        block_size=16,
        max_batch_size=8, 
    )

    loop_task = asyncio.create_task(background_engine_loop())
    print("Serving engine is READY!")
    yield

    loop_task.cancel()
    

app = FastAPI(title = "Fault Tolerant vLLM Engine", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/stats")
async def stats():
    """Returns real-time engine metrics."""
    return {
        "free_blocks": engine.block_manager.num_free_blocks,
        "waiting_requests": len(engine.scheduler.waiting),
        "running_requests": len(engine.scheduler.running),
        "finished_requests": len(engine.scheduler.finished),
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    await request_queue.put((request, future))

    try:
        response = await asyncio.wait_for(future, timeout=60.0)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request generation timed out.")