import uuid
import torch
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer

from model.transformer import GPTModel
from engine.block_manager import BlockManager
from engine.scheduler import Sequence, ContinuousBatchingScheduler, SequenceStatus

class SimpleLLMEngine:
    def __init__(
        self,
        model: GPTModel,
        tokenizer: AutoTokenizer,
        num_blocks: int = 128,
        block_size: int = 16,
        max_batch_size: int = 4,
        device: str = "cpu",
    ):

        self.model = model 
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Continuous Batching

        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = ContinuousBatchingScheduler(self.block_manager, max_batch_size)

        # KV Cache

        self.seq_kv_cache: Dict[str, list] = {}

    def add_request(self, prompt: str, max_new_tokens: int = 50) -> str:
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        tokens = self.tokenizer.encode(prompt)

        seq = Sequence(
            seq_id = req_id, 
            prompt = prompt,
            max_new_tokens = max_new_tokens, 
            prompt_token_ids = tokens
        )

        self.scheduler.add_sequence(seq)
        return req_id

    def step(self, temperature: float = 0.0) -> Dict[str, str]:
        """
        Executes a single iteration step across all currently scheduled sequences.
        Returns a dictionary of {seq_id: latest_generated_token_str}.
        """
        active_batch = self.scheduler.schedule()
        if not active_batch:
            return {}
        
        next_tokens: Dict[str, str] = {}
        newly_generated_strings: Dict[str, str] = {}

        with torch.no_grad():
            for seq in active_batch:
                past_cache = self.seq_kv_cache.get(seq.seq_id, None)

                if past_cache is None:
                    # PREFILL - feed everything
                    input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)
                else:
                    # DECODE - feed only the last token
                    input_ids = torch.tensor([[seq.last_token_id]], device=self.device)
                
                # Forward pass through model 
                logits, new_kv = self.model(input_ids=input_ids, kv_cache=past_cache)
                self.seq_kv_cache[seq.seq_id] = new_kv

                # Next token sampling
                last_logits = logits[0, -1, :]

                if temperature > 0.0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token_id = torch.argmax(last_logits, dim=-1).item()
                
                next_tokens[seq.seq_id] = next_token_id
                newly_generated_strings[seq.seq_id] = self.tokenizer.decode([next_token_id])
        
        self.scheduler.post_step(next_tokens, eos_token_id=self.tokenizer.eos_token_id)

        for seq in self.scheduler.finished:
            if seq.seq_id in self.seq_kv_cache:
                del self.seq_kv_cache[seq.seq_id]
        
        return newly_generated_strings
    
    def generate(self, prompts: Union[str, list[str]], max_new_tokens: int=50, temperature: float=0.0) -> Dict[str, str]:
        """
        Runs the engine step-by-step until all prompts complete generation.
        Returns {prompt: completed_text}.
        """

        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Add all requests
        req_to_prompt = {}
        for p in prompts:
            req_id = self.add_request(p, max_new_tokens)
            req_to_prompt[req_id] = p
        
        #Step loop until all requests are finished 
        while self.scheduler.has_unfinished_sequences():
            self.step(temperature=temperature)
        
        results = {}

        for seq in self.scheduler.finished:
            if seq.seq_id in req_to_prompt:
                full_text = seq.prompt + self.tokenizer.decode(seq.output_token_ids)
                results[seq.prompt] = full_text
        
        return results 