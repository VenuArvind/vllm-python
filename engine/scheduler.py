from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
from engine.block_manager import BlockManager

class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"

@dataclass
class Sequence:
    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    max_new_tokens: int = 50
    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    
    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)
    
    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        
        return self.prompt_token_ids[-1]
    
    def is_finished(self, eos_token: int = 50256) -> bool:
        if len(self.output_token_ids) >= self.max_new_tokens:
            return True

        if self.output_token_ids and self.output_token_ids[-1] == eos_token:
            return True

        return False
    
class StaticBatchingScheduler:
    """
    Simple scheduler that processes all available sequences in a single batch.
    """
    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []

    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> List[Sequence]:
        if not self.running and self.waiting:
            batch_to_run = self.waiting[:self.max_batch_size]
            self.waiting = self.waiting[self.max_batch_size:]
            for seq in batch_to_run:
                seq.status = SequenceStatus.RUNNING
            self.running = batch_to_run
        
        return self.running

    def post_step(self, next_tokens: dict[str, int], eos_token_id: int = 50256):
        still_running = []
        for seq in self.running:
            if seq.seq_id in next_tokens:
                token_id = next_tokens[seq.seq_id]
                seq.output_token_ids.append(token_id)
            
            if seq.is_finished(eos_token_id):
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                still_running.append(seq)
        
        self.running = still_running

    def has_unfinished_sequences(self) -> bool:
        return len(self.running) > 0 or len(self.waiting) > 0

class ContinuousBatchingScheduler:
    """
    Iteration-Level Continuous Batching Scheduler:
    - Admits waiting sequences as soon as physical blocks become free.
    - Evicts finished sequences immediately on every forward step.
    - Preempts younger requests if physical memory blocks run out.
    """
    def __init__(self, block_manager: BlockManager, max_batch_size: int = 4):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []

    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)
    
    def schedule(self) -> List[Sequence]:
        """
        Runs on every token step:
        1. Checks memory for currently running requests (preempts if needed).
        2. Admits waiting requests into running if blocks are available.
        """

        # 1. Preempt most recent running requests if memory is insufficient
        while self.running and self.block_manager.num_free_blocks < len(self.running):
            preempted_seq = self.running.pop()
            preempted_seq.status = SequenceStatus.WAITING
            self.block_manager.free_sequence(preempted_seq.seq_id)
            self.waiting.insert(0, preempted_seq)

        # 2. Admission check 
        while self.waiting and len(self.running) < self.max_batch_size:
            next_seq = self.waiting[0]

            if self.block_manager.can_allocate(next_seq.total_len):
                self.waiting.pop(0)
                self.block_manager.allocate(next_seq.seq_id, next_seq.total_len)
                next_seq.status = SequenceStatus.RUNNING
                self.running.append(next_seq)
            else:
                break
        
        return self.running

    def post_step(self, next_tokens: dict[str, int], eos_token_id: int = 50256):
        """
        Updates running sequences with newly generated tokens,
        grows their block allocations, and frees finished requests immediately.
        """
        still_running = []
        for seq in self.running:
            if seq.seq_id in next_tokens:
                token_id = next_tokens[seq.seq_id]
                seq.output_token_ids.append(token_id)
                self.block_manager.append_slot(seq.seq_id)
            
            if seq.is_finished(eos_token_id):
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
                self.block_manager.free_sequence(seq.seq_id)
            else:
                still_running.append(seq)
            
        self.running = still_running

    def has_unfinished_sequences(self) -> bool:
        return len(self.running) > 0 or len(self.waiting) > 0        