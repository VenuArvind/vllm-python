import math
from typing import Dict, List, Optional 

class BlockManager:
    def __init__(
        self,
        num_blocks: int, block_size: int = 16
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        self.free_blocks: List[int] = list(range(num_blocks))

        self.req_to_blocks: Dict[str, List[int]] = {}

        self.req_to_token_count: Dict[str, int] = {}

    def can_allocate(self, num_tokens: int) -> bool:

        required_blocks = math.ceil(num_tokens / self.block_size)

        return len(self.free_blocks) >= required_blocks
    
    def allocate(self, req_id: str, num_tokens: int) -> List[int]:
        """
        Allocates blocks for a request.
        """

        required_blocks = math.ceil(num_tokens / self.block_size)
        assert self.can_allocate(num_tokens), "Not enough memory to allocate blocks"

        allocated = [self.free_blocks.pop(0) for _ in range(required_blocks)]

        self.req_to_blocks[req_id] = allocated
        self.req_to_token_count[req_id] = num_tokens

        return allocated

    def append_slot(self, req_id: str) -> Optional[int]:
        """
        Called when a request needs one more token
        """

        assert req_id in self.req_to_blocks, f"Request {req_id} not found."

        current_tokens = self.req_to_token_count[req_id]
        
        if current_tokens % self.block_size == 0:
            assert len(self.free_blocks) > 0, "Out of blocks! Preemption required."
            new_block = self.free_blocks.pop(0)
            self.req_to_blocks[req_id].append(new_block)
            self.req_to_token_count[req_id] += 1
            return new_block
        else:
            self.req_to_token_count[req_id] += 1
            return None
    
    def free_sequence(self, req_id: str):
        """
        Free memory associated with a request
        """
        
        if req_id in self.req_to_blocks:
            blocks_to_free = self.req_to_blocks.pop(req_id)
            self.free_blocks.extend(blocks_to_free)
            self.req_to_token_count.pop(req_id, None)

    def get_block_table(self, req_id: str) -> List[int]:
        """
        Returns the list of blocks for a request.
        """
        return self.req_to_blocks.get(req_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)