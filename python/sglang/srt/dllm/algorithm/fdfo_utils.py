from typing import Callable, List, Tuple
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

def fdfo_post_process(
    forward_batch: ForwardBatch,
    full_logits: torch.Tensor,
    pick_tokens_func: Callable[[ForwardBatch, torch.Tensor], None],
    mask_id: int,
    block_size: int,
) -> Tuple[List[List[int]], List[int]]:
    """
    Post-forward processing: pick tokens and build next token IDs list.
    This function combines:
        1. Batch size validation
        2. Mask count computation (before input_ids modification)
        3. Token picking based on confidence
        4. Building next token IDs list with decode markers
    Args:
        forward_batch: The forward batch containing input_ids
        full_logits: Full logits from model forward pass
        pick_tokens_func: Function to pick tokens based on logits
        mask_id: The mask token ID
        block_size: The block size for reshaping
    Returns:
        List of token ID lists, with -1 appended for decode blocks
    """
    # Validate batch size
    batch_size = forward_batch.batch_size
    expected_batch_size = forward_batch.input_ids.shape[0] // block_size
    if batch_size != expected_batch_size:
        raise RuntimeError(
            f"Batch size mismatch: forward_batch.batch_size={batch_size}, "
            f"but input_ids shape {forward_batch.input_ids.shape[0]} / "
            f"block_size {block_size} = {expected_batch_size}"
        )
    # Compute mask counts per block BEFORE pick_tokens_func modifies input_ids
    # Convert mask counts to CPU once to avoid multiple D2H transfers
    mask_counts_cpu = (
        (forward_batch.input_ids == mask_id)
        .view(batch_size, block_size)
        .sum(dim=1)
        .tolist()
    )
    # Update input_ids based on confidence
    pick_tokens_func(forward_batch, full_logits)
    # Build output token IDs list with decode markers
    # Reshape and convert to CPU list in one operation
    next_token_ids = forward_batch.input_ids.view(batch_size, block_size).tolist()
    next_token_ids_list = []
    accept_length_per_req_cpu = []
    for i in range(batch_size):
        next_token_ids_list.append(next_token_ids[i])
        accept_length_per_req_cpu.append(block_size if mask_counts_cpu[i] == 0 else 0)
        
    return next_token_ids_list, accept_length_per_req_cpu