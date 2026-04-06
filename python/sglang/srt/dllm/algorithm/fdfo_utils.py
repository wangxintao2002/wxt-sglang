from typing import Callable, List, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def fdfo_sp_post_process(
    forward_batch: ForwardBatch,
    full_logits: torch.Tensor,
    pick_tokens_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask_id: int,
    block_size: int,
) -> Tuple[List[List[int]], List[int]]:
    """Post-forward processing for Super Prefill (2-block) FDFO mode.

    Three scenarios based on mask_counts per block:
    -------------------------------------------------------------------------
      | mask_counts | action
    -------------------------------------------------------------------------
    0 |   [>0, >0]  | pick from block0 (both blocks still decoding), accept=0
    1 |   [0, >0]   | pick from block1 (block0 confirmed), accept=block_size
    2 |   [0, 0]    | both confirmed, accept=double_block_size
    -------------------------------------------------------------------------
    """
    torch.cuda.nvtx.range_push("fdfo_sp_post_process")

    batch_size = forward_batch.batch_size
    vocab_size = full_logits.shape[-1]
    double_block_size = block_size * 2
    expected_batch_size = forward_batch.input_ids.shape[0] // double_block_size
    if batch_size != expected_batch_size:
        raise RuntimeError(
            f"Batch size mismatch: forward_batch.batch_size={batch_size}, "
            f"but input_ids shape {forward_batch.input_ids.shape[0]} / "
            f"double_block_size {double_block_size} = {expected_batch_size}"
        )

    # Compute mask counts per (batch, block) BEFORE pick_tokens modifies input_ids
    mask_counts_tensor = (
        (forward_batch.input_ids == mask_id)
        .view(batch_size, 2, block_size)
        .sum(dim=2)
    )  # shape: [batch_size, 2]

    # For each batch: select the block to apply pick_tokens to
    # block0 has masks → pick from block0 (idx=0)
    # block0 is clean → pick from block1 (idx=1)
    block_select_idx = (mask_counts_tensor[:, 0] == 0).long()  # [batch_size]
    batch_indices = torch.arange(batch_size, device=full_logits.device)

    full_logits_reshaped = full_logits.view(batch_size, 2, block_size, vocab_size)
    input_ids_reshaped = forward_batch.input_ids.view(batch_size, 2, block_size)

    selected_logits = full_logits_reshaped[batch_indices, block_select_idx]  # [bs, block_size, vocab]
    selected_ids = input_ids_reshaped[batch_indices, block_select_idx]       # [bs, block_size]

    torch.cuda.nvtx.range_push("sp_pick_tokens")
    updated_ids = pick_tokens_func(selected_logits, selected_ids)
    torch.cuda.nvtx.range_pop()  # sp_pick_tokens

    # Write updated ids back
    original_ids = forward_batch.input_ids.view(batch_size, 2, block_size)
    original_ids[batch_indices, block_select_idx] = updated_ids
    forward_batch.input_ids = original_ids.view(-1)

    # Compute accept_length on GPU
    accept_lens = torch.where(
        mask_counts_tensor[:, 1] == 0,
        double_block_size,
        torch.where(
            mask_counts_tensor[:, 0] == 0,
            block_size,
            0,
        ),
    )  # [batch_size], values in {0, block_size, double_block_size}

    # Single D2H sync point
    next_token_ids = forward_batch.input_ids.view(batch_size, 2, block_size).tolist()
    mask_counts_cpu = mask_counts_tensor.tolist()
    accept_lens_cpu = accept_lens.tolist()

    next_token_ids_list: List[List[int]] = []
    accept_length_per_req_cpu: List[int] = []
    for i in range(batch_size):
        m1 = mask_counts_cpu[i][0]
        l1, l2 = next_token_ids[i][0], next_token_ids[i][1]
        # Return the flat [block0, block1] token list for the scheduler to consume
        next_token_ids_list.append(l1 + l2)
        accept_length_per_req_cpu.append(accept_lens_cpu[i])

    torch.cuda.nvtx.range_pop()  # fdfo_sp_post_process
    return next_token_ids_list, accept_length_per_req_cpu
