from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda and not _is_hip:
    import triton
    import triton.language as tl


    @triton.jit
    def _fdfo_masked_top1_logconf_kernel(
        logits_ptr,
        top1_ids_ptr,
        log_conf_ptr,
        num_rows,
        vocab_size,
        stride_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        if pid < num_rows:
            row_ptr = logits_ptr + pid * stride_row
            offsets = tl.arange(0, BLOCK_SIZE)

            best_val = float("-inf")
            best_idx = 0
            running_max = float("-inf")
            running_sum = 0.0

            num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
            for block_idx in range(0, num_blocks):
                cols = block_idx * BLOCK_SIZE + offsets
                mask = cols < vocab_size
                vals = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(
                    tl.float32
                )

                local_max = tl.max(vals, axis=0)
                local_idx = tl.argmax(vals, axis=0) + block_idx * BLOCK_SIZE

                new_running_max = tl.maximum(running_max, local_max)
                running_sum = running_sum * tl.exp(running_max - new_running_max) + tl.sum(
                    tl.exp(vals - new_running_max), axis=0
                )
                running_max = new_running_max

                should_update = (local_max > best_val) | (
                    (local_max == best_val) & (local_idx < best_idx)
                )
                best_val = tl.where(should_update, local_max, best_val)
                best_idx = tl.where(should_update, local_idx, best_idx)

            log_conf = best_val - (running_max + tl.log(running_sum))
            tl.store(top1_ids_ptr + pid, best_idx)
            tl.store(log_conf_ptr + pid, log_conf)


def _pick_kernel_config(vocab_size: int) -> tuple[int, int]:
    if vocab_size <= 1024:
        return 1024, 4
    if vocab_size <= 2048:
        return 2048, 8
    return 4096, 8


def fdfo_masked_top1_logconf(masked_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        not _is_cuda
        or _is_hip
        or not masked_logits.is_cuda
        or masked_logits.ndim != 2
    ):
        raise RuntimeError("fdfo_masked_top1_logconf requires a CUDA 2D tensor")

    masked_logits = masked_logits.contiguous()
    num_rows, vocab_size = masked_logits.shape
    top1_ids = torch.empty((num_rows,), dtype=torch.int32, device=masked_logits.device)
    log_conf = torch.empty((num_rows,), dtype=torch.float32, device=masked_logits.device)

    if num_rows == 0:
        return top1_ids, log_conf

    block_size, num_warps = _pick_kernel_config(vocab_size)
    _fdfo_masked_top1_logconf_kernel[(num_rows,)](
        masked_logits,
        top1_ids,
        log_conf,
        num_rows,
        vocab_size,
        masked_logits.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return top1_ids, log_conf
