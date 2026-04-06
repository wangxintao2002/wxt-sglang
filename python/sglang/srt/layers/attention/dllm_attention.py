"""DLLM (Diffusion LLM) Super Prefill attention kernel.

Implements fused 2-block attention for DLLM inference:
  Q1 output = merge(Q1×K1,  Q1×Kc)
  Q2 output = merge(Q2×K2,  merge(Q2×K1, Q2×Kc))

Three FlashInfer wrappers are used:
  wrapper_paged         : [Q1,Q2] × Kc  (historical paged KV)
  wrapper_ragged_shared : [Q1,Q2] × K1  (block0 ragged KV)
  wrapper_ragged_local  : Q2     × K2  (block1 ragged KV)
"""

from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fill three indptr buffers in one launch
# ---------------------------------------------------------------------------

_INDPTR_BLOCK_SIZE = 256


@triton.jit
def _fill_three_indptr_kernel_impl(
    indptr1_ptr,
    indptr2_ptr,
    indptr3_ptr,
    step1: tl.constexpr,
    step2: tl.constexpr,
    step3: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill three indptr arrays with arithmetic sequences in one kernel launch."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(indptr1_ptr + offsets, (offsets * step1).to(tl.int32), mask=mask)
    tl.store(indptr2_ptr + offsets, (offsets * step2).to(tl.int32), mask=mask)
    tl.store(indptr3_ptr + offsets, (offsets * step3).to(tl.int32), mask=mask)


def _fill_three_indptr_fused(
    indptr1: torch.Tensor,
    indptr2: torch.Tensor,
    indptr3: torch.Tensor,
    step1: int,
    step2: int,
    step3: int,
    n: int,
):
    """Fill three indptr buffers via a single Triton kernel launch."""
    grid = ((n + _INDPTR_BLOCK_SIZE - 1) // _INDPTR_BLOCK_SIZE,)
    _fill_three_indptr_kernel_impl[grid](
        indptr1, indptr2, indptr3, step1, step2, step3, n, _INDPTR_BLOCK_SIZE
    )


# ---------------------------------------------------------------------------
# Lazy merge_state import
# ---------------------------------------------------------------------------

_merge_state_fn = None


def _get_merge_state():
    global _merge_state_fn
    if _merge_state_fn is None:
        from sglang.srt.layers.attention.merge_state import merge_state
        _merge_state_fn = merge_state
    return _merge_state_fn


# ---------------------------------------------------------------------------
# super_prefill_fused_attn_v2: forward pass (wrappers pre-initialised)
# ---------------------------------------------------------------------------

def super_prefill_fused_attn_v2(
    block_size: int,
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch,
    layer,
    prefill_wrapper_paged,
    prefill_wrapper_ragged,
    prefill_wrapper_ragged_local,
    logits_soft_cap,
):
    """Fused 2-block DLLM attention (V2: wrappers already begin_forward'd).

    Input shape: [batch_size * 2 * block_size, hidden_dim]
    Output shape: [batch_size * 2 * block_size, num_heads, head_dim]

    Decomposition:
      Q1_final = merge(Q1×K1, Q1×Kc)
      Q2_final = merge(Q2×K2, merge(Q2×K1, Q2×Kc))
    """
    merge_state = _get_merge_state()
    b = block_size

    q_r = q.view(-1, layer.tp_q_head_num, layer.head_dim)
    k_r = k.view(-1, layer.tp_k_head_num, layer.head_dim)
    v_r = v.view(-1, layer.tp_v_head_num, layer.head_dim)

    # Split into block0 / block1
    q_blocks = q_r.view(batch_size, 2, b, layer.tp_q_head_num, layer.head_dim)
    k_blocks = k_r.view(batch_size, 2, b, layer.tp_k_head_num, layer.head_dim)
    v_blocks = v_r.view(batch_size, 2, b, layer.tp_v_head_num, layer.head_dim)

    q2 = q_blocks[:, 1].reshape(-1, layer.tp_q_head_num, layer.head_dim)
    k1 = k_blocks[:, 0].reshape(-1, layer.tp_k_head_num, layer.head_dim)
    k2 = k_blocks[:, 1].reshape(-1, layer.tp_k_head_num, layer.head_dim)
    v1 = v_blocks[:, 0].reshape(-1, layer.tp_v_head_num, layer.head_dim)
    v2 = v_blocks[:, 1].reshape(-1, layer.tp_v_head_num, layer.head_dim)

    # --- Step 1: Paged attention [Q1,Q2] × Kc ---
    torch.cuda.nvtx.range_push("sp_paged_attn")
    kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    o_paged, s_paged = prefill_wrapper_paged.forward_return_lse(
        q_r,
        kv_buffer,
        causal=False,
        sm_scale=layer.scaling,
        logits_soft_cap=logits_soft_cap,
        k_scale=layer.k_scale_float,
        v_scale=layer.v_scale_float,
    )
    torch.cuda.nvtx.range_pop()

    o_paged_b = o_paged.view(batch_size, 2, b, layer.tp_q_head_num, layer.head_dim)
    s_paged_b = s_paged.view(batch_size, 2, b, layer.tp_q_head_num)
    o_paged_q1 = o_paged_b[:, 0].reshape(-1, layer.tp_q_head_num, layer.head_dim)
    o_paged_q2 = o_paged_b[:, 1].reshape(-1, layer.tp_q_head_num, layer.head_dim)
    s_paged_q1 = s_paged_b[:, 0].reshape(-1, layer.tp_q_head_num)
    s_paged_q2 = s_paged_b[:, 1].reshape(-1, layer.tp_q_head_num)

    # --- Step 2: Ragged shared [Q1,Q2] × K1 ---
    torch.cuda.nvtx.range_push("sp_shared_attn")
    o_shared, s_shared = prefill_wrapper_ragged.forward_return_lse(
        q_r,
        k1,
        v1,
        causal=False,
        sm_scale=layer.scaling,
        logits_soft_cap=logits_soft_cap,
    )
    torch.cuda.nvtx.range_pop()

    o_shared_b = o_shared.view(batch_size, 2, b, layer.tp_q_head_num, layer.head_dim)
    s_shared_b = s_shared.view(batch_size, 2, b, layer.tp_q_head_num)
    o_shared_q1 = o_shared_b[:, 0].reshape(-1, layer.tp_q_head_num, layer.head_dim)
    o_shared_q2 = o_shared_b[:, 1].reshape(-1, layer.tp_q_head_num, layer.head_dim)
    s_shared_q1 = s_shared_b[:, 0].reshape(-1, layer.tp_q_head_num)
    s_shared_q2 = s_shared_b[:, 1].reshape(-1, layer.tp_q_head_num)

    # --- Step 3: Ragged local Q2 × K2 ---
    torch.cuda.nvtx.range_push("sp_local_attn")
    o_local_q2, s_local_q2 = prefill_wrapper_ragged_local.forward_return_lse(
        q2,
        k2,
        v2,
        causal=False,
        sm_scale=layer.scaling,
        logits_soft_cap=logits_soft_cap,
    )
    torch.cuda.nvtx.range_pop()

    # --- Step 4: Merge ---
    torch.cuda.nvtx.range_push("sp_merge")
    o1_final, _ = merge_state(o_shared_q1, s_shared_q1, o_paged_q1, s_paged_q1)
    o_q2_hist, s_q2_hist = merge_state(o_shared_q2, s_shared_q2, o_paged_q2, s_paged_q2)
    o2_final, _ = merge_state(o_local_q2, s_local_q2, o_q2_hist, s_q2_hist)
    torch.cuda.nvtx.range_pop()

    # --- Step 5: Re-interleave output ---
    o1_b = o1_final.view(batch_size, b, layer.tp_q_head_num, layer.head_dim)
    o2_b = o2_final.view(batch_size, b, layer.tp_q_head_num, layer.head_dim)
    return torch.stack([o1_b, o2_b], dim=1).reshape(-1, layer.tp_q_head_num, layer.head_dim)


# ---------------------------------------------------------------------------
# call_dllm_begin_forward: initialise all three wrappers
# ---------------------------------------------------------------------------

def call_dllm_begin_forward(
    wrapper_ragged_shared,
    wrapper_ragged_local,
    wrapper_paged,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    paged_kernel_lens_sum: int,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    kv_start_idx: Optional[torch.Tensor],
    kv_indptr: torch.Tensor,
    qo_indptr: torch.Tensor,
    use_ragged: bool,
    spec_info,
    # DLLM-specific
    block_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    kv_last_page_len: torch.Tensor,
    req_to_token: torch.Tensor,
    # Pre-allocated CudaGraph buffers (optional)
    qo_indptr_shared_buf: Optional[torch.Tensor] = None,
    kv_indptr_shared_buf: Optional[torch.Tensor] = None,
    qo_indptr_local_buf: Optional[torch.Tensor] = None,
    fixed_split_size: Optional[int] = None,
):
    """Initialise all three attention wrappers for a DLLM SP decode step.

    Wrappers:
      wrapper_ragged_shared : [Q1,Q2] × K1  (qo_step=2b, kv_step=b)
      wrapper_ragged_local  : Q2 × K2       (qo_step=b,  kv_step=b)
      wrapper_paged         : [Q1,Q2] × Kc  (paged, qo_step=2b)
    """
    from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

    b = block_size
    bs = len(seq_lens)
    device = req_pool_indices.device

    # --- Paged KV indices (same as standard extend) ---
    torch.cuda.nvtx.range_push("sp_paged_kv_indices")
    if spec_info is None:
        kv_indptr[1: bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr_paged = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum + 256, dtype=torch.int32, device=device
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr_paged,
            kv_start_idx,
            kv_indices,
            req_to_token.shape[1],
        )
    else:
        kv_indices, kv_indptr_paged, _, _ = spec_info.generate_attn_arg_prefill(
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            req_to_token,
        )
    torch.cuda.nvtx.range_pop()

    # --- Ragged indptrs for shared and local ---
    # qo_indptr_shared: [0, 2b, 4b, ...]  (Q=2b per batch item)
    # kv_indptr_shared: [0, b,  2b, ...]  (K=b  per batch item)
    # qo_indptr_local:  [0, b,  2b, ...]  (Q=b  per batch item, reuse kv_indptr_shared)
    torch.cuda.nvtx.range_push("sp_ragged_indptrs")
    if qo_indptr_shared_buf is not None:
        qo_indptr_shared = qo_indptr_shared_buf[: bs + 1]
        kv_indptr_shared = kv_indptr_shared_buf[: bs + 1]
        qo_indptr_local  = qo_indptr_local_buf[: bs + 1]
        _fill_three_indptr_fused(
            qo_indptr_shared, kv_indptr_shared, qo_indptr_local,
            2 * b, b, b, bs + 1,
        )
    else:
        qo_indptr_shared = torch.arange(0, (bs + 1) * 2 * b, 2 * b, dtype=torch.int32, device=device)
        kv_indptr_shared = torch.arange(0, (bs + 1) * b,     b,     dtype=torch.int32, device=device)
        qo_indptr_local  = kv_indptr_shared  # same step
    kv_indptr_local = qo_indptr_local
    torch.cuda.nvtx.range_pop()

    # --- begin_forward: ragged shared ---
    torch.cuda.nvtx.range_push("sp_begin_shared")
    wrapper_ragged_shared.begin_forward(
        qo_indptr_shared,
        kv_indptr_shared,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=q_data_type,
    )
    torch.cuda.nvtx.range_pop()

    # --- begin_forward: ragged local ---
    torch.cuda.nvtx.range_push("sp_begin_local")
    wrapper_ragged_local.begin_forward(
        qo_indptr_local,
        kv_indptr_local,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=q_data_type,
    )
    torch.cuda.nvtx.range_pop()

    # --- begin_forward: paged (uses shared qo_indptr since Q length = 2b) ---
    torch.cuda.nvtx.range_push("sp_begin_paged")
    wrapper_paged.begin_forward(
        qo_indptr_shared,
        kv_indptr_paged,
        kv_indices,
        kv_last_page_len[:bs],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,  # page_size
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        custom_mask=None,
        non_blocking=True,
        fixed_split_size=fixed_split_size,
    )
    torch.cuda.nvtx.range_pop()
