from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _block_causal_varlen_paged_fwd_kernel(
    Q,
    k_cache,
    v_cache,
    Out,
    req_to_token,
    req_pool_indices,
    seq_lens,
    extend_prefix_lens,
    extend_seq_lens,
    cu_seqlens_q,
    sm_scale,
    block_size,
    pool_stride,
    groups,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kc,
    stride_kh,
    stride_kd,
    stride_vc,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_tile = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    extend_len = tl.load(extend_seq_lens + seq_idx)
    prefix_len = tl.load(extend_prefix_lens + seq_idx)
    seq_len_kv = tl.load(seq_lens + seq_idx)
    q_global_off = tl.load(cu_seqlens_q + seq_idx)
    req_pool_idx = tl.load(req_pool_indices + seq_idx)

    q_local_start = q_tile * BLOCK_M
    if q_local_start >= extend_len:
        return

    kv_head_idx = head_idx // groups
    offs_m = q_local_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_abs = prefix_len + offs_m
    q_global = q_global_off + offs_m

    q = tl.load(
        Q + q_global[:, None] * stride_qt
        + head_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < extend_len,
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    max_q_blk = (prefix_len + q_local_start + BLOCK_M - 1) // block_size
    max_kv_pos = tl.minimum(seq_len_kv, (max_q_blk + 1) * block_size)
    end_n = tl.cdiv(max_kv_pos, BLOCK_N) * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)

    for n_start in range(0, end_n, BLOCK_N):
        kv_abs = n_start + offs_n
        kv_slots = tl.load(
            req_to_token + req_pool_idx * pool_stride + kv_abs,
            mask=kv_abs < seq_len_kv,
            other=0,
        )

        k = tl.load(
            k_cache
            + kv_slots[None, :] * stride_kc
            + kv_head_idx * stride_kh
            + offs_d[:, None] * stride_kd,
            mask=kv_abs[None, :] < seq_len_kv,
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale
        q_blk = q_abs[:, None] // block_size
        kv_blk = kv_abs[None, :] // block_size
        qk = tl.where(q_blk >= kv_blk, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        safe_diff = tl.where(m_new > float("-inf"), m_i - m_new, 0.0)
        alpha = tl.exp(safe_diff)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_cache
            + kv_slots[:, None] * stride_vc
            + kv_head_idx * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=kv_abs[:, None] < seq_len_kv,
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        Out + q_global[:, None] * stride_ot
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od,
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < extend_len,
    )


def block_causal_varlen_paged_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    groups = num_heads // num_kv_heads
    num_seqs = int(seq_lens.shape[0])

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    assert q.is_cuda

    block_d = triton.next_power_of_2(head_dim)
    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=q.device)
    if num_seqs > 0:
        cu_seqlens_q[1:] = extend_seq_lens.to(torch.int32).cumsum(0)

    max_extend_len = int(extend_seq_lens.max().item()) if num_seqs > 0 else 0
    if out is None:
        out = torch.empty_like(q)
    if max_extend_len == 0:
        return out

    grid = (triton.cdiv(max_extend_len, BLOCK_M), num_heads, num_seqs)
    _block_causal_varlen_paged_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_prefix_lens,
        extend_seq_lens,
        cu_seqlens_q,
        sm_scale=scale,
        block_size=block_size,
        pool_stride=req_to_token.stride(0),
        groups=groups,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kc=k_cache.stride(0),
        stride_kh=k_cache.stride(1),
        stride_kd=k_cache.stride(2),
        stride_vc=v_cache.stride(0),
        stride_vh=v_cache.stride(1),
        stride_vd=v_cache.stride(2),
        stride_ot=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
