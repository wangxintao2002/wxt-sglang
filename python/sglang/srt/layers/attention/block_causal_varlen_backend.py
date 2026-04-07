from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.block_causal_varlen_paged import (
    block_causal_varlen_paged_forward,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_ENV_BLOCK_M = int(os.environ.get("SGLANG_VARLEN_BLOCK_M", "0")) or None
_ENV_BLOCK_N = int(os.environ.get("SGLANG_VARLEN_BLOCK_N", "0")) or None


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    fp8_types = tuple(
        t
        for t in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if t is not None
    )
    return dtype in fp8_types


def _pick_tile_config(max_extend_len: int, head_dim: int):
    if _ENV_BLOCK_M and _ENV_BLOCK_N:
        return _ENV_BLOCK_M, _ENV_BLOCK_N, 4
    if head_dim > 128:
        return 32, 32, 4
    if head_dim > 64:
        return 64, 32, 4
    return 64, 64, 4


class BlockCausalVarlenBackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner", block_size: int = 4):
        super().__init__()
        self.block_size = block_size
        self.device = model_runner.device
        self.num_heads = model_runner.model_config.num_attention_heads
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        logger.info(
            "BlockCausalVarlenBackend: block_size=%d, num_heads=%d, num_kv_heads=%d, head_dim=%d",
            block_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        pass

    def forward_extend(
        self, q, k, v, layer, forward_batch: ForwardBatch, save_kv_cache: bool = True
    ):
        from sglang.srt.layers.radix_attention import AttentionType

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.attn_type == AttentionType.ENCODER_ONLY:
            save_kv_cache = False

        if save_kv_cache:
            k_scale = getattr(layer, "k_scale", None)
            v_scale = getattr(layer, "v_scale", None)
            try:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v, k_scale, v_scale
                )
            except TypeError:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_forward_extend(
            query=q_,
            output=o_,
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_seq_lens=forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            kv_k_scale=getattr(layer, "k_scale_float", None),
            kv_v_scale=getattr(layer, "v_scale_float", None),
        )
        return o

    def _run_forward_extend(
        self,
        query,
        output,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_prefix_lens,
        extend_seq_lens,
        scaling: Optional[float] = None,
        kv_k_scale=None,
        kv_v_scale=None,
    ):
        num_seqs = int(seq_lens.shape[0])
        if num_seqs == 0:
            return output

        compute_dtype = query.dtype

        kv_k_scale = _coerce_scale(kv_k_scale)
        kv_v_scale = _coerce_scale(kv_v_scale)
        k_cache_is_fp8 = _is_fp8_dtype(k_cache.dtype)
        v_cache_is_fp8 = _is_fp8_dtype(v_cache.dtype)
        if k_cache_is_fp8 or v_cache_is_fp8:
            k_cache, v_cache, req_to_token, req_pool_indices = _dequant_fp8_caches(
                k_cache,
                v_cache,
                k_cache_is_fp8,
                v_cache_is_fp8,
                kv_k_scale,
                kv_v_scale,
                req_to_token,
                req_pool_indices,
                seq_lens,
                num_seqs,
                compute_dtype,
            )

        max_extend_len = int(extend_seq_lens.max().item())
        block_m, block_n, num_warps = _pick_tile_config(
            max_extend_len, query.shape[-1]
        )
        block_causal_varlen_paged_forward(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            block_size=self.block_size,
            scale=scaling,
            out=output,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )
        return output

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        raise NotImplementedError(
            "BlockCausalVarlenBackend does not support decode. "
            "Use --prefill-attention-backend block_causal_varlen_attention "
            "--decode-attention-backend flashinfer."
        )

    def support_triton(self) -> bool:
        return True


def _coerce_scale(scale) -> float:
    if scale is None:
        return 1.0
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _dequant_fp8_caches(
    k_cache,
    v_cache,
    k_is_fp8: bool,
    v_is_fp8: bool,
    k_scale: float,
    v_scale: float,
    req_to_token,
    req_pool_indices,
    seq_lens,
    num_seqs: int,
    compute_dtype: torch.dtype,
):
    device = req_to_token.device
    slot_lists = []
    for seq_idx in range(num_seqs):
        req_pool_idx = int(req_pool_indices[seq_idx].item())
        seq_len = int(seq_lens[seq_idx].item())
        slot_lists.append(req_to_token[req_pool_idx, :seq_len].to(torch.int64))

    if slot_lists:
        all_slots = torch.cat(slot_lists, dim=0)
        unique_slots, inverse = torch.unique(all_slots, sorted=True, return_inverse=True)
    else:
        unique_slots = torch.empty((0,), dtype=torch.int64, device=device)
        inverse = torch.empty((0,), dtype=torch.int64, device=device)

    new_k_cache = k_cache.index_select(0, unique_slots)
    new_v_cache = v_cache.index_select(0, unique_slots)
    if k_is_fp8:
        new_k_cache = new_k_cache.to(compute_dtype) * k_scale
    if v_is_fp8:
        new_v_cache = new_v_cache.to(compute_dtype) * v_scale

    max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
    remapped_req_to_token = torch.zeros(
        (req_to_token.shape[0], max_seq_len),
        dtype=torch.int32,
        device=device,
    )
    cursor = 0
    for seq_idx in range(num_seqs):
        req_pool_idx = int(req_pool_indices[seq_idx].item())
        seq_len = int(seq_lens[seq_idx].item())
        remapped_req_to_token[req_pool_idx, :seq_len] = inverse[
            cursor : cursor + seq_len
        ].to(torch.int32)
        cursor += seq_len

    return new_k_cache, new_v_cache, remapped_req_to_token, req_pool_indices
