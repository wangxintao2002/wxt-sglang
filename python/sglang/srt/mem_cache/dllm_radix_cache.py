from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class DllmRadixCache(BasePrefixCache):
    def __init__(self, params: "CacheInitParams", max_cache_tokens: int = 2048):
        self._radix_cache = RadixCache(params)
        self.req_to_token_pool = self._radix_cache.req_to_token_pool
        self.token_to_kv_pool_allocator = self._radix_cache.token_to_kv_pool_allocator
        self.page_size = self._radix_cache.page_size
        self.device = self._radix_cache.device
        self.disable = self._radix_cache.disable
        self.max_cache_tokens = max_cache_tokens

    def __getattr__(self, name: str):
        return getattr(self._radix_cache, name)

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        req = getattr(params, "req", None)
        if req is not None and req.is_dllm() and not req.is_dllm_prefill():
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=self._radix_cache.root_node,
                last_host_node=self._radix_cache.root_node,
            )
        return self._radix_cache.match_prefix(params)

    def insert(self, params: InsertParams) -> InsertResult:
        return self._radix_cache.insert(params)

    def cache_finished_req(self, req: "Req", is_insert: bool = True):
        assert req.is_dllm()
        if req.last_node is None or not getattr(req, "dllm_tree_lock_held", False):
            token_ids = req.origin_input_ids + req.output_ids
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            keys = self._radix_cache._page_align_keys(token_ids)
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )
            self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])
            req.dllm_tree_lock_held = False
            return
        self._radix_cache.cache_finished_req(req, is_insert=False)
        req.dllm_tree_lock_held = False

    def cache_unfinished_req(self, req: "Req", chunked: bool = False):
        if not req.is_dllm() or req.is_dllm_prefill():
            fill_len = len(req.fill_ids)
            if self.max_cache_tokens > 0 and fill_len > self.max_cache_tokens:
                aligned_len = fill_len // self.page_size * self.page_size
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :aligned_len
                ]
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            else:
                self._radix_cache.cache_unfinished_req(req, chunked=chunked)
                if req.is_dllm():
                    req.dllm_tree_lock_held = True
        else:
            fill_len = len(req.fill_ids)
            aligned_len = fill_len // self.page_size * self.page_size
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :aligned_len
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, params: EvictParams) -> EvictResult:
        return self._radix_cache.evict(params)

    def inc_lock_ref(self, node: Any):
        return self._radix_cache.inc_lock_ref(node)

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return self._radix_cache.dec_lock_ref(node)

    def reset(self):
        self._radix_cache.reset()

    def evictable_size(self):
        return self._radix_cache.evictable_size()

    def protected_size(self):
        return self._radix_cache.protected_size()

    def total_size(self):
        return self._radix_cache.total_size()

    def pretty_print(self):
        return self._radix_cache.pretty_print()

    def take_events(self):
        return self._radix_cache.take_events()
