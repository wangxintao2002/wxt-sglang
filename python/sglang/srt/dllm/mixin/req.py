from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    SKIP_FORWARD = "skip_forward"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_incomplete_ids = []
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_metric_decode_block_offset: Optional[int] = None
        self.dllm_metric_decode_block_start_ts: Optional[float] = None
        self.dllm_metric_decode_block_rounds = 0
        self.dllm_kv_reusable = False
        self.dllm_tree_lock_held = False

        if self.dllm_config is not None:
            block_size = self.dllm_config.get_block_size()
            self.dllm_origin_len_aligned = self.origin_input_ids[
                : (len(self.origin_input_ids) // block_size * block_size)
            ]
            if len(self.dllm_origin_len_aligned) == 0:
                self.dllm_phase = DllmReqPhase.STAGING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase == DllmReqPhase.INCOMING_PREFILL

    def determine_dllm_phase(self: Req):
        return

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_phase == DllmReqPhase.SKIP_FORWARD:
            return
        if self.dllm_config.enable_super_prefill:
            self._init_fill_ids_for_dllm_fdfo_sp()
            return
        if self.dllm_config.enable_fdfo:
            self._init_fill_ids_for_dllm_fdfo()
            return
        self._init_fill_ids_for_dllm_basic()

    def _init_fill_ids_for_dllm_basic(self: Req):
        block_size = self.dllm_config.get_block_size()
        first_call = not self.dllm_ids
        if first_call:
            self.dllm_ids = self.origin_input_ids + [self.dllm_config.mask_id] * (
                -len(self.origin_input_ids) % block_size
            )

        if self.dllm_phase == DllmReqPhase.INCOMING_PREFILL:
            self.fill_ids = list(self.dllm_origin_len_aligned)
        elif first_call and len(self.dllm_origin_len_aligned) == 0:
            self.dllm_block_offset = 0
            self.fill_ids = list(self.dllm_ids)
        elif len(self.fill_ids) < len(self.dllm_ids):
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids = list(self.dllm_ids)
        else:
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids += [self.dllm_config.mask_id] * block_size

    def _init_fill_ids_for_dllm_fdfo(self: Req):
        block_size = self.dllm_config.block_size
        len_prefix = len(self.prefix_indices)
        first_call = not self.dllm_ids
        if first_call:
            self.dllm_ids = self.origin_input_ids + [self.dllm_config.mask_id] * (
                -len(self.origin_input_ids) % block_size
            )

        if self.dllm_phase == DllmReqPhase.INCOMING_PREFILL:
            self.fill_ids = list(self.dllm_origin_len_aligned)
        elif first_call and len(self.dllm_origin_len_aligned) == 0:
            self.dllm_block_offset = 0
            self.fill_ids = list(self.dllm_ids)
        elif len(self.fill_ids) < len(self.dllm_ids):
            # AR-prefill is intended to cover the aligned prompt in one round,
            # but under scheduler edge cases we can still arrive here with a
            # shorter prompt prefix. Recover by jumping to the padded prompt.
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids = list(self.dllm_ids)
        elif self.dllm_incomplete_ids:
            assert len(self.dllm_incomplete_ids) == block_size
            self.fill_ids = self.fill_ids[:len_prefix] + self.dllm_incomplete_ids
        else:
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids += [self.dllm_config.mask_id] * block_size

    def _init_fill_ids_for_dllm_fdfo_sp(self: Req):
        block_size = self.dllm_config.block_size
        double_block_size = self.dllm_config.double_block_size
        mask_id = self.dllm_config.mask_id
        one_block_of_masks = [mask_id] * block_size

        first_call = not self.dllm_ids
        if first_call:
            self.dllm_ids = (
                self.origin_input_ids
                + [mask_id] * (-len(self.origin_input_ids) % double_block_size)
            )

        if self.dllm_phase == DllmReqPhase.INCOMING_PREFILL:
            self.fill_ids = list(self.dllm_origin_len_aligned)
        elif first_call and len(self.dllm_origin_len_aligned) == 0:
            self.dllm_block_offset = 0
            self.fill_ids = list(self.dllm_ids)
        elif len(self.fill_ids) < len(self.dllm_ids):
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids = list(self.dllm_ids)
        elif len(self.dllm_incomplete_ids) == double_block_size:
            self.fill_ids = self.fill_ids[:-double_block_size] + self.dllm_incomplete_ids
        elif len(self.dllm_incomplete_ids) == block_size:
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids = (
                self.fill_ids[:-block_size]
                + self.dllm_incomplete_ids
                + one_block_of_masks
            )
        else:
            self.dllm_block_offset = len(self.fill_ids)
            self.fill_ids = self.fill_ids + one_block_of_masks + one_block_of_masks
