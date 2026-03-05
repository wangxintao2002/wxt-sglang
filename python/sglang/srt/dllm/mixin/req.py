from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_incomplete_ids = []

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_config.enable_fdfo_mode:
            self._init_fill_ids_for_dllm_fdfo()
        else:
            self._init_fill_ids_for_dllm_basic()

    def _init_fill_ids_for_dllm_basic(self: Req):
        block_size = self.dllm_config.get_block_size()
        if not self.dllm_ids:
            self.dllm_ids = (
                self.origin_input_ids
                + [self.dllm_config.mask_id] * block_size
            )
        else:
            self.dllm_block_offset += block_size
            self.dllm_ids += [self.dllm_config.mask_id] * block_size
        self.fill_ids = list(self.dllm_ids)
      
    def _init_fill_ids_for_dllm_fdfo(self: Req):
        block_size = self.dllm_config.get_block_size()
        mask_id = self.dllm_config.mask_id
        len_prefix = len(self.prefix_indices)
        if not self.dllm_ids:
            padding = (-len(self.origin_input_ids)) % block_size
            self.dllm_ids = self.origin_input_ids + [mask_id] * padding
            self.fill_ids = self.dllm_ids[:block_size]
        elif (
            self.dllm_incomplete_ids
        ): # revert chrrent fill_ids and refill them by dllm_incomplete_ids
            assert (
                len(self.dllm_incomplete_ids) == block_size
                and len(self.fill_ids) == len_prefix + block_size
            ), f"len(self.dllm_incomplete_ids) error"
            self.fill_ids = self.fill_ids[:len_prefix] + self.dllm_incomplete_ids
        else:
            self.dllm_block_offset += block_size
            fill_len, dllm_len = len(self.fill_ids), len(self.dllm_ids)
            if fill_len < dllm_len:
                # prefill
                self.fill_ids += self.dllm_ids[fill_len : fill_len + block_size]
            else:
                # decode
                self.fill_ids += [mask_id] * block_size
