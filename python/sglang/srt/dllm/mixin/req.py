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
        self.dllm_incomplete_ids = []
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_metric_decode_block_offset: Optional[int] = None
        self.dllm_metric_decode_block_start_ts: Optional[float] = None
        self.dllm_metric_decode_block_rounds = 0

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
        len_prefix = len(self.prefix_indices)
        block_size = self.dllm_config.block_size
        if not self.dllm_ids:
            padding = (-len(self.origin_input_ids)) % block_size
            self.dllm_ids = self.origin_input_ids + [self.dllm_config.mask_id] * padding
            self.fill_ids = self.dllm_ids[:block_size]
        elif self.dllm_incomplete_ids:
            self.fill_ids = self.fill_ids[:len_prefix] + self.dllm_incomplete_ids
        else:
            self.dllm_block_offset += block_size
            fill_len, dllm_len = len(self.fill_ids), len(self.dllm_ids)
            if fill_len < dllm_len:
                # prefill
                self.fill_ids += self.dllm_ids[fill_len : fill_len + block_size]
            else:
                # decode
                self.fill_ids += [self.dllm_config.mask_id] * block_size
