from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, RequestStage, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        import torch.cuda.nvtx as nvtx

        nvtx.range_push("dllm::scheduler::get_new_batch_dllm")
        if self.try_preemption:
            self.running_batch.batch_is_full = False

        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        if self._should_skip_scheduling():
            nvtx.range_pop()
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)
        adder = self._create_dllm_prefill_adder(running_bs)
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            nvtx.range_pop()
            return None

        self._update_metrics_and_state_for_batch(can_run_list, adder, running_bs)
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        nvtx.range_pop()
        return new_batch

    def _fetch_waiting_reqs(self: Scheduler):
        free_slots = self.dllm_manager.num_free_slots()
        min_batch = self.dllm_manager.prefill_min_batch_size
        if free_slots < min_batch:
            return

        num_requests_to_add = min(min_batch, free_slots, len(self.waiting_queue))
        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_scheduling(self: Scheduler) -> bool:
        if self.dllm_manager.is_empty() and not self.waiting_queue:
            return True

        if self.dllm_manager.is_empty():
            return True

        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.try_preemption
        ):
            self.running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        decode_reqs = self.dllm_manager.get_decode_requests()

        if prefill_reqs:
            self.process_dllm_incoming_reqs(adder, prefill_reqs)
            if adder.can_run_list:
                return ForwardMode.EXTEND
            logger.debug(
                "DLLM prefill deferred: %s prefill reqs, %s decode reqs",
                len(prefill_reqs),
                len(decode_reqs),
            )

        if decode_reqs:
            self.process_dllm_staging_reqs(adder, decode_reqs)
        return ForwardMode.DLLM_EXTEND

    def _update_metrics_and_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        if self.enable_metrics:
            for req in can_run_list:
                req.add_latency(RequestStage.PREFILL_WAITING)

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

        for req in can_run_list:
            if req.time_stats.forward_entry_time == 0:
                req.time_stats.forward_entry_time = time.perf_counter()
                if self.enable_metrics:
                    self.metrics_collector.observe_queue_time(
                        req.time_stats.get_queueing_time(),
                    )

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        is_ar_prefill = forward_mode == ForwardMode.EXTEND
        decode_round_counts = []
        now = time.perf_counter()
        for req in can_run_list:
            if req.dllm_phase == DllmReqPhase.STAGING_DECODE:
                if req.dllm_metric_decode_block_offset != req.dllm_block_offset:
                    req.dllm_metric_decode_block_offset = req.dllm_block_offset
                    req.dllm_metric_decode_block_start_ts = now
                    req.dllm_metric_decode_block_rounds = 0
                req.dllm_metric_decode_block_rounds += 1
                decode_round_counts.append(req.dllm_metric_decode_block_rounds)

        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=None if is_ar_prefill else self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None
        new_batch.dllm_ar_prefill = is_ar_prefill

        new_batch.dllm_metric_max_decode_rounds = (
            max(decode_round_counts) if decode_round_counts else 0
        )
        new_batch.dllm_metric_num_decode_reqs = len(decode_round_counts)
        new_batch.dllm_metric_num_prefill_reqs = len(can_run_list) - len(
            decode_round_counts
        )

        from sglang.srt.managers.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats(
            log_input_tokens=self.adder.log_input_tokens,
            log_hit_tokens=self.adder.log_hit_tokens,
            new_token_ratio=self.adder.new_token_ratio,
            running_bs=len(self.running_batch.reqs),
            num_new_seqs=len(can_run_list),
        )
        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        res = AddReqResult.CONTINUE
        for req in reqs:
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if not self.try_preemption or not adder.preempt_to_schedule(
                    req, self.server_args
                ):
                    break

            req.init_next_round_input(self.tree_cache)
            if req.extend_input_len == 0:
                # This request fully hits the cached prefill prefix, so it skips
                # the AR-prefill forward. We still need to protect the matched
                # radix prefix for the lifetime of the request.
                adder._req_inc_lock_ref(req)
                req.dllm_tree_lock_held = True
                req.dllm_phase = DllmReqPhase.STAGING_DECODE
                self.dllm_manager.add_staging_reqs([req])
                continue

            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )
            if res == AddReqResult.CONTINUE:
                req.dllm_tree_lock_held = True
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break
        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        for req in reqs:
            if not req.dllm_ids:
                req.init_next_round_input()
            block_size = self.dllm_config.get_block_size()
            if len(req.fill_ids) - len(req.prefix_indices) != block_size:
                target_prefix_len = max(0, len(req.fill_ids) - block_size)
                req.prefix_indices = req.prefix_indices[:target_prefix_len]
                req.set_extend_input_len(block_size)
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res
        return AddReqResult.CONTINUE


class DllmManager:
    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        if dllm_config is not None and dllm_config.prefill_ratio > 0.0:
            self.prefill_min_batch_size = max(
                1, math.ceil(self.max_running_reqs * dllm_config.prefill_ratio)
            )
        else:
            self.prefill_min_batch_size = 1
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        assert self.dllm_config is not None, "Diffusion LLM config is not set."
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")
        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def num_free_slots(self) -> int:
        return self.max_running_reqs - len(self.waiting_queue)

    def increment_chunked_count(self) -> None:
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
