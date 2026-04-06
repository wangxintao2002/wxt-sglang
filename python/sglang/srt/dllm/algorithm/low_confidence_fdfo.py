import math
import os
from typing import List, Tuple, Union

import torch
import torch.cuda.nvtx as nvtx

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.instrumentation import record_dllm_event
from sglang.srt.dllm.triton_ops import fdfo_masked_top1_logconf
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_hip

_is_hip = is_hip()
_disable_triton_pick = os.getenv("SGLANG_DLLM_FDFO_DISABLE_TRITON", "0") == "1"


class LowConfidenceFDFO(DllmAlgorithm):
    """Low Confidence algorithm for DLLM. Requiring first done first out mode"""

    requires_fdfo_mode: bool = True

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.log_threshold = math.log(self.threshold)

    def _pick_tokens(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> float:
        """PyTorch implementation of pick_tokens (fallback)."""
        import time

        t0 = time.perf_counter()
        nvtx.range_push("dllm::FDFO::pick_tokens")
        batch_size = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        full_logits = full_logits.view(batch_size, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        block_mask_index = input_ids == self.mask_id
        if not torch.any(block_mask_index):
            nvtx.range_pop()  # pick_tokens
            return (time.perf_counter() - t0) * 1000

        flat_mask_index = block_mask_index.view(-1)
        masked_flat_indices = torch.nonzero(flat_mask_index, as_tuple=False).squeeze(-1)
        masked_logits = full_logits.view(-1, vocab_size).index_select(
            0, masked_flat_indices
        )

        if masked_logits.is_cuda and not _is_hip and not _disable_triton_pick:
            top1_token_ids, log_confidence = fdfo_masked_top1_logconf(masked_logits)
        else:
            top1_logits, top1_token_ids = masked_logits.max(dim=-1)
            log_confidence = top1_logits - torch.logsumexp(masked_logits, dim=-1)

        transfer_index = torch.zeros_like(block_mask_index)
        transfer_index.view(-1)[masked_flat_indices] = (
            log_confidence > self.log_threshold
        )
        has_transfer = transfer_index.any(dim=1)

        if not torch.all(has_transfer):
            confidence = torch.full(
                (batch_size, self.block_size),
                float("-inf"),
                dtype=log_confidence.dtype,
                device=log_confidence.device,
            )
            confidence.view(-1)[masked_flat_indices] = log_confidence
            fallback_blocks = torch.nonzero(~has_transfer, as_tuple=False).squeeze(-1)
            top1_indices = confidence.argmax(dim=1)
            transfer_index[fallback_blocks, top1_indices[fallback_blocks]] = True

        candidate_ids = input_ids.clone()
        candidate_ids.view(-1)[masked_flat_indices] = top1_token_ids.to(
            candidate_ids.dtype
        )
        input_ids = torch.where(transfer_index, candidate_ids, input_ids)

        forward_batch.input_ids = input_ids.view(-1)
        nvtx.range_pop()  # pick_tokens
        return (time.perf_counter() - t0) * 1000

    def _post_forward_process(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> List[List[int]]:
        import time

        t_total = time.perf_counter()
        nvtx.range_push("dllm::FDFO::post_forward_process")
        # Validate batch size
        batch_size = forward_batch.batch_size
        expected_batch_size = forward_batch.input_ids.shape[0] // self.block_size
        if batch_size != expected_batch_size:
            raise RuntimeError(
                f"Batch size mismatch: forward_batch.batch_size={batch_size}, "
                f"but input_ids shape {forward_batch.input_ids.shape[0]} / "
                f"block_size {self.block_size} = {expected_batch_size}"
            )

        # Compute mask counts per block BEFORE _pick_tokens modifies input_ids
        # Convert mask counts to CPU once to avoid multiple D2H transfers
        mask_counts_cpu = (
            (forward_batch.input_ids == self.mask_id)
            .view(batch_size, self.block_size)
            .sum(dim=1)
            .tolist()
        )

        # Update input_ids based on confidence
        pick_tokens_ms = self._pick_tokens(forward_batch, full_logits)

        # Build output token IDs list with decode markers
        nvtx.range_push("dllm::FDFO::build_output")
        t_build = time.perf_counter()
        # Reshape and convert to CPU list in one operation
        next_token_ids = forward_batch.input_ids.view(
            batch_size, self.block_size
        ).tolist()

        next_token_ids_list = []
        accept_length_per_req_cpu = []
        for i in range(batch_size):
            next_token_ids_list.append(next_token_ids[i])
            accept_length_per_req_cpu.append(
                self.block_size if mask_counts_cpu[i] == 0 else 0
            )
        build_output_ms = (time.perf_counter() - t_build) * 1000
        nvtx.range_pop()  # build_output

        nvtx.range_pop()  # post_forward_process
        return (
            next_token_ids_list,
            accept_length_per_req_cpu,
            {
                "pick_tokens_ms": pick_tokens_ms,
                "build_output_ms": build_output_ms,
                "postprocess_ms": (time.perf_counter() - t_total) * 1000,
                "accept_zero_blocks": sum(1 for x in mask_counts_cpu if x != 0),
                "accept_full_blocks": sum(1 for x in mask_counts_cpu if x == 0),
            },
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], List[List[int]], List[int], bool
    ]:
        import time

        nvtx.range_push("dllm::FDFO::run")
        run_t0 = time.perf_counter()

        # Forward pass through the model (no preprocessing needed)
        # each block may contain mask or finished tokens
        nvtx.range_push("dllm::FDFO::model_forward")
        t0 = time.perf_counter()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        model_forward_ms = (time.perf_counter() - t0) * 1000
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        nvtx.range_pop()  # model_forward

        # Post-forward processing: pick tokens and build output list
        next_token_ids_list, accept_length_per_req_cpu, stats = (
            self._post_forward_process(forward_batch, logits_output.full_logits)
        )
        record_dllm_event(
            "dllm_algo_run",
            algo="LowConfidenceFDFO",
            batch_size=forward_batch.batch_size,
            run_ms=(time.perf_counter() - run_t0) * 1000,
            model_forward_ms=model_forward_ms,
            postprocess_ms=stats["postprocess_ms"],
            pick_tokens_ms=stats["pick_tokens_ms"],
            build_output_ms=stats["build_output_ms"],
            accept_zero_blocks=stats["accept_zero_blocks"],
            accept_full_blocks=stats["accept_full_blocks"],
        )

        nvtx.range_pop()  # run
        return (
            logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            can_run_cuda_graph,
        )


Algorithm = LowConfidenceFDFO
