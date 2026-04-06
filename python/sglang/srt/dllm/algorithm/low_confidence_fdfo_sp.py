import time
from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.fdfo_utils import fdfo_sp_post_process
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.instrumentation import record_dllm_event
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceFDFOSuperPrefill(DllmAlgorithm):
    """Low Confidence FDFO with Super Prefill: 2-block decode per round."""

    requires_fdfo_mode: bool = True
    enable_super_prefill: bool = True

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def _pick_tokens(
        self, full_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Pick tokens for a single selected block (shape [batch, block_size, vocab])."""
        batch_size = input_ids.shape[0]
        block_mask_index = input_ids == self.mask_id

        x = torch.argmax(full_logits, dim=-1)  # [batch, block_size]
        probs = torch.nn.functional.softmax(full_logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

        # Mask non-mask positions so they can't be selected
        confidence = torch.where(
            block_mask_index,
            confidence,
            torch.tensor(float("-inf"), device=confidence.device, dtype=confidence.dtype),
        )

        transfer_index = confidence > self.threshold

        # Fallback: if no position exceeds threshold, force top-1 mask position
        has_transfer = transfer_index.sum(dim=1) > 0  # [batch_size]
        _, top1_indices = torch.topk(confidence, k=1, dim=1)  # [batch_size, 1]
        top1_indices = top1_indices.squeeze(-1)
        batch_indices = torch.arange(batch_size, device=top1_indices.device)
        top1_mask = torch.zeros_like(transfer_index, dtype=torch.bool)
        top1_mask[batch_indices, top1_indices] = True

        transfer_index = torch.where(
            has_transfer.unsqueeze(-1), transfer_index, top1_mask
        )

        # Apply: only update mask positions that are selected
        x = torch.where(block_mask_index, x, input_ids)
        return torch.where(transfer_index, x, input_ids)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[List[int]], List[int], bool]:
        torch.cuda.nvtx.range_push("dllm::FDFO_SP::run")
        run_t0 = time.perf_counter()

        torch.cuda.nvtx.range_push("dllm::FDFO_SP::model_forward")
        t0 = time.perf_counter()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        model_forward_ms = (time.perf_counter() - t0) * 1000
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        torch.cuda.nvtx.range_pop()  # model_forward

        torch.cuda.nvtx.range_push("dllm::FDFO_SP::post_process")
        t0 = time.perf_counter()
        next_token_ids_list, accept_length_per_req_cpu = fdfo_sp_post_process(
            forward_batch=forward_batch,
            full_logits=logits_output.full_logits,
            pick_tokens_func=self._pick_tokens,
            mask_id=self.mask_id,
            block_size=self.block_size,
        )
        postprocess_ms = (time.perf_counter() - t0) * 1000
        torch.cuda.nvtx.range_pop()  # post_process

        record_dllm_event(
            "dllm_algo_run",
            algo="LowConfidenceFDFOSuperPrefill",
            batch_size=forward_batch.batch_size,
            run_ms=(time.perf_counter() - run_t0) * 1000,
            model_forward_ms=model_forward_ms,
            postprocess_ms=postprocess_ms,
        )

        torch.cuda.nvtx.range_pop()  # run
        return (
            logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            can_run_cuda_graph,
        )


Algorithm = LowConfidenceFDFOSuperPrefill
