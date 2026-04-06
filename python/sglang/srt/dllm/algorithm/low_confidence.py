from typing import List, Tuple, Union

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.instrumentation import record_dllm_event
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        import time

        nvtx.range_push("dllm::LowConfidence::run")
        run_t0 = time.perf_counter()
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        model_forward_ms = 0.0
        postprocess_ms = 0.0
        block_iterations = [0] * batch_size
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            nvtx.range_push("dllm::LowConfidence::fast_path")
            t0 = time.perf_counter()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            model_forward_ms += (time.perf_counter() - t0) * 1000
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            nvtx.range_pop()

            next_token_ids = []
            record_dllm_event(
                "dllm_algo_run",
                algo="LowConfidence",
                batch_size=batch_size,
                run_ms=(time.perf_counter() - run_t0) * 1000,
                model_forward_ms=model_forward_ms,
                postprocess_ms=postprocess_ms,
                block_iterations=block_iterations,
                max_block_iterations=0,
                num_completed_decode_blocks=0,
            )
            nvtx.range_pop()  # run
            return logits_output, next_token_ids, None, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for iter_i in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            nvtx.range_push(f"dllm::LowConfidence::denoise_iter_{iter_i}")

            nvtx.range_push("dllm::LowConfidence::model_forward")
            t0 = time.perf_counter()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            model_forward_ms += (time.perf_counter() - t0) * 1000
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            nvtx.range_pop()  # model_forward

            nvtx.range_push("dllm::LowConfidence::confidence_and_transfer")
            t0 = time.perf_counter()
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                block_iterations[batch_id] += 1
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]
            postprocess_ms += (time.perf_counter() - t0) * 1000
            nvtx.range_pop()  # confidence_and_transfer

            nvtx.range_pop()  # denoise_iter

        nvtx.range_push("dllm::LowConfidence::final_forward")
        t0 = time.perf_counter()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        model_forward_ms += (time.perf_counter() - t0) * 1000
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        nvtx.range_pop()  # final_forward

        nvtx.range_push("dllm::LowConfidence::reshape_output")
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        nvtx.range_pop()  # reshape_output
        record_dllm_event(
            "dllm_algo_run",
            algo="LowConfidence",
            batch_size=batch_size,
            run_ms=(time.perf_counter() - run_t0) * 1000,
            model_forward_ms=model_forward_ms,
            postprocess_ms=postprocess_ms,
            block_iterations=block_iterations,
            max_block_iterations=max(block_iterations) if block_iterations else 0,
            num_completed_decode_blocks=len(next_token_ids_list),
        )

        nvtx.range_pop()  # run
        return logits_output, next_token_ids_list, block_iterations, can_run_cuda_graph


Algorithm = LowConfidence
