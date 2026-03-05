from typing import List
import torch
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.fdfo_utils import fdfo_post_process
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceFDFO(DllmAlgorithm):
    enable_first_done_first_out = True
    USE_TORCH_COMPILE = False
    USE_TRITON_KERNEL = True # Use Triton kernel by default
    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
         
    def _pick_tokens(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
        ) -> None:
        """
        Pick tokens based on confidence threshold using PyTorch implementation.
        For each position:
        1. Compute softmax to get probabilities
        2. Find argmax and its confidence (probability)
        3. If confidence > threshold, transfer the token
        4. If no token exceeds threshold in a batch, force select the highest confidence one
        """
        self._pick_tokens_torch(forward_batch, full_logits)
            
    def _pick_tokens_torch(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> None:
        batch_size = forward_batch.batch_size
        # Reshape to [batch_size, block_size, vocab_size]
        vocab_size = full_logits.shape[-1]
        full_logits = full_logits.view(batch_size, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        block_mask_index = input_ids == self.mask_id
        x = torch.argmax(full_logits, dim=-1)
        probs = torch.nn.functional.softmax(full_logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        # Apply mask to confidence (set non-mask positions to -inf)
        # Apply mask to confidence (set non-mask positions to -inf)
        confidence = torch.where(
        block_mask_index,
        confidence,
        torch.tensor(-float("inf"), device=confidence.device),
        )
        transfer_index = confidence > self.threshold
        # For batches with no transfer, force select the highest confidence token
        # has_transfer: [batch_size]
        has_transfer = transfer_index.sum(dim=1) > 0
        # Find top-1 for each batch: [batch_size]
        _, top1_indices = torch.topk(confidence, k=1, dim=1) # [batch_size, 1]
        top1_indices = top1_indices.squeeze(-1) # [batch_size]
        batch_indices = torch.arange(batch_size, device=top1_indices.device)
        top1_mask = torch.zeros_like(transfer_index, dtype=torch.bool)
        top1_mask[batch_indices, top1_indices] = True
        # Merge: if has_transfer, use transfer_index; otherwise use top1_mask
        transfer_index = torch.where(
        has_transfer.unsqueeze(-1), transfer_index, top1_mask # [batch_size, 1]
        )
        # Update tokens: only update where transfer_index is True and mask exists
        x = torch.where(block_mask_index, x, input_ids)
        input_ids = torch.where(transfer_index, x, input_ids)
        # Write back (flatten)
        forward_batch.input_ids = input_ids.view(-1)
        
    def run(
    self,
    model_runner: ModelRunner,
    forward_batch: ForwardBatch,
    ) -> GenerationBatchResult:
    # Forward pass through the model (no preprocessing needed)
    # each block may contain mask or finished tokens
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Post-forward processing pick tokens and build output list
        next_token_ids_list, accept_length_per_req_cpu = fdfo_post_process(
            forward_batch=forward_batch,
            full_logits=logits_output.full_logits,
            pick_tokens_func=self._pick_tokens,
            mask_id=self.mask_id,
            block_size=self.block_size,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids_list,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
        
Algorithm = LowConfidenceFDFO
