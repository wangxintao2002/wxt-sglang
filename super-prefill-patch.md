# Super Prefill (SP) Patch Analysis

## 概述

Super Prefill 是 FDFO (First Decode First Out) 的一种变体，通过**每轮 forward 同时处理 2 个 block**来加速 dLLM 推理。

**核心思想对比：**

| 模式 | 每轮处理 | 特点 |
|------|----------|------|
| 普通 FDFO | 1 个 block | 细粒度调度，调度开销大 |
| Super Prefill | 2 个 block | 通过 fused attention 优化，减少 kernel launch |

**核心优化原理：**
- 普通 FDFO 每次处理 1 个 block，需要多次 kernel launch
- SP 同时处理 2 个 block，将多个 attention 计算融合成一个 kernel 序列
- 特别适合小 block_size（如 4）的场景，可减少约 70% 的执行时间

---

## 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `python/sglang/srt/layers/attention/dllm_attention.py` | 新增 | SP 核心 attention kernel |
| `python/sglang/srt/dllm/algorithm/low_confidence_fdfo_sp.py` | 新增 | SP 专用算法类 |
| `python/sglang/srt/dllm/algorithm/fdfo_utils.py` | 修改 | 新增 `fdfo_sp_post_process` |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | 修改 | 支持 SP 的 attention 路径 |
| `python/sglang/srt/dllm/config.py` | 修改 | 新增 SP 配置参数 |
| `python/sglang/srt/dllm/algorithm/base.py` | 修改 | 新增 `enable_super_prefill` 标记 |
| `python/sglang/srt/dllm/algorithm/__init__.py` | 修改 | 新增 SP 检测函数 |
| `python/sglang/srt/dllm/mixin/req.py` | 修改 | 新增 SP 状态管理 |
| `python/sglang/srt/managers/scheduler.py` | 修改 | 新增 SP 处理分支 |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | 修改 | 新增 SP 输出处理 |

---

## 1. 核心 Attention Kernel (`dllm_attention.py`)

### 1.1 `super_prefill_fused_attn_v2`

**功能：** 融合的 2-block attention 计算

**输入形状：** `[batch_size * 2 * block_size, hidden_dim]`

**计算分解：**

```
Q = [Q1, Q2], K = [K1, K2], V = [V1, V2]
Kc = 历史 paged KV cache

Step 1: Paged attention [Q1,Q2] × Kc
  → o_paged, s_paged

Step 2: Ragged shared attention [Q1,Q2] × K1
  → o_shared, s_shared

Step 3: Ragged local attention Q2 × K2
  → o_local_q2, s_local_q2

Step 4: Merge 结果
  O1_final = merge(Q1×K1, Q1×Kc)
  O2_final = merge(Q2×K2, merge(Q2×K1, Q2×Kc))
```

**使用的三个 FlashInfer wrapper：**

| Wrapper | 用途 | Query | Key/Value |
|---------|------|-------|-----------|
| `prefill_wrapper_paged` | 历史上下文 | [Q1,Q2] | Kc (paged KV) |
| `prefill_wrapper_ragged` | block0 当前 KV | [Q1,Q2] | K1,V1 (ragged) |
| `prefill_wrapper_ragged_local` | block1 当前 KV | Q2 | K2,V2 (ragged) |

### 1.2 `call_dllm_begin_forward`

**功能：** 初始化所有三个 attention wrapper

**关键逻辑：**
- 填充 paged KV indices (标准 extend 逻辑)
- 填充 ragged indptrs：
  - `qo_indptr_shared`: `[0, 2b, 4b, ...]` (Q=2b per batch)
  - `kv_indptr_shared`: `[0, b, 2b, ...]` (K=b per batch)
  - `qo_indptr_local`: `[0, b, 2b, ...]` (Q=b per batch)
- 调用三个 wrapper 的 `begin_forward`

### 1.3 Triton Kernel `_fill_three_indptr_kernel_impl`

**功能：** 单次 kernel launch 填充三个 indptr buffer

**优化点：** 避免多次 CPU-GPU 同步，减少 kernel launch 开销

---

## 2. FlashInfer Backend 适配 (`flashinfer_backend.py`)

### 2.1 新增 Dispatch Reason

```python
class WrapperDispatch(Enum):
    ...
    DLLM_SUPER_PREFILL = auto()  # 新增
```

### 2.2 初始化逻辑修改

```python
# 检测 SP 模式
elif self.is_dllm_model and self.dllm_config.enable_super_prefill:
    self.num_wrappers = 1
    self.dispatch_reason = WrapperDispatch.DLLM_SUPER_PREFILL

# 初始化额外的 wrapper 和 buffer
if self.is_dllm_model and self.dllm_config.enable_super_prefill:
    self.prefill_wrapper_ragged_local = BatchPrefillWithRaggedKVCacheWrapper(...)
    self.dllm_qo_indptr_shared_buf = torch.zeros((max_bs + 1,), dtype=torch.int32)
    self.dllm_kv_indptr_shared_buf = torch.zeros((max_bs + 1,), dtype=torch.int32)
    self.dllm_qo_indptr_local_buf = torch.zeros((max_bs + 1,), dtype=torch.int32)
```

### 2.3 Attention Forward 路径

```python
if self.is_dllm_model and self.dllm_config.enable_super_prefill:
    from sglang.srt.layers.attention.dllm_attention import super_prefill_fused_attn_v2
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
    o = super_prefill_fused_attn_v2(
        self.dllm_config.block_size,
        forward_batch.batch_size,
        q, k, v,
        forward_batch,
        layer,
        prefill_wrapper_paged,
        self.prefill_wrapper_ragged,
        self.prefill_wrapper_ragged_local,
        logits_soft_cap,
    )
    return o.view(-1, layer.tp_q_head_num * layer.head_dim)
```

---

## 3. DllmConfig 配置 (`dllm/config.py`)

### 3.1 新增参数

```python
class DllmConfig:
    def __init__(
        self,
        ...
        enable_super_prefill: bool = False,  # 新增
    ):
        ...
        self.enable_super_prefill = enable_super_prefill
        self.double_block_size = block_size * 2  # 新增
```

### 3.2 动态 Block Size

```python
def get_block_size(self) -> int:
    """Return the effective block size for batch/KV slot allocation.
    SP mode uses 2*block_size as the unit; non-SP uses block_size.
    """
    if self.enable_super_prefill:
        return self.double_block_size
    return self.block_size
```

### 3.3 自动检测 SP 需求

```python
from sglang.srt.dllm.algorithm import get_algorithm_sp_requirement
enable_super_prefill = get_algorithm_sp_requirement(server_args.dllm_algorithm)
```

---

## 4. 算法基类支持 (`algorithm/base.py` & `algorithm/__init__.py`)

### 4.1 基类新增标记

```python
class DllmAlgorithm:
    enable_super_prefill: bool = False  # 新增类属性
```

### 4.2 SP 检测函数

```python
def get_algorithm_sp_requirement(algorithm_name: str) -> bool:
    """Return True if algorithm requires super prefill mode."""
    return getattr(algo_name_to_cls[algorithm_name], "enable_super_prefill", False)
```

---

## 5. SP 专用算法 (`low_confidence_fdfo_sp.py`)

```python
class LowConfidenceFDFOSuperPrefill(DllmAlgorithm):
    """Low Confidence FDFO with Super Prefill: 2-block decode per round."""

    requires_fdfo_mode: bool = True
    enable_super_prefill: bool = True  # 关键标记

    def run(self, model_runner, forward_batch):
        # 标准 forward + SP 专用后处理
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        next_token_ids_list, accept_length_per_req_cpu = fdfo_sp_post_process(...)
        return logits_output, next_token_ids_list, accept_length_per_req_cpu, can_run_cuda_graph
```

---

## 6. SP 后处理逻辑 (`fdfo_utils.py`)

### 6.1 `fdfo_sp_post_process`

**功能：** 处理 2-block 的 mask 状态和 accept 判定

**三种情况：**

| mask_counts (block0, block1) | 动作 | accept_length |
|------------------------------|------|---------------|
| [>0, >0] | 两个 block 都还在 decode，从 block0 pick | 0 |
| [0, >0] | block0 已完成，从 block1 pick | block_size |
| [0, 0] | 两个 block 都完成 | double_block_size |

**关键逻辑：**
```python
# Compute mask counts per (batch, block) BEFORE pick_tokens
mask_counts_tensor = (
    (forward_batch.input_ids == mask_id)
    .view(batch_size, 2, block_size)
    .sum(dim=2)
)  # shape: [batch_size, 2]

# Select block to apply pick_tokens
# block0 has masks → pick from block0 (idx=0)
# block0 is clean → pick from block1 (idx=1)
block_select_idx = (mask_counts_tensor[:, 0] == 0).long()

# Compute accept_length on GPU
accept_lens = torch.where(
    mask_counts_tensor[:, 1] == 0,
    double_block_size,
    torch.where(
        mask_counts_tensor[:, 0] == 0,
        block_size,
        0,
    ),
)
```

---

## 7. Req Mixin SP 支持 (`dllm/mixin/req.py`)

### 7.1 初始化 SP 状态

```python
def init_diffusion_llm(self, dllm_config):
    if self.dllm_config is not None:
        if self.dllm_config.enable_super_prefill:
            # SP 直接从 prefill 开始
            self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
        else:
            # 普通 FDFO 根据长度判断
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
```

### 7.2 `_init_fill_ids_for_dllm_fdfo_sp`

**三种状态处理：**

| `dllm_incomplete_ids` 长度 | 状态 | 动作 |
|---------------------------|------|------|
| 0 | 首次 decode 或全部完成 | 追加两个新的 mask block |
| block_size | block0 完成，block1 还在 decode | 滑动一个 block 前进 |
| 2*block_size | 两个 block 都还在 decode | 原地替换最后两个 block |

```python
def _init_fill_ids_for_dllm_fdfo_sp(self):
    if first_call:
        # Ceil-align dllm_ids to double_block_size
        self.dllm_ids = (
            self.origin_input_ids
            + [mask_id] * (-len(self.origin_input_ids) % double_block_size)
        )
        self.fill_ids = list(self.dllm_ids[:double_block_size])

    elif len(self.dllm_incomplete_ids) == double_block_size:
        # Both blocks still decoding: replace last 2b tokens in-place
        self.fill_ids = self.fill_ids[:-double_block_size] + self.dllm_incomplete_ids

    elif len(self.dllm_incomplete_ids) == block_size:
        # block0 confirmed, block1 still decoding: slide forward one block
        self.fill_ids = (
            self.fill_ids[:-block_size]
            + self.dllm_incomplete_ids
            + one_block_of_masks
        )

    else:
        # Both blocks confirmed: advance to next 2 blocks
        if fill_len < dllm_len:
            # Still in prefill region
            self.fill_ids = self.fill_ids + self.dllm_ids[fill_len:fill_len + double_block_size]
        else:
            # In decode region
            self.fill_ids = self.fill_ids + one_block_of_masks + one_block_of_masks
```

---

## 8. Scheduler SP 支持 (`managers/scheduler.py`)

### 8.1 输出处理分支

```python
def process_batch_result(self, batch, result):
    if batch.forward_mode.is_extend():
        if batch.is_dllm():
            if self.dllm_config.enable_super_prefill:
                self.process_batch_result_dllm_fdfo_sp(batch, result)
            elif self.dllm_config.enable_fdfo:
                self.process_batch_result_dllm_fdfo(batch, result)
            else:
                self.process_batch_result_dllm(batch, result)
```

---

## 9. SP 输出处理 (`scheduler_output_processor_mixin.py`)

### 9.1 `process_batch_result_dllm_fdfo_sp`

**功能：** 处理 2-block 的 KV cache 管理

**关键逻辑：**
- 根据 `accept_length` 判断哪些 block 完成
- block0 完成：保留其 KV，释放 block1 的 KV（如果还在 decode）
- block1 完成：保留其 KV
- 两个都完成：推进到下一对 blocks

---

## 性能收益分析

### 为什么 SP 更快？

1. **减少 Kernel Launch**
   - 普通 FDFO：每个 block 需要独立的 attention kernel launch
   - SP：两个 block 的 attention 融合在一个 kernel 序列中

2. **更好的 GPU 利用率**
   - 小 block_size（如 4）时，单次 kernel 计算量小，GPU 利用率低
   - SP 通过处理 2 个 block，提高了每次 kernel 的计算密度

3. **减少 CPU-GPU 同步**
   - 融合的 indptr 填充 kernel 减少了同步点

### 实测数据

| block_size | 普通 FDFO | Super Prefill | 加速比 |
|------------|-----------|---------------|--------|
| 4 | baseline | +70% | 1.7x |
| 32 | baseline | +20% | 1.2x |

**结论：** SP 在小 block_size 时收益更大，因为小 block 的调度开销占比更高。

---

## 调试与 Profile

### NVTX Range

SP 代码中添加了详细的 NVTX range 用于 profiling：

```python
# dllm_attention.py
nvtx.range_push("sp_paged_attn")
nvtx.range_push("sp_shared_attn")
nvtx.range_push("sp_local_attn")
nvtx.range_push("sp_merge")

# fdfo_utils.py
nvtx.range_push("fdfo_sp_post_process")
nvtx.range_push("sp_pick_tokens")
```

### 关键事件记录

```python
record_dllm_event(
    "dllm_algo_run",
    algo="LowConfidenceFDFOSuperPrefill",
    batch_size=batch_size,
    run_ms=...,
    model_forward_ms=...,
    postprocess_ms=...,
)
```

---

## 总结

Super Prefill 通过以下改动实现了 dLLM 推理的加速：

1. **核心创新**：`dllm_attention.py` 中的 fused 2-block attention kernel
2. **配置支持**：`DllmConfig` 新增 SP 参数和动态 block size
3. **算法支持**：`LowConfidenceFDFOSuperPrefill` 类标记 SP 模式
4. **状态管理**：`ReqDllmMixin` 处理 2-block 的 fill_ids 逻辑
5. **调度支持**：Scheduler 新增 SP 处理和输出处理路径
6. **后处理**：`fdfo_sp_post_process` 处理 2-block 的 mask 判定

**适用场景：**
- 小 block_size（如 4）的模型
- 对吞吐敏感的场景
- GPU 利用率不高的场景

**注意事项：**
- SP 只适用于 FlashInfer backend
- 需要额外的 memory buffer 用于三个 wrapper
- 算法必须显式标记 `enable_super_prefill = True`
