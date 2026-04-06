# FDFO (First Decode First Out) 深入分析与优化方案

## 1. FDFO 设计动机：为什么需要 FDFO？

### 1.1 LowConfidence 的根本问题

LowConfidence 算法在单次 `run()` 调用中执行**最多 block_size（32）次**前向推理：

```
scheduler.run_batch()
  └─ tp_worker.forward_dllm()
       └─ LowConfidence.run()           # 一次 run_batch 调用
            ├─ model_forward (iter 0)     # 第 1 次前向
            ├─ confidence_and_transfer    # CPU 逐 block Python 循环
            ├─ model_forward (iter 1)     # 第 2 次前向
            ├─ confidence_and_transfer
            ├─ ...                        # 最多 32 次迭代
            ├─ model_forward (iter N)
            └─ final_forward              # 第 N+1 次前向
```

**nsys 实测数据（LowConfidence，3 请求 × 128 tokens，TP=2）：**

| 标记 | 总时间 | 调用次数 | 平均耗时 |
|------|--------|----------|----------|
| `LowConfidence::run` | 6.72s | 36 | **186.7ms** |
| `model_forward` | 2.14s | 750 | 2.85ms |
| `confidence_and_transfer` | 4.39s | 750 | **5.85ms** |
| `final_forward` | 84.4ms | 30 | 2.81ms |
| `reshape_output` | 1.08ms | 30 | 36μs |

关键瓶颈：

1. **`confidence_and_transfer` 耗时是 `model_forward` 的 2.05 倍**（5.85ms vs 2.85ms）。原因是该逻辑对每个 block 使用 Python for 循环串行处理：`for batch_id in range(batch_size):`，在循环内部做逐 block 的 GPU tensor 操作、D2H 传输和条件判断。
2. **scheduler 被阻塞**：一次 `run_batch` 平均 186.7ms，期间 scheduler 无法接收新请求、无法调度其他请求、完全被堵死。
3. **无法 batch 多请求**：因为每个请求的迭代次数不同（取决于 confidence 收敛速度），所以即使有多个请求，也只能逐个串行处理每个 block 的所有迭代。

### 1.2 FDFO 的核心思路

FDFO 将 LowConfidence 的"N 次迭代 → 1 次调度" 拆解为 "1 次迭代 → 1 次调度"：

```
scheduler loop iter 1:  get_next_batch → run_batch → process_result
  └─ FDFO.run(): 1 次 model_forward + 1 次 post_forward_process
scheduler loop iter 2:  get_next_batch → run_batch → process_result
  └─ FDFO.run(): 1 次 model_forward + 1 次 post_forward_process
...
```

每一轮 scheduler 循环只执行**一次**前向推理，然后将控制权交还 scheduler。scheduler 根据 `accept_length` 判断该请求当前这轮能否直接提交结果：
- `accept_length == 0`：该 block 在本轮 forward 输入中包含 mask token，因此本轮结果只会写入 `dllm_incomplete_ids`，下一轮还要基于这些 token 再做一次 forward
- `accept_length == block_size`：该 block 在本轮 forward 输入中已经是完整 token block，可以直接提交 output tokens 并进入下一个 block

注意：当前代码里，`accept_length` 不是“`_pick_tokens()` 之后是否已经没有 mask”的标记，而是“本轮 forward 之前这个 block 是否已经是完整 block”的标记。这个语义由 `mask_counts` 在 `_pick_tokens()` 之前计算决定。

## 2. FDFO 完整数据流

### 2.1 请求生命周期

```
用户请求到达
  │
  ▼
waiting_queue（scheduler 主队列）
  │  _fetch_waiting_reqs()
  ▼
DllmManager.waiting_queue
  │  get_decode_requests() / get_prefill_requests()
  ▼
_process_dllm_batches() → PrefillAdder
  │  add_dllm_staging_req() / add_one_req()
  ▼
can_run_list → DllmManager.staging_queue
  │  _create_dllm_batch()
  ▼
ScheduleBatch（forward_mode=DLLM_EXTEND）
  │  run_batch() → tp_worker.forward_dllm()
  ▼
FDFO.run()  →  1 次 forward  →  post_forward_process
  │
  ▼
process_batch_result_dllm_fdfo()
  ├─ accept_length == 0  → dllm_incomplete_ids = next_token_ids
  │     释放 kv cache（因为本轮 forward 的输入 block 含 mask，基于该输入算出的 kv 不能复用）
  │     下一轮：req._init_fill_ids_for_dllm() 重新填充 incomplete ids
  │     重新进入 staging_queue → 继续去噪
  │
  └─ accept_length == block_size  → dllm_incomplete_ids = []
        output_ids.append(token) × N
        check_finished()
        下一轮：req._init_fill_ids_for_dllm() 推进 block_offset
        填充下一个 block（prefill: dllm_ids 中下一段 / decode: 新 mask block）
```

### 2.2 fill_ids 的演进（以 origin_input_ids 长度 50，block_size=32 为例）

```python
# _init_fill_ids_for_dllm 首次调用：
padding = (-50) % 32 = 14  # 凑到 64 的整数倍
dllm_ids = origin_input_ids + [mask_id] * 14  # len=64
fill_ids = dllm_ids[:32]  # 第一个 block（纯 input，无 mask）

# Prefill 第 1 个 block 完成后（accept_length == 32）：
fill_ids += dllm_ids[32:64]  # [origin_ids[:32]] + [origin_ids[32:50] + mask*14]
# 第二个 block 含 14 个 mask token → 需要去噪

# 去噪迭代中（accept_length == 0）：
fill_ids = fill_ids[:prefix_len] + dllm_incomplete_ids
# prefix_len 是已经 cache 的前缀长度，incomplete_ids 是上次 pick_tokens 后的结果

# 第二个 block 去噪完成后（accept_length == 32）：
# 进入 decode 阶段：
fill_ids += [mask_id] * 32  # 新增全 mask block
```

### 2.3 KV Cache 管理

FDFO 有一个关键的 KV cache 策略：**凡是本轮 forward 输入里包含 mask 的 block，其对应新增 KV cache 都必须释放**。

```python
# process_batch_result_dllm_fdfo，accept_length == 0 分支：
if new_fill_len > old_prefix_len:
    kv_indices_to_free = batch.req_to_token_pool.req_to_token[
        req.req_pool_idx, old_prefix_len:new_fill_len
    ]
    self.token_to_kv_pool_allocator.free(kv_indices_to_free)
```

原因：block 中的 mask token 在每次迭代后都可能被替换成不同的 token（pick_tokens 结果），而 KV cache 是基于旧 input_ids 计算的，内容已经不正确。下一次迭代必须重新计算这些位置的 KV。

而在 `get_next_batch_to_run` 中，对于没有 incomplete_ids 的请求，会调用 `stash_chunked_request`（即 `tree_cache.cache_unfinished_req`）来缓存已完成的前缀。

## 3. nsys Profile 数据分析

### 3.1 FDFO 耗时分布

**FDFO 实测数据（3 请求 × 128 tokens，TP=2）：**

| 标记 | 总时间 | 调用次数 | 平均耗时 | 占比 |
|------|--------|----------|----------|------|
| `FDFO::run` | 5.47s | 708 | **7.73ms** | 100% |
| `FDFO::post_forward_process` | 4.05s | 708 | **5.72ms** | 74.0% |
| `FDFO::model_forward` | 1.42s | 708 | **2.00ms** | 25.9% |
| `FDFO::pick_tokens` | 558.6ms | 708 | 789μs | 10.2% |
| `FDFO::build_output` | 27.6ms | 708 | 39μs | 0.5% |
| `output::process_fdfo` | 187.0ms | 708 | 264μs | — |
| `scheduler::get_next_batch` | 501.3ms | 6464 | 77.6μs | — |
| `scheduler::get_new_batch_dllm` | 424.6ms | 6464 | 65.7μs | — |

### 3.2 瓶颈分析

**核心瓶颈：`post_forward_process` 占 FDFO::run 的 74%**，其中包含：

#### 瓶颈 1：mask_counts D2H 传输（~70% of post_forward_process）

```python
mask_counts_cpu = (
    (forward_batch.input_ids == self.mask_id)
    .view(batch_size, self.block_size)
    .sum(dim=1)
    .tolist()  # D2H 同步传输！
)
```

这是一个隐式的 `cudaMemcpy D2H` + `cudaDeviceSynchronize`：`.tolist()` 会触发 GPU → CPU 的同步数据传输。在 batch_size=1、block_size=32 的情况下，传输的数据量极小（1 个 int），但 **同步开销**（CUDA stream 等待 + PCIe 延迟）是固定的。

#### 瓶颈 2：pick_tokens 中的多次 GPU 操作

```python
def _pick_tokens(self, forward_batch, full_logits):
    full_logits = full_logits.view(batch_size, self.block_size, vocab_size)  # reshape
    input_ids = forward_batch.input_ids.view(batch_size, self.block_size)    # reshape
    block_mask_index = input_ids == self.mask_id                              # 比较
    x = torch.argmax(full_logits, dim=-1)                                     # argmax
    probs = torch.nn.functional.softmax(full_logits, dim=-1)                  # softmax
    confidence = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1) # gather
    confidence = torch.where(block_mask_index, confidence, -inf)              # where
    transfer_index = confidence > self.threshold                              # 比较
    has_transfer = transfer_index.sum(dim=1) > 0                              # reduce+比较
    _, top1_indices = torch.topk(confidence, k=1, dim=1)                      # topk
    # ... 更多操作
```

pick_tokens 平均耗时 789μs，包含约 15 次 CUDA kernel launch。每次 launch 的 overhead 约 5-10μs，但因为 batch_size 很小（1-3），实际 kernel 执行时间极短，launch overhead 占主导。

#### 瓶颈 3：build_output 中的 `.tolist()` 又一次 D2H

```python
next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size).tolist()
```

这又是一次 D2H 同步传输。虽然单次只有 39μs，但它紧跟在 pick_tokens 之后，形成第二个同步点。

#### 瓶颈 4：scheduler 空转开销

`get_next_batch` 被调用 6464 次但只有 708 次产生了实际 batch（生成 run_batch 调用）。也就是说，**约 89% 的 scheduler 循环是空转**——在 recv_requests 和 get_next_batch 之间反复轮询。这本身不是 FDFO 的问题，而是 scheduler 事件循环的固有行为，但它意味着 scheduler CPU 侧的效率也有优化空间。

### 3.3 LowConfidence vs FDFO 对比

| 指标 | LowConfidence | FDFO | 比值 |
|------|--------------|------|------|
| 总 run 时间 | 6.72s | 5.47s | **0.81×** |
| 平均每次 run_batch | 186.7ms | 7.73ms | **0.041×** |
| model_forward 总时间 | 2.14s + 84.4ms = 2.22s | 1.42s | **0.64×** |
| model_forward 调用次数 | 750 + 30 = 780 | 708 | 0.91× |
| 后处理总时间 | 4.39s (confidence_and_transfer) | 4.05s (post_forward) | 0.92× |
| scheduler 阻塞时间/次 | 186.7ms | 7.73ms | **24× 改善** |

关键发现：
1. FDFO 的总 run 时间更短（5.47s vs 6.72s），因为 FDFO 的 `pick_tokens` 是向量化的 batch 操作，而 LowConfidence 的 `confidence_and_transfer` 是 Python for 循环逐 block 处理。
2. FDFO 的 `run()` 中没有显式 `final_forward`，但这不等于对应工作完全消失。对于带 mask 的 block，FDFO 会把“基于最终 token 再 forward 一次以建立正确 KV”拆到下一轮 scheduler 迭代里完成；而 LowConfidence 是在同一次 `run()` 里通过 `final_forward` 完成这一步。
3. **FDFO 最大的收益是将 scheduler 阻塞时间从 186.7ms 降到 7.73ms**，使 scheduler 能够在去噪迭代之间插入新请求的处理。

## 4. 优化方案

### 4.1 优化 1：消除 mask_counts 的 D2H 同步（高优先级）

**问题**：`mask_counts_cpu` 的 `.tolist()` 是 `post_forward_process` 中最大的延迟来源。

**前提约束**：当前实现中，`mask_counts` 必须在 `_pick_tokens()` 之前统计，才能维持现有的两阶段完成语义。因此，任何“合并 D2H”的优化都必须先确认是否接受语义变化。

**重要说明：不能直接把 `mask_counts` 延迟到 `_pick_tokens()` 之后再计算。**

当前实现中，`mask_counts` 是在 `_pick_tokens()` 之前统计的：

```python
mask_counts_cpu = (
    (forward_batch.input_ids == self.mask_id)
    .view(batch_size, self.block_size)
    .sum(dim=1)
    .tolist()
)

self._pick_tokens(forward_batch, full_logits)
```

这样做的目的不是偶然实现细节，而是为了保留两阶段语义：

1. 本轮如果输入里还有 mask，则本轮结果只能作为 `dllm_incomplete_ids`
2. 下一轮基于这些 token 再做一次 forward，才能得到与最终 token 对齐的 KV cache

如果把 `mask_counts` 挪到 `_pick_tokens()` 之后，`accept_length` 的语义就会从“本轮 forward 输入是否完整”变成“本轮 pick 之后是否完整”，这会改变当前调度与 KV 管理语义，不是一个无损优化。

**方案 A（仅在接受语义变化的前提下讨论）：延迟到 build_output 之后统一做 D2H**

如果未来明确要把 FDFO 改成“本轮 pick 完就允许提交”的语义，可以考虑把 mask_counts 的判断推迟到已经有 `.tolist()` 调用的地方合并：

```python
def _post_forward_process(self, forward_batch, full_logits):
    batch_size = forward_batch.batch_size
    # 不在这里做 mask_counts 的 D2H
    
    # GPU 上计算 mask counts（保留 tensor，不 .tolist()）
    mask_counts = (
        (forward_batch.input_ids == self.mask_id)
        .view(batch_size, self.block_size)
        .sum(dim=1)
    )  # 仍在 GPU 上

    self._pick_tokens(forward_batch, full_logits)

    # build_output 需要 .tolist() 做 D2H，合并到这一个同步点
    next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size)
    
    # 拼接后一次性传输
    combined = torch.cat([next_token_ids, mask_counts.unsqueeze(1)], dim=1)
    combined_cpu = combined.tolist()  # 单次 D2H
    
    next_token_ids_list = []
    accept_length_per_req_cpu = []
    for i in range(batch_size):
        next_token_ids_list.append(combined_cpu[i][:self.block_size])
        mc = combined_cpu[i][self.block_size]
        accept_length_per_req_cpu.append(self.block_size if mc == 0 else 0)
    
    return next_token_ids_list, accept_length_per_req_cpu
```

**风险**：这不是纯性能优化，而是行为变化。它会让“本轮刚刚补完 mask 的 block”直接走完成路径，跳过当前实现依赖的“下一轮再 forward 一次以建立正确 KV”这一步。

**预期收益**：如果行为允许修改，理论上可以消除一个 D2H 同步点，节省约 3-4ms/iter。

**方案 B：完全在 GPU 上判断是否完成**

mask_counts == 0 的判断可以完全在 GPU 上完成，只有最终需要返回 token ids 时才传输：

```python
# GPU 上判断
block_finished = mask_counts == 0  # bool tensor, 在 GPU 上

# 只传输 finished blocks 的 token ids
# 未完成的 block 不需要传输 token ids 到 CPU（scheduler 不需要看到它们）
```

但这同样需要先定义清楚“完成”的语义究竟是以 forward 前还是 pick 后为准；否则只是把当前歧义搬到 GPU 侧，改动较大且容易引入行为偏差。

### 4.2 优化 2：融合 pick_tokens 的 CUDA kernels（中优先级）

**问题**：`_pick_tokens` 中约 15 个小 kernel 逐个 launch，每个 kernel 执行时间极短但 launch overhead 累积。

**方案 A：写一个融合的 Triton kernel**

将 softmax → gather → threshold → topk → where 融合为单个 kernel：

```python
@triton.jit
def fused_pick_tokens_kernel(
    logits_ptr, input_ids_ptr, output_ids_ptr,
    mask_id, threshold, block_size, vocab_size,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
):
    batch_id = tl.program_id(0)
    # 在单个 kernel 中完成：
    # 1. argmax + softmax + gather → confidence
    # 2. threshold 判断 → transfer_index
    # 3. fallback to top1 if no transfer
    # 4. 写回 output_ids
    ...
```

**预期收益**：从 ~789μs 降到 ~100-200μs（消除 14 次 kernel launch overhead）。

**方案 B：用 torch.compile 自动融合**

```python
@torch.compile
def _pick_tokens_compiled(input_ids, full_logits, mask_id, threshold, block_size):
    ...
```

torch.compile 可以自动融合部分操作，但可能无法处理 topk 和条件逻辑的融合。需要实测。

## 6. Code Review 更正

下面是结合当前代码对本文前述结论做的校正，避免后续优化基于错误假设展开。

### 6.1 `accept_length` 的语义不是“本轮 pick 后是否完成”

当前代码：

```python
mask_counts_cpu = (
    (forward_batch.input_ids == self.mask_id)
    .view(batch_size, self.block_size)
    .sum(dim=1)
    .tolist()
)

self._pick_tokens(forward_batch, full_logits)

accept_length_per_req_cpu.append(
    self.block_size if mask_counts_cpu[i] == 0 else 0
)
```

这说明 `accept_length` 取决于 forward 前的输入，而不是 `_pick_tokens()` 后的结果。

因此：

- 一个 block 即使在 `_pick_tokens()` 后已经没有 mask，也可能仍返回 `accept_length == 0`
- 它会先被写入 `req.dllm_incomplete_ids`
- 下一轮由 `req._init_fill_ids_for_dllm()` 把这些 token 重新装回 `fill_ids`
- 再做一次 forward 后，才会以 `accept_length == block_size` 进入完成路径

### 6.2 `accept_length == 0` 不等价于“当前结果里仍有 mask”

`process_batch_result_dllm_fdfo()` 中：

```python
if result.accept_length_per_req_cpu[idx] == 0:
    req.dllm_incomplete_ids = next_token_ids
```

这里保存的是 `_pick_tokens()` 后的 `next_token_ids`。它可能仍含 mask，也可能已经全部变成普通 token；但只要本轮 forward 的输入里原本有 mask，就仍走 incomplete 路径。

更准确地说，`accept_length == 0` 表示：当前 block 还不能提交，因为还缺一轮基于最终 token 的 forward 来建立正确 KV。

### 6.3 “FDFO 没有 final_forward”不等于“省掉了 final_forward 的工作”

LowConfidence 在同一次 `run()` 中显式执行 `final_forward`。FDFO 没有这个函数级别的尾部 forward，但它把同样的语义拆到了下一轮 scheduler：

1. 本轮用带 mask 的 block 做 forward
2. `_pick_tokens()` 生成 candidate token block
3. 该 block 先保存在 `dllm_incomplete_ids`
4. 下一轮重新 forward 这个 candidate block
5. 这时 block 才作为完整 block 提交

因此，FDFO 的优势更准确地说是：

- 把长时间阻塞的一次 `run_batch` 拆成多个短轮次
- 让 scheduler 能在轮次之间重新获得控制权
- 而不是简单“删除了 final forward”

### 6.4 对优化方案的影响

基于上面的语义，本文里所有涉及 `mask_counts`、`accept_length`、KV 释放时机的优化，都必须先回答一个问题：

**我们是要保持当前两阶段语义，还是要改成 pick 后即可提交的新语义？**

如果不先明确这一点：

- 某些看似“只省一次 D2H”的优化，实际是在修改调度行为
- 某些看似“减少一次 forward”的优化，实际是在改变 KV 正确性假设

所以建议后续优化分成两类：

1. **语义保持型优化**：不改变 `accept_length` 的定义，只减少同步和 kernel launch
2. **语义变更型优化**：允许 pick 后立刻完成，但要重新设计 KV 正确性与输出提交路径

## 7. 附录：LowConfidence confidence_and_transfer 的向量化改造

虽然本报告聚焦 FDFO，但 LowConfidence 的 `confidence_and_transfer` 瓶颈值得一提，因为 FDFO 的 `_pick_tokens` 本质上就是它的向量化版本。

**LowConfidence 当前实现（Python for 循环，5.85ms/call）：**

```python
for batch_id in range(batch_size):
    block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
    block_mask_index = block_input_ids == self.mask_id
    if torch.sum(block_mask_index).item() == 0:  # D2H！
        continue
    curr_logits = logits_output.full_logits[curr_block_start:curr_block_end]
    x = torch.argmax(curr_logits, dim=-1)
    p = torch.gather(F.softmax(curr_logits, dim=-1), ...)
    confidence = torch.where(block_mask_index, p, -inf)
    transfer_index = confidence > self.threshold
    if transfer_index.sum().item() == 0:  # D2H！
        _, select_index = torch.topk(confidence, k=1)
        transfer_index[select_index] = True
    block_input_ids[transfer_index] = x[transfer_index]
```

每个 batch_id 都有 2 次 D2H 同步（`.item()` 调用）+ 多个小 kernel launch。

**FDFO 的 _pick_tokens 已经做了向量化（789μs/call）**——没有 Python for 循环，所有操作都是 batch 级别的 tensor 操作。如果将 FDFO 的 `_pick_tokens` 逻辑反向移植到 LowConfidence，可以将 `confidence_and_transfer` 从 5.85ms 降到 ~1ms 级别。
