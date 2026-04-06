# dLLM (Diffusion Language Model) Inference Pipeline in SGLang

## 1. Overview

dLLM (Diffusion LLM) is a non-autoregressive generation paradigm. Unlike传统 AR 模型逐 token 生成，dLLM 通过**迭代去噪**的方式，每次对一个 block 内的多个 masked position 同时预测，逐步将低置信度的 mask token 替换为真实 token。

SGLang 支持两种 dLLM 推理算法：
- **LowConfidence**: 多次 forward pass 迭代去噪
- **LowConfidenceFDFO**: 单次 forward pass + First Decode First Out 调度

### 支持的模型

| 模型架构 | block_size | mask_id |
|---|---|---|
| LLaDA2MoeModelLM | 32 | 156895 |
| SDARForCausalLM | 4 | 151669 |
| SDARMoeForCausalLM | 4 | 151669 |

> 配置来源: `python/sglang/srt/dllm/config.py:36-40`

---

## 2. 整体请求生命周期

```
HTTP Request
    │
    ▼
┌─────────────┐
│  Tokenizer  │  将 prompt 分词为 origin_input_ids
└─────┬───────┘
      │
      ▼
┌─────────────────────────────┐
│  Request Init (ReqDllmMixin)│  初始化 dllm_ids, fill_ids, dllm_phase
│  req.py:20-31               │  短 prompt → INCOMING_DECODE
└─────┬───────────────────────┘  长 prompt → INCOMING_PREFILL
      │
      ▼
┌─────────────────────────────┐
│  DllmManager.waiting_queue  │  受 max_running_requests 限制
│  scheduler.py:250-322       │
└─────┬───────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Scheduler: get_new_batch_dllm()        │
│  scheduler.py:28-59                     │
│  1. decode 请求优先 (STAGING_DECODE)     │
│  2. 然后 prefill 请求 (STAGING_PREFILL)  │
│  3. ForwardMode = DLLM_EXTEND           │
└─────┬───────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  TP Worker: dllm_algorithm.run()     │
│  tp_worker.py:413-424                │
│  调用 LowConfidence 或 FDFO 算法      │
└─────┬────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────┐
│  Output Processor                             │
│  - LowConfidence: process_batch_result_dllm   │
│  - FDFO: process_batch_result_dllm_fdfo       │
│  scheduler_output_processor_mixin.py:377-483  │
└─────┬────────────────────────────────────────┘
      │
      ▼
  返回 HTTP Response (或继续下一个 block)
```

---

## 3. 核心数据结构

### 3.1 Request 状态 (`dllm/mixin/req.py`)

```python
class ReqDllmMixin:
    dllm_phase: DllmReqPhase     # 当前阶段
    dllm_ids: List[int]          # 完整的 padded input (origin + mask padding)
    dllm_incomplete_ids: List[int] # FDFO 模式下未完成的 block tokens
    dllm_block_offset: int       # 当前处理到第几个 block
    dllm_config: DllmConfig      # 算法配置
    fill_ids: List[int]          # 当前要喂给模型的 token 序列
```

### 3.2 阶段枚举

```python
class DllmReqPhase(str, enum.Enum):
    INCOMING_PREFILL  = "incoming_prefill"   # 新请求，prompt 较长，需要 prefill
    INCOMING_DECODE   = "incoming_decode"    # 新请求，prompt 较短，直接 decode
    STAGING_PREFILL   = "staging_prefill"    # 已分配资源，prefill 中
    STAGING_DECODE    = "staging_decode"     # 已分配资源，decode（去噪）中
```

### 3.3 fill_ids 构建逻辑 (`req.py:58-75`)

```
初始化:
  dllm_ids = origin_input_ids + [mask_id] * padding   // padding 到 block_size 的整数倍
  fill_ids = dllm_ids[:block_size]                     // 取第一个 block

后续 block 推进 (非 FDFO):
  dllm_block_offset += block_size
  if fill_len < dllm_len:
      fill_ids += dllm_ids[fill_len : fill_len+block_size]   // prefill 阶段: 取下一块已知 token
  else:
      fill_ids += [mask_id] * block_size                      // decode 阶段: 追加全 mask block

FDFO incomplete 恢复:
  fill_ids = fill_ids[:len_prefix] + dllm_incomplete_ids      // 用上次未完成的 token 继续
```

---

## 4. LowConfidence 算法

> 文件: `python/sglang/srt/dllm/algorithm/low_confidence.py`

### 4.1 核心思想

对 block 内的 mask 位置进行**多轮迭代去噪**：
1. 每轮 forward pass 获取所有位置的 logits
2. 对 mask 位置计算 argmax 预测和 softmax 置信度
3. 置信度 > threshold 的位置接受预测，替换 mask
4. 若没有任何位置超过 threshold，强制接受置信度最高的 1 个位置
5. 循环直到 block 内无 mask token，最多 block_size 次

### 4.2 算法流程（伪代码）

```
Input: forward_batch.input_ids = [block_size * batch_size] 包含 mask tokens
Parameters: threshold=0.95, block_size=32

# Fast path: 无 mask token → 单次 forward 缓存 KV
if no mask tokens:
    forward once → return

# 计算每个 block 的非 mask 起始位置
for each block:
    start_list[i] = block_size - count(mask_tokens)

# 迭代去噪循环 (最多 block_size 轮)
for step in range(block_size):
    if no mask tokens left: break

    logits = model.forward(forward_batch)        # [block_size * bs, vocab]

    for each block:
        if block has no mask: continue

        x = argmax(logits[block])                # 预测 token
        p = softmax(logits[block])               # 概率分布
        confidence = p[x]                        # 预测 token 的置信度

        # 只考虑 mask 位置的置信度
        confidence[non_mask] = -inf
        x[non_mask] = original_token

        # 阈值筛选
        transfer = (confidence > threshold)
        if transfer.sum() == 0:
            transfer[topk(confidence, k=1)] = True   # 至少接受 1 个

        input_ids[block][transfer] = x[transfer]     # 替换 mask

# 最终 forward 缓存 KV cache
model.forward(forward_batch)

# 提取输出 (跳过 prefix 部分)
output[i] = input_ids[block_i][start_list[i]:]
```

### 4.3 关键特性

| 特性 | 说明 |
|---|---|
| **迭代次数** | 最多 block_size 次，但通常更少（高置信 token 很快被接受） |
| **确定性** | 使用 argmax 而非采样，输出确定性（无 temperature/top_p） |
| **变长输出** | 每个 request 输出长度不同（取决于 prefix 长度） |
| **Forward 次数** | N+1 次（N 轮迭代 + 1 次最终 KV cache） |
| **KV Cache** | 每轮 forward 都重新计算（未利用前一轮 KV cache） |

### 4.4 输出处理 (`scheduler_output_processor_mixin.py:445-483`)

```python
def process_batch_result_dllm(batch, result):
    for idx in range(batch_size):
        if not result.next_token_ids:  # prefill 阶段无输出
            break
        req = batch.reqs[idx]
        next_token_ids = result.next_token_ids[idx].tolist()
        for token in next_token_ids:
            req.output_ids.append(token)
            req.check_finished()       # 检查 EOS / max_tokens
            if req.finished():
                release_kv_cache(req)
                break
```

---

## 5. LowConfidenceFDFO 算法

> 文件: `python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py`

### 5.1 核心思想

**FDFO = First Decode First Out**。与 LowConfidence 的核心区别：

| | LowConfidence | LowConfidenceFDFO |
|---|---|---|
| **Forward 次数** | 多次（最多 block_size+1） | **1 次** |
| **去噪策略** | 迭代直到 block 无 mask | 单次置信度筛选 |
| **完成判定** | block 内全部去噪 → 输出 | accept_length == block_size → 输出 |
| **未完成处理** | 不存在（必须做完） | 存 incomplete_ids，下轮继续 |
| **KV Cache** | 最终一次性缓存 | 未完成时释放 KV，完成时保留 |
| **调度优势** | 无 | 完成的请求立即退出，腾出资源 |

### 5.2 算法流程（伪代码）

```
Input: forward_batch.input_ids = [block_size * batch_size]
Parameters: threshold=0.95, block_size=32

# 记录每个 block 的 mask 数量 (forward 之前)
mask_counts[i] = count(mask_tokens in block_i)

# 单次 forward
logits = model.forward(forward_batch)     # [block_size * bs, vocab]

# Batch 化的 token picking (全部在 GPU 上完成)
logits = logits.view(batch_size, block_size, vocab_size)
input_ids = input_ids.view(batch_size, block_size)

x = argmax(logits, dim=-1)                # [bs, block_size]
probs = softmax(logits, dim=-1)
confidence = gather(probs, x)             # [bs, block_size]
confidence[non_mask] = -inf               # 只考虑 mask 位置

# 阈值筛选 (per-request)
transfer = (confidence > threshold)       # [bs, block_size] bool
has_transfer = transfer.sum(dim=1) > 0    # [bs] per-request flag

# Fallback: 没有任何 token 超阈值 → 选 top-1
top1_indices = topk(confidence, k=1, dim=1)
top1_mask = zeros_like(transfer)
top1_mask[batch_indices, top1_indices] = True
transfer = where(has_transfer, transfer, top1_mask)

# 更新 input_ids
x = where(mask, x, input_ids)             # 保留非 mask 位置原值
input_ids = where(transfer, x, input_ids) # 只替换被选中的位置

# 构建输出
for i in range(batch_size):
    next_token_ids[i] = input_ids[i]      # 整个 block 的 token
    if mask_counts[i] == 0:
        accept_length[i] = block_size     # 这个 block 在 forward 前已无 mask → 完成
    else:
        accept_length[i] = 0              # 还有 mask → 未完成
```

### 5.3 accept_length 的含义

- `accept_length == 0`: block 未完成（forward 前仍有 mask token）
  - 将 `next_token_ids` 存入 `req.dllm_incomplete_ids`
  - **释放对应的 KV cache**（因为 incomplete block 含 mask，KV cache 无效）
  - 下一轮 scheduler 会用 `dllm_incomplete_ids` 重新构建 `fill_ids`

- `accept_length == block_size`: block 完成
  - 清空 `dllm_incomplete_ids`
  - 将 decoded tokens 追加到 `req.output_ids`
  - 检查是否触发 EOS / max_tokens 完成条件

### 5.4 FDFO 输出处理 (`scheduler_output_processor_mixin.py:377-443`)

```python
def process_batch_result_dllm_fdfo(batch, result):
    for idx in range(batch_size):
        req = batch.reqs[idx]
        next_token_ids = result.next_token_ids[idx]

        if result.accept_length_per_req_cpu[idx] == 0:
            # 未完成: 存储 incomplete_ids，释放 KV cache
            req.dllm_incomplete_ids = next_token_ids
            kv_indices = req_to_token_pool[req, old_prefix:new_fill]
            free(kv_indices)                    # 释放含 mask 的 KV cache
            continue

        # 完成: 清空 incomplete，输出 tokens
        req.dllm_incomplete_ids = []
        for token in next_token_ids:
            req.output_ids.append(token)
            req.check_finished()
            if req.finished():
                release_kv_cache(req)
                break
```

### 5.5 为什么 FDFO 需要释放 KV Cache？

FDFO 单次 forward 后，如果 block 仍含 mask token（accept_length=0），那么：
1. 这次 forward 生成的 KV cache 对应的 input 包含 mask_id
2. 下一轮会用 `dllm_incomplete_ids`（mask 已被部分替换）重新构建 input
3. 之前的 KV cache 与新 input 不匹配，必须释放

---

## 6. Scheduler 调度机制

> 文件: `python/sglang/srt/dllm/mixin/scheduler.py`

### 6.1 DllmManager 双队列

```
                    max_running_requests 限制
                           │
waiting_queue ─────────────┼───► DllmManager.waiting_queue
(全局等待队列)              │    (dLLM 内部等待队列)
                           │
                           └───► DllmManager.staging_queue
                                (已分配资源，等待 forward)
```

### 6.2 调度优先级

```python
def _process_dllm_batches(adder):
    # 1. Decode 请求优先（正在去噪的 block）
    process(dllm_manager.get_decode_requests())     # STAGING_DECODE + INCOMING_DECODE

    # 2. 然后 Prefill 请求（prefill 阶段的 block）
    process(dllm_manager.get_prefill_requests())    # STAGING_PREFILL + INCOMING_PREFILL
```

**Decode 优先的原因**: 正在去噪的 block 需要尽快完成迭代，释放资源给后续请求。

### 6.3 每个阶段内的细分

```
STAGING 请求: 已有资源分配，调用 add_dllm_staging_req()
              ↓
INCOMING 请求: 新请求，需要分配资源，调用 add_one_req()
              可能触发 preemption
```

### 6.4 dLLM 模式下的约束 (server_args.py)

启用 dLLM 时自动设置：
- `disable_overlap_schedule = True` — 禁用调度重叠
- `disable_radix_cache = True` — 禁用 radix tree cache
- `pp_size = 1` — 不支持 pipeline parallelism
- `enable_mixed_chunk = False` — 禁用 mixed chunk
- `max_running_requests` 默认为 1（可配置更高）
- `attention_backend = "flashinfer"` — 必须使用 flashinfer

---

## 7. 完整执行示例

### 场景: prompt=100 tokens, max_output=64, block_size=32, LowConfidence

```
Step 1: Request Init
  origin_input_ids = [t0, t1, ..., t99]          // 100 tokens
  padding = (-100) % 32 = 28
  dllm_ids = [t0..t99] + [MASK]*28               // 128 tokens (4 blocks)
  fill_ids = [t0..t31]                            // Block 0 (全是真实 token)
  dllm_phase = INCOMING_PREFILL

Step 2: Block 0 Prefill (无 mask → fast path)
  forward once → cache KV for [t0..t31]
  output: empty (prefill only)
  → advance: fill_ids = [t0..t63]

Step 3: Block 1 Prefill (无 mask → fast path)
  forward once → cache KV for [t32..t63]
  output: empty
  → advance: fill_ids = [t0..t95]

Step 4: Block 2 Prefill (无 mask → fast path)
  forward once → cache KV for [t64..t95]
  output: empty
  → advance: fill_ids = [t0..t99, MASK*28]

Step 5: Block 3 Decode (有 28 个 mask)
  dllm_phase = STAGING_DECODE
  迭代去噪:
    Iter 1: confidence > 0.95 的 10 个位置被接受
    Iter 2: 又有 8 个位置被接受
    Iter 3: 又有 6 个位置被接受
    Iter 4: 剩余 4 个全部接受 (其中 1 个靠 top-1 fallback)
    Final forward: cache KV
  output: 28 tokens (跳过 start=4 的前缀部分)

Step 6: Block 4 Decode (全 mask)
  fill_ids = [..., MASK*32]
  迭代去噪 → output: 32 tokens

Step 7: 检查 finished (已生成 28+32=60 tokens, 达到 max_output=64 附近)
  继续或结束取决于 EOS / max_tokens
```

### 场景: 同样配置, LowConfidenceFDFO

```
Step 5: Block 3 Decode (有 28 个 mask)
  Round 1: forward → pick tokens → mask_count=28 → accept_length=0
    存 incomplete_ids, 释放 KV cache
  Round 2: forward(incomplete_ids) → pick more → mask_count 减少 → accept_length=0
    存 incomplete_ids, 释放 KV cache
  ...
  Round N: forward → mask_count=0 → accept_length=32 → 输出完整 block
```

FDFO 关键区别: **每轮只做 1 次 forward**，但需要多轮 scheduler 调度。
其优势在多请求并发时体现——已完成的请求立即退出，不阻塞其他请求。

---

## 8. 性能对比分析

### LowConfidence
- **延迟**: 单请求延迟较低（一次性完成 block 内所有迭代）
- **吞吐**: 较差（block 迭代期间独占 GPU，其他请求等待）
- **Forward 次数**: N+1（N 为迭代次数，通常 < block_size）
- **KV Cache**: 最终统一缓存
- **适用场景**: 低并发、追求单请求延迟

### LowConfidenceFDFO
- **延迟**: 单请求延迟可能更高（需要多轮 scheduler 调度）
- **吞吐**: 较好（完成的请求立即退出，释放资源）
- **Forward 次数**: 每轮 1 次，多轮执行
- **KV Cache**: 每轮未完成都释放，开销更大
- **适用场景**: 高并发、追求系统吞吐

---

## 9. 关键文件索引

| 组件 | 文件 | 行号 | 说明 |
|---|---|---|---|
| Config | `dllm/config.py` | 1-82 | DllmConfig, 模型参数表 |
| Base Algorithm | `dllm/algorithm/base.py` | - | 算法基类 |
| LowConfidence | `dllm/algorithm/low_confidence.py` | 14-104 | 迭代去噪算法 |
| LowConfidenceFDFO | `dllm/algorithm/low_confidence_fdfo.py` | 12-119 | 单次 forward + FDFO |
| Algorithm Registry | `dllm/algorithm/__init__.py` | - | 算法注册与加载 |
| Request Mixin | `dllm/mixin/req.py` | 19-75 | 请求 dLLM 状态管理 |
| Scheduler Mixin | `dllm/mixin/scheduler.py` | 19-322 | DllmManager, 调度逻辑 |
| TP Worker | `managers/tp_worker.py` | 413-424 | 调用 algorithm.run() |
| Output (dLLM) | `managers/scheduler_output_processor_mixin.py` | 445-483 | LowConfidence 输出处理 |
| Output (FDFO) | `managers/scheduler_output_processor_mixin.py` | 377-443 | FDFO 输出处理 + KV 释放 |
| Batch Prep | `managers/schedule_batch.py` | 1466-1542 | DLLM_EXTEND forward mode |
| Schedule Policy | `managers/schedule_policy.py` | 563-589 | dLLM token budget 管理 |
| Server Args | `srt/server_args.py` | 2797-2850 | dLLM 约束自动设置 |

---

## 10. Code Review

以下 review 基于当前仓库实现逐段对照，重点指出文档中和代码不一致、会误导后续理解的地方。

### Finding 1: FDFO 的 `accept_length` 语义写错了

文档当前描述：
- “`accept_length == block_size` 表示这个 block 在本轮完成”
- “Round N: forward -> mask_count=0 -> accept_length=32 -> 输出完整 block”

这和代码不一致。

实际实现见 `python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py`：

```python
# 先统计 _pick_tokens 之前的 mask 数
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

也就是说：
- `accept_length` 不是“本轮 pick 完后是否无 mask”
- 而是“进入这一轮 forward 之前，这个 block 是否已经无 mask”

直接后果：
- 如果某个 block 在这一轮 `_pick_tokens()` 里刚好把最后几个 mask 全部补齐，`accept_length` 仍然是 `0`
- 这个 block 不会在本轮输出
- 要等下一轮 scheduler 再跑一次，此时输入已经没有 mask，`accept_length` 才会变成 `block_size`

因此，前文 5.2 / 5.3 / 7 节里关于 FDFO “本轮补齐即完成输出”的描述都偏乐观，和实际代码执行时机不一致。

### Finding 2: FDFO 的输出逻辑遗漏了 prefill/decode 边界裁剪

文档当前把 FDFO 完成态近似为：

```python
req.dllm_incomplete_ids = []
for token in next_token_ids:
    req.output_ids.append(token)
```

这不够准确。

实际输出处理见 `python/sglang/srt/managers/scheduler_output_processor_mixin.py`：

```python
len_input = len(req.origin_input_ids)
len_fill = len(req.fill_ids)

if len_fill < len_input:
    continue

if len_fill - len_cur_tokens < len_input:
    next_token_ids = next_token_ids[len_input - len_fill: ]
```

这说明完成态还有两层分支：

1. 如果 `len_fill < len_input`
   - 当前仍处于 prefill 覆盖原始 prompt 的阶段
   - 即使 `accept_length == block_size`，也不会把这一整块输出到 `output_ids`

2. 如果当前 block 横跨 prompt 尾部和 decode 新 token 区域
   - 只会输出真正位于 decode 区域的后缀
   - 不会把 prefill 部分也追加到 `output_ids`

所以文档里“FDFO 完成就输出整个 block”的说法不成立，至少在边界 block 上是错的。

### Finding 3: `dllm_ids` / `fill_ids` 的初始化时机写早了

第 2 节生命周期图里写的是：
- `Request Init (ReqDllmMixin)` 初始化 `dllm_ids`, `fill_ids`, `dllm_phase`

但代码不是这样。

`python/sglang/srt/dllm/mixin/req.py` 里的 `init_diffusion_llm()` 只做了：
- 清空 `dllm_ids`
- 清空 `dllm_incomplete_ids`
- 设置 `dllm_block_offset`
- 根据 prompt 长度设置初始 `dllm_phase`

真正构造 `dllm_ids` / `fill_ids` 的地方是后续的 `init_next_round_input()`：

```python
if self.is_dllm():
    self._init_fill_ids_for_dllm()
    self.determine_dllm_phase()
```

因此更准确的表述应当是：
- request 初始化阶段只建立 dLLM 元数据和初始 phase
- 第一次进入调度轮次时，才真正派生出 `dllm_ids` 与 `fill_ids`

### Finding 4: “必须使用 flashinfer” 这个约束写得过强

第 6.4 节写的是：

- `attention_backend = "flashinfer"` — 必须使用 flashinfer

这和代码不一致。

`python/sglang/srt/server_args.py` 的实际逻辑是：

- 在 NVIDIA/CUDA 且没有禁用 cuda graph 时，会把 backend 切到 `flashinfer`
- 但在 AMD/HIP 上，会关闭 cuda graph，并把 attention backend 调整到 `triton`（或保留 `aiter`）

也就是说更准确的描述应为：

- CUDA 路径为了 dLLM + cuda graph，会强制使用 `flashinfer`
- HIP 路径不是 `flashinfer` 必选，反而会切到 `triton` / `aiter`

所以“必须使用 flashinfer”不适合作为通用结论。

### Minor Note: `DllmManager.waiting_queue` 的容量限制是在 Scheduler 里施加的

第 6.1 节提到 `DllmManager.waiting_queue` 受 `max_running_requests` 限制，这个方向基本对，但实现位置需要更精确。

真正的限制代码在 `SchedulerDllmMixin._fetch_waiting_reqs()`：

```python
max_dllm_capacity = self.server_args.max_running_requests - len(
    self.dllm_manager.waiting_queue
)
num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))
```

也就是说：
- 不是 `DllmManager` 自己在内部 enforcing 这个限制
- 而是 scheduler 在把全局 `waiting_queue` 搬运到 `dllm_manager.waiting_queue` 时施加上限

如果后续想把文档写得更严谨，建议把这层责任关系写清楚。

### Summary

这份文档对 `LowConfidence` 主流程的理解整体是对的，但 `LowConfidenceFDFO` 这部分有两个关键误差：

1. 把 `accept_length` 理解成了“本轮去噪完成判定”
2. 把完成态输出理解成了“整块直接进入 `output_ids`”

这两个点都会直接影响读者对 FDFO 调度节奏、输出时机和 KV 释放必要性的理解，建议优先修正文档对应章节。
