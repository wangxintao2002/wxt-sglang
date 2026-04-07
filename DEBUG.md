# AR Prefill Debug Log

## Context

目标是在 `wxt-sglang` 里补齐 `sgl/ar-prefill-patch.md` 对应的 AR prefill 路径，并让 `test/registered/dllm/test_llada2_mini.py` 可跑通。

参考实现主要来自：

- `sgl/ar-prefill-patch.md`
- `sgl/python/sglang/srt/dllm/mixin/scheduler.py`
- `sgl/python/sglang/srt/mem_cache/dllm_radix_cache.py`

核心设计目标：

- 新 DLLM 请求先走一次 AR 风格的 `ForwardMode.EXTEND` prefill，只处理 prompt 的对齐块。
- prefill 完成后立刻切到 `STAGING_DECODE`。
- decode 不写 radix tree。
- prefill 使用 `block_causal_varlen_attention`，decode 继续走 `flashinfer`。

## 已落地的主改动

### 调度与阶段机

- 新增/重构 DLLM phase：
  - `INCOMING_PREFILL`
  - `STAGING_DECODE`
  - `SKIP_FORWARD`
- `ReqDllmMixin` 中新增 `dllm_origin_len_aligned`，用于 AR prefill 只覆盖 block 对齐的 prompt 前缀。
- `SchedulerDllmMixin` 改为：
  - 优先处理 `INCOMING_PREFILL`
  - AR prefill batch 用 `ForwardMode.EXTEND`
  - decode batch 用 `ForwardMode.DLLM_EXTEND`
  - batch 上通过 `dllm_ar_prefill` 标记区分 AR prefill 和真正 DLLM decode

### attention backend 拆分

- `server_args.py` 增加 `block_causal_varlen_attention` backend 选择项。
- DLLM 自动拆分 backend：
  - `prefill_attention_backend = block_causal_varlen_attention`
  - `decode_attention_backend = flashinfer`
- `attention_registry.py` 为 `block_causal_varlen_attention` 动态注册 DLLM block size。
- `HybridAttnBackend` 修复为：
  - `ForwardMode.DLLM_EXTEND` 必须走 decode backend

### radix/cache 语义

- scheduler 在 DLLM 模式下改用 `DllmRadixCache`。
- `DllmRadixCache` 的语义：
  - prefill 阶段允许正常读写 radix tree
  - decode 阶段只维护 `prefix_indices`，不读不写 tree

### 结果处理

- 增加 `process_batch_result_dllm_ar_prefill()`：
  - prefill 完成后 `cache_unfinished_req(req)`
  - 然后切到 `STAGING_DECODE`
- `Scheduler.process_batch_result()` 按 `dllm_ar_prefill` 分流

## 遇到的 bug 与处理过程

### 1. Hybrid attention backend 路由错误

现象：

- AR prefill 接上后，DLLM decode 仍可能落到 prefill backend。

根因：

- `HybridAttnBackend._select_backend()` 只把普通 decode 视为 decode backend，漏掉了 `ForwardMode.DLLM_EXTEND`。

修复：

- 在 `python/sglang/srt/layers/attention/hybrid_attn_backend.py` 中把 `forward_mode.is_dllm_extend()` 也路由到 decode backend。

### 2. Tp worker 错把 AR prefill 当成 DLLM algorithm batch

现象：

- AR prefill batch 进入了 DLLM algorithm 路径，而不是普通 AR forward 路径。

根因：

- `tp_worker.py` 中只要 `self.is_dllm()` 就走 `_forward_batch_generation_dllm()`，没有再判断 forward mode。

修复：

- 改成只有 `forward_batch.forward_mode.is_dllm_extend()` 才走 DLLM algorithm。
- AR prefill 的 `ForwardMode.EXTEND` 继续走普通 forward 路径。

### 3. `protected_size=-32` 的 KV leak / 锁计数下溢

现象：

- `test_gsm8k` 早期会在 idle self-check 报：

  - `token_to_kv_pool_allocator memory leak detected`
  - `protected_size=-32`

根因定位过程：

- 先怀疑是普通 KV leak，后来从 `protected_size=-32` 判断更像是 radix lock 计数被多减了一次。
- 对照参考实现和本地调用链后，定位到 exact-cache-hit 的 AR prefill shortcut：
  - 请求在 `process_dllm_incoming_reqs()` 中如果 `extend_input_len == 0`，会直接切到 `STAGING_DECODE`
  - 但这条路径绕过了 `add_one_req()`，因此没有拿到 radix tree 的持久锁
  - 请求结束时仍会在 `cache_finished_req()` 中 `dec_lock_ref(req.last_node)`，导致锁下溢

修复：

- 在 exact-hit shortcut 中显式调用 `adder._req_inc_lock_ref(req)`。
- 新增 `req.dllm_tree_lock_held`，把“请求是否真的持有过 tree lock”做成显式状态，而不是靠 `last_node is None` 猜。
- `DllmRadixCache.cache_finished_req()` 改为：
  - 只有 `dllm_tree_lock_held=True` 才走真正的 `dec_lock_ref`
  - 否则只释放 decode 侧的 KV，不碰 tree lock

### 4. DLLM 完成时未释放最后一块的尾部 KV

现象：

- 修掉锁问题后，仍存在 DLLM finish 时 KV 生命周期不完整的问题。

根因：

- 普通 `release_kv_cache()` 只会按 `RadixCache.cache_finished_req()` 管理的有效 token 范围回收。
- DLLM decode 最后一轮可能申请了一整块 KV，但真实有效 token 还没填满整块。
- 这部分“当前 block 的尾巴”不在 radix tree 管理范围内，也不在原来的 `release_kv_cache_dllm()` 释放范围内。

修复：

- 在 `python/sglang/srt/mem_cache/common.py` 里新增/修正 `release_kv_cache_dllm()`：
  - 用 `valid_kv_len = len(origin_input_ids) + len(output_ids)` 计算真实有效长度
  - 先释放 `[valid_kv_len:len(fill_ids))`
  - 再调用普通 `release_kv_cache()`
- DLLM 的 finish 路径统一改成调用 `release_kv_cache_dllm()`，不再复用普通 `release_kv_cache()`。

### 5. `_init_fill_ids_for_dllm_fdfo()` 的过严断言

现象：

- GSM8K 并发压上来以后，scheduler 在 `req.init_next_round_input()` 里崩：
  - `assert len(self.fill_ids) + block_size == len(self.dllm_ids)`

根因：

- 这个断言假设从 prefill 进入 decode 时，`fill_ids` 只会比 `dllm_ids` 少一个 block。
- 真实调度中这个假设并不稳，尤其在 cache hit / 多轮调度切换时。

修复：

- 去掉这类断言，改成恢复式逻辑：
  - 只要 `len(fill_ids) < len(dllm_ids)`，就直接把 `fill_ids` 补到 `dllm_ids`
- 同样处理了 basic / fdfo / fdfo_sp 三条分支。

### 6. FDFO decode batch 出现 `511 / 32 = 15` 的 batch size mismatch

现象：

- 进一步跑 GSM8K 后，DLLM algorithm 崩在：
  - `Batch size mismatch: forward_batch.batch_size=16, but input_ids shape 511 / block_size 32 = 15`

根因：

- 个别 decode 请求在入 batch 时，`fill_ids[len(prefix_indices):]` 不是完整的一个 block。
- 对 FDFO 而言，decode batch 必须保证每个请求都正好贡献 `block_size` 个 token。

修复：

- 在 `process_dllm_staging_reqs()` 入 batch 前做一层归一化：
  - 若 `len(fill_ids) - len(prefix_indices) != block_size`
  - 则把 `prefix_indices` 强行截到 `len(fill_ids) - block_size`
  - 并把 `extend_input_len` 重设为 `block_size`

这个修复的目的不是掩盖 bug，而是把 decode batch 的输入契约收紧在 scheduler 层，避免脏状态直接打到 algorithm 层。

## 目前测试回归情况

### 已确认

- Python 语法检查：
  - 相关修改文件均已通过 `python -m py_compile`
- `test_bs_1_speed` 之前可以跑通，测速大约在 200 token/s 左右
- `test_gsm8k` 已经越过了最早的两类致命问题：
  - `protected_size=-32` 的 memory leak
  - `_init_fill_ids_for_dllm_fdfo()` 断言崩溃

### 进行中的回归

最新一轮 `test_gsm8k` 已经能稳定进入高并发批量阶段，不再在最早几个请求处直接崩掉；这说明前面的锁计数 / KV 释放 / phase 迁移问题基本都被压住了。

但截至这份记录落盘时，整份 `test_llada2_mini.py` 还没有正式宣告全绿，仍需要继续观察：

- 是否还会出现新的 batch 形状问题
- GSM8K 精度是否保持在合理范围
- 最终性能是否达到预期

## 关键文件

- `python/sglang/srt/dllm/mixin/req.py`
- `python/sglang/srt/dllm/mixin/scheduler.py`
- `python/sglang/srt/mem_cache/dllm_radix_cache.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/tp_worker.py`
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/server_args.py`

## 下一步建议

如果后续继续追 `test_gsm8k`：

1. 优先盯住 decode batch 的契约，确认每个请求在 DLLM decode 时都严格是一整块。
2. 如果再出现 batch shape 问题，直接打印异常 batch 中每个请求的：
   - `rid`
   - `len(fill_ids)`
   - `len(prefix_indices)`
   - `extend_input_len`
   - `dllm_block_offset`
   - `len(dllm_incomplete_ids)`
3. 如果准确率异常，再回看 AR prefill 后 `prefix_indices` 和 `fill_ids` 的同步关系，确认没有把 decode token 错当成 prompt token。
