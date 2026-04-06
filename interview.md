# FDFO 调度分析

## 为什么 `mix decode and prefill` 能让 `LowConfidenceFDFO` 的吞吐提升好几倍

关键点在于，这个 commit 并没有让模型本身的 forward 更快。它真正改变的是 scheduler 的行为：

- 旧行为：只要有任何 prefill request，就只调度 prefill；只有 prefill 队列空了才会跑 decode
- 新行为：先调度 decode，再在同一轮里用剩余 budget 调度 prefill

相关改动在 `python/sglang/srt/dllm/mixin/scheduler.py`。

旧逻辑：

```python
prefill_reqs = self.dllm_manager.get_prefill_requests()
if prefill_reqs:
    self._process_batch_by_phase(... prefill ...)
else:
    self._process_batch_by_phase(... decode ...)
```

新逻辑：

```python
self._process_batch_by_phase(... decode ...)
self._process_batch_by_phase(... prefill ...)
```

这件事对 `LowConfidenceFDFO` 特别重要，因为 FDFO 对 scheduler round 非常敏感。

### 1. FDFO 的一个 block 需要跨多个 scheduler round 才能完成

`LowConfidenceFDFO` 每一轮只做一次 forward，然后把未完成的 block 放进 `req.dllm_incomplete_ids`，等待下一轮调度。

所以，一个 block 的完成不是一次性发生的，而是分散在多个 round 里：

1. 跑一次 forward
2. 挑出一部分 token
3. 保留 incomplete block
4. 再次被调度
5. 重复

这意味着 decode 的推进速度，非常依赖 decode request 能不能尽快拿到下一次执行机会。

### 2. 旧 scheduler 在持续 prefill 压力下会让 decode 饥饿

在旧版本里，只要 `dllm_manager.waiting_queue` 里还有任何 prefill request，这一轮 batch 就会变成纯 prefill。

高并发下，系统里几乎总会有请求还停留在 prefill 阶段，于是会进入一种坏状态：

- 老请求已经进入 decode / incomplete 阶段
- 新请求还在不断消耗 scheduler round 去做 prefill
- decode request 越等越久
- GPU 虽然很忙，但很多工作都只是“不直接产出 output token”的 prefill

所以系统确实在做事，但这些工作并没有有效转化成 output throughput。

### 3. FDFO 即使补完最后一个 mask，也还会再延后一轮输出

当前 `LowConfidenceFDFO` 的实现里，`accept_length_per_req_cpu` 是基于 `_pick_tokens()` 之前的 mask count 计算的，而不是基于之后的状态。

这意味着：

- 如果一个 block 在本轮刚好把最后的 mask 补完
- 它在这一轮里仍然不会被视为 complete
- 它还需要再经历一个“进入时已经没有 mask”的 scheduler round

这让 FDFO 对 scheduler 延迟更加敏感。如果 decode 被 prefill 压住，那么最后那个真正产生输出的 round 也会被一起拖后。

所以旧 scheduler 对 FDFO 的伤害有两层：

- 它会拖慢中间那些 decode refinement round
- 它也会拖慢最后那个真正让 block 变成 output-visible 的 round

### 4. 新 scheduler 允许 decode 和 prefill 共享同一个 dLLM budget

dLLM 模式下，`PrefillAdder` 用的是统一的 block 级 token budget：

```python
self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size
```

prefill block 和 decode block 消耗的是同一套 dLLM block budget，所以它们天然可以混在同一个 batch 里。

新 scheduler 正是利用了这一点：

- 先用 decode block 填 batch，因为它们离真正产出 output 更近
- 再用剩余 budget 去补 prefill block

这样会提升有效吞吐，因为同一个 batch 现在同时做了两件有价值的事：

- 让接近完成的请求继续向 output 推进
- 同时让新请求继续推进 prefill

### 5. 为什么收益会是“好几倍”，而不是边际优化

旧 scheduler 很容易进入一种病态状态：

- prefill 队列始终不空
- decode 持续被推迟
- 大量 FDFO request 长时间卡在 incomplete 状态
- 观测到的 output throughput 主要被等待时间支配，而不是被计算速度支配

一旦 decode starvation 被消除，很多之前“离完成只差一点”的请求会更早开始真正产出 token。即使 kernel 本身并没有变快，最终测到的 output throughput 也可能大幅上涨。

所以这个 commit 的本质是一次 scheduler 公平性和 batch 组装策略的修复，而不是模型执行路径的优化。

## 为什么同样的改动对 basic 版本几乎没有收益

简短结论是：这个 commit 修复的是 FDFO 特有的瓶颈，而 basic 版本的主要瓶颈在别处。

### 1. FDFO 依赖反复 re-scheduling，basic 不依赖

`LowConfidenceFDFO` 是把一个 block 的完成过程拆散到多个 scheduler round 里。

而 dLLM 的 basic 版本，例如 `LowConfidence`，是在一次 `run()` 调用里把 denoising loop 全部做完：

- 多轮 denoise iteration 都发生在同一次 worker 执行里
- 然后再做一次 final forward
- block 在这些内部步骤之间不会反复返回 scheduler 重新竞争下一次执行机会

所以 scheduler policy 对 FDFO 的影响要远大于对 basic 的影响。

### 2. 这个 commit 修的是 decode starvation，而 basic 的 decode 没那么脆弱

对 FDFO 来说，decode 是细粒度的，而且会反复回到队列里。如果 scheduler 一直优先挑 prefill，那么 decode 的推进就会停住。

而 basic 版本里，一个 block 一旦进入 `LowConfidence.run()`，它的内部去噪过程就在这一次调用里继续推进。它不需要在每一步 denoise 之间都重新依赖 scheduler。

因此：

- FDFO 在 decode 被饿死时会受很大影响
- basic 对这种 starvation 模式不那么敏感

### 3. FDFO 的吞吐更容易被 prefill-heavy 调度稀释

在 FDFO 里，很多工作发生时并不会立刻变成 output token：

- 部分 unmask
- incomplete block 的保留与续跑
- 等待下一轮才真正变成可输出状态

所以如果 scheduler 过度偏向 prefill，FDFO 的 measured output throughput 会掉得很快。

而在 basic 版本里，一个 block 一旦被选中，`run()` 内的大多数工作都会直接服务于这个 block 的完成。它的跨轮调度开销更少，也更不容易因为排队而延迟 output。

### 4. Basic 更像是单个 batch 内部的 compute-bound

对 `LowConfidence` 来说，主要成本在内部循环：

- 多次 model forward
- confidence 计算
- final forward

这些成本主要发生在 batch 内部，是典型的 intra-batch compute cost。`mix decode and prefill` 这个 commit 并不会减少这些 forward 次数，也不会让这些 kernel 变便宜，所以它的收益上限天然比较低。

换句话说：

- FDFO 部分是 scheduler-bound
- basic 更偏 batch-compute-bound

这就是为什么同样的调度修复，对 FDFO 可以是质变级收益，而对 basic 几乎看不见。

## 最终总结

`mix decode and prefill` 能帮助 `LowConfidenceFDFO`，是因为它去掉了原本“prefill-first”的调度策略，而这个策略会让 decode request 饥饿。FDFO 对这件事尤其敏感，因为一个 block 的完成需要多个 scheduler round，而且即使 block 已经 fully unmasked，最终 output 还可能再多延后一轮。

basic 版本收益不大，是因为它的 denoising 是在单次 worker 调用内部完成的。它的主要瓶颈是内部计算，而不是跨多个 round 的反复 re-scheduling。所以修复 decode starvation 对它的影响小得多。

## 这个优化点大概率是如何被发现的

这个优化点大概率不是先从理论推导出来的，更像是沿着一条性能调试链路被定位出来的：

1. 先观察到 `LowConfidenceFDFO` 的吞吐低于预期
2. 用 `nsys` 做 profile
3. 发现单轮 FDFO 执行并不够慢，不能解释吞吐差距
4. 转而检查 scheduler 的行为
5. 最终定位到 prefill-first 调度导致的 decode starvation

### 1. 最初的信号大概率是吞吐异常

仓库里有多处迹象表明，FDFO 吞吐当时是被重点关注的：

- `test/registered/dllm/test_llada2_mini.py` 对 `LowConfidenceFDFO` 有明确的 output throughput 门槛
- `research-fdfo.md` 里保留了详细的 `nsys` profile 数据
- 当前分支里还有 `nsys_reports/` 和 `scripts/profile_dllm.sh`

这些都强烈说明，事情的起点是一个经验层面的性能问题：

- 理论上 FDFO 应该让调度更灵活
- 但实际吞吐并没有达到预期

于是团队开始做 profile。

### 2. Profiling 大概率先排除了“model forward 太慢”这个方向

根据 `research-fdfo.md` 里的数字：

- `FDFO::run` 平均大约 `7.73ms`
- `FDFO::model_forward` 平均大约 `2.00ms`
- 单轮整体执行时间并不长

这很重要，因为它会改变调试方向。

如果单轮 FDFO 已经足够重，那下一步自然应该去做 kernel 优化。但数据给出的信号刚好相反：

- 单轮成本比较低
- worker 在每次 `run()` 里并不会被阻塞很久
- 真正的端到端吞吐问题，应该出在这些大量小 round 是如何被调度的

所以分析方向自然会从模型执行路径，转移到 scheduler 行为。

### 3. 下一条线索是 FDFO 天生依赖 repeated re-scheduling

FDFO 并不会在一次 scheduler 级别的步骤里完成一个 block。它的流程更像是反复做下面几件事：

- 跑一次 forward
- 部分填充 token
- 保存 incomplete 状态
- 等待下一轮调度

这意味着 FDFO 的性能不只取决于单轮速度，也取决于 decode request 能多快拿到下一次执行机会。

一旦想清楚这一点，接下来最关键的问题就变成了：

- decode request 到底有没有被足够快地重新调度？

### 4. Scheduler 指标大概率暴露出很多 loop 并没有转化成有效 decode 进展

`research-fdfo.md` 记录过，`get_next_batch` 的调用次数远高于真正执行的 FDFO batch 数。即使这不是完全一一对应的归因，它仍然是一个很强的信号：

- scheduler 很活跃
- 但这些活跃度并没有高效地转化成 output-producing work

再结合 FDFO 的语义，这很容易指向一个系统级问题：

- decode round 可能等待太久
- 很多 request 可能长时间卡在 incomplete 状态
- scheduler 的精力可能更多花在 prefill-heavy 的 round 上，而不是推进那些接近完成的 decode block

### 5. 一看代码就会暴露真正的瓶颈：prefill-first 的排他策略

旧版 dLLM scheduler 的逻辑是：

```python
prefill_reqs = self.dllm_manager.get_prefill_requests()
if prefill_reqs:
    self._process_batch_by_phase(... prefill ...)
else:
    self._process_batch_by_phase(... decode ...)
```

它的含义很直接：

- 只要还有任何 prefill request
- 这一轮就完全不会调度 decode request

高并发下，prefill request 几乎总是存在。于是就形成了一个稳定的 starvation 模式：

- 老请求已经进入 decode / incomplete 阶段
- 新请求还在不断进入 prefill
- scheduler 持续优先挑 prefill
- decode 一直等不到机会

到这一步，根因基本就清楚了：问题不在于单个 FDFO round 本身有多慢，而在于 scheduler policy 系统性地拖慢了 decode round。

### 6. 为什么 mixed scheduling 修复会成为最自然的下一步实验

一旦确认是 decode starvation，最小的修复思路其实很直接：

- 先让 decode 消耗 budget
- 再用剩余的 dLLM block budget 去跑 prefill

这之所以自然，是因为 `PrefillAdder` 本来就已经使用了统一的 dLLM block budget：

```python
self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size
```

这意味着 prefill block 和 decode block 在 budget 层面本来就是兼容的。旧的分离只是一个策略选择，不是机制上的硬限制。

因此，`mix decode and prefill` 是一个风险最低的验证实验：

- 不需要重设计算法
- 不需要重写 kernel
- 只需要改 scheduler，让 decode 不再被 prefill 完全排斥

### 最终解释

所以，这个优化点最可能的发现路径是：

1. 先看到 FDFO 吞吐异常
2. profile worker 侧执行
3. 发现单轮成本不是核心问题
4. 进一步思考 FDFO 对 repeated scheduling 的依赖
5. 回看 scheduler policy
6. 发现 prefill-first 对 decode 的 starvation
7. 用 mixed decode + prefill scheduling 验证假设

简而言之，这个优化点大概率是通过 profiling 加 scheduler 语义分析一起定位出来的，而不是通过孤立的模型路径微优化得出的。

## 埋点结果：为什么 FDFO 在 `test_llada2_mini.py` 里明显强于 basic

我在 dLLM 的 scheduler 链路上加了 trace 点，然后在同一台机器上把同一套测试跑了两遍：

- `LowConfidenceFDFO`
- `LowConfidence`

这里 benchmark 的口径是：

```python
output_throughput = sum(completion_tokens) / latency
```

其中 `latency` 是整批 200 个 GSM8K request 跑完的 wall-clock time。

### 1. 端到端结果

实测结果：

- `LowConfidenceFDFO`: `output_throughput = 626.461 token/s`, `latency = 41.005 s`, `accuracy = 0.910`
- `LowConfidence`: `output_throughput = 165.571 token/s`, `latency = 150.491 s`, `accuracy = 0.915`

所以 FDFO 的 output throughput 大约是 basic 的 `3.78x`。

两边 accuracy 基本在同一量级，所以吞吐提升并不是靠明显的质量塌缩换来的。真正的差异在于分母，也就是整批请求的完成时间，大幅下降了。

### 2. 埋了哪些点

这次增加的 trace 会记录：

- 算法侧的 `run_ms`、`model_forward_ms`、`postprocess_ms`
- scheduler 侧的 `run_batch_ms`
- 每个 request 的 decode block 何时开始、何时完成
- 每个 block 完成所需的 rounds
- 不同 batch 难度下 fast block 的完成时延
- 每个 request 的 completion latency

这样就能把局部行为和最终吞吐指标连起来。

### 3. 核心指标 1：单次 batch 阻塞时间大幅下降

`run_batch_ms`：

- FDFO: `p50 = 26.82 ms`, `p95 = 27.50 ms`, `p99 = 27.75 ms`
- basic: `p50 = 267.69 ms`, `p95 = 760.73 ms`, `p99 = 844.25 ms`

这是最清晰的差异之一。

在 basic 里，一个 scheduler batch 会长时间卡在 `LowConfidence.run()` 内部，因为整个 denoising loop 都包含在一次调用里。而在 FDFO 里，一个 scheduler batch 只做一轮 forward 和一轮 post-process，然后就把控制权还给 scheduler。

所以 FDFO 在 p50 上大约把长阻塞压低了一个数量级，在 tail 上下降更多。

### 4. 核心指标 2：快 block 不再被慢 block 绑死

这是最重要的 FDFO 专属收益。

对于一轮就能完成的 block：

- FDFO 的 `block_completion_latency` 平均是 `56.2 ms`
- basic 的 `block_completion_latency` 平均是 `525.4 ms`

对于 `2-4` 轮完成的 block：

- FDFO 平均 `108.1 ms`
- basic 平均 `519.4 ms`

这说明，即使是容易完成的 block，在 basic 里仍然要等同一轮 `run()` 里最难的 block；而在 FDFO 里，它们可以早得多地完成。

所以真正的收益并不是“困难 block 消失了”，而是“困难 block 不再把同批次里的简单 block 一起拖慢”。

### 5. 核心指标 3：slow-block externality 大幅下降

我还额外看了一个指标：同一个 batch 里最慢的 block 会对 fast block 造成多大拖累。

当 batch 里存在非常慢的 block，也就是 `slowest_round_bucket = 5_plus` 时：

- FDFO 的 fast-block latency 平均是 `56.2 ms`
- basic 的 fast-block latency 平均是 `608.7 ms`

这就是下面这个直觉最直接的量化形式：

- 在 basic 里，低置信 block 会把高置信 block 一起拖走
- 在 FDFO 里，这种耦合被大幅削弱了

这是最能说明问题的一组证据：block-level early-exit，更准确地说是 block-level de-coupling，是吞吐提升的主要来源之一。

### 6. 核心指标 4：request completion 的 tail 明显改善

`request_completion_latency_ms`：

- FDFO: `p50 = 18.3 s`, `p95 = 23.9 s`, `p99 = 25.6 s`
- basic: `p50 = 55.4 s`, `p95 = 109.1 s`, `p99 = 110.9 s`

这非常关键，因为测试里的 throughput 指标用的是整批请求的 wall-clock time。整个 benchmark 的总时长，会被尾部慢请求强烈支配。

所以一旦 FDFO 同时缩短了长 batch 阻塞，又削弱了 slow-block drag，request tail 就会明显收缩，最终 benchmark latency 也就降下来了。

### 7. 核心指标 5：block 难度分布差不多，但系统行为好很多

两边面对的 block 难度本身，并没有明显更简单。

- FDFO 的 `rounds_to_completion`: 平均 `14.12`，`p95 = 25`
- basic 的非零 `block_iterations`: 平均 `14.18`，`p95 = 26`

也就是说，这两种算法面对的大致是同一类 denoising difficulty distribution。

这一点很重要，因为它说明吞吐提升并不是主要来自“FDFO 碰到的 workload 更简单”，而是来自系统如何处理同样的 workload：

- basic 把这些工作耦合在长 `run()` 里
- FDFO 把它拆成很多短 round，让已经完成的 block 更早退出长链路

### 8. 这些指标是如何一起构成吞吐提升的

整条因果链是这样的：

1. FDFO 把一次长 denoise 调用拆成很多短 scheduler round
2. 这会显著降低 `run_batch_ms`，尤其是尾部 batch time
3. 简单 block 不再一直等同 batch 里最难的 block
4. fast-block latency 和 block completion latency 都整体左移
5. request completion 的 p95/p99 大幅下降
6. 整个 200-request benchmark 更早结束
7. 因此 `sum(completion_tokens) / latency` 会明显上升

所以，FDFO 在这里强于 basic，核心原因并不是单次 forward 更便宜，而是它去掉了 fast block 和 slow block 之间的 batch 级耦合，并且更频繁地把控制权还给了 scheduler。

## `pick_tokens` 优化结果

在确认 FDFO 相比 basic 的主要优势之后，我又进一步优化了 FDFO 的 `pick_tokens` 路径，目标不是改算法语义，而是减少这一段 eager PyTorch 实现里的无效计算和多余算子。

### 1. 这次具体优化了什么

原始实现里，`pick_tokens` 的主要问题是：

- 对整个 `[batch, block, vocab]` 都做 `argmax`
- 为了得到 top1 token 的 confidence，先算完整 `softmax`
- 即使很多 position 已经不是 mask，也仍然参与这些计算
- fallback 路径用了 `topk(k=1)`

这次做的优化有四点：

- 只对 masked positions 做计算，不再对整个 block 的所有 position 做 full-vocab 处理
- 用 `max + logsumexp` 替代 `softmax + gather`
- 用 `argmax` 替代 `topk(k=1)` 做 fallback
- 减少中间张量和无意义的临时分配

这些优化都没有改 FDFO 的决策语义，只是在保持等价行为的前提下压低 `pick_tokens` 的执行成本。

### 2. 优化前后测试结果

我在同一台机器上、同一套 `test_llada2_mini.py` 数据集测试里，对 FDFO 跑了优化前后的对比。

优化前：

- `output_throughput = 626.461 token/s`
- `latency = 41.005 s`
- `accuracy = 0.910`

优化后：

- `output_throughput = 653.846 token/s`
- `latency = 37.385 s`
- `accuracy = 0.920`

对应变化：

- 吞吐提升约 `+4.37%`
- 总时延下降约 `-8.83%`

所以，这次优化是有效的，而且收益已经能稳定体现在 benchmark 上。

### 3. trace 指标如何证明这次优化生效

从 trace 看，热点确实降下来了，而不是纯 benchmark 抖动：

- `pick_tokens_ms mean`: `0.922 -> 0.867`，下降约 `5.97%`
- `postprocess_ms mean`: `6.109 -> 5.837`，下降约 `4.45%`
- `request_completion_latency p50`: `18.32s -> 15.86s`
- `request_completion_latency p95`: `23.88s -> 22.53s`
- `block_completion_latency p50`: `425.72ms -> 413.72ms`

这说明这次优化真正压低了 FDFO 的 postprocess 开销，并且已经进一步传导到了 request tail latency 和最终 output throughput。

### 4. 这次优化的意义

这次优化说明一个很重要的事实：

- `pick_tokens` 仍然是值得优化的热点
- 但它已经不是 FDFO 的头号瓶颈

因为即使 `pick_tokens` 明显变快，整体吞吐提升也只有大约 `4%`，这说明当前 FDFO 更大的成本仍然在别处。

## 下一个更值得做的优化方向

如果按收益优先级排，我认为下一个最值得做的方向不是继续抠 `pick_tokens` 的常数项，而是**减少一个 block 完成所需的 round 数**。

### 1. 为什么这是更大的瓶颈

从前面的 trace 看，当前 FDFO 的 block 完成轮数仍然很高：

- `rounds_to_completion` 平均 `14.12`
- `p95 = 25`
- 绝大多数 block 都落在 `5_plus` bucket

这意味着当前真正的大头不是“单轮太慢”，而是“同一个 block 要反复跑很多轮”。

只要 round 数高：

- 同一个 block 就会反复消耗 `model_forward`
- block completion latency 就会被拉长
- request completion latency 也会被连带拉长

相比之下，`pick_tokens` 现在已经只是 postprocess 里的次一级热点。

### 2. 最可能的优化切入点

下一个我最想看的点是 FDFO 的完成判定语义。

当前实现里，`accept_length` 是按 `_pick_tokens()` 之前的 mask count 算的。于是会出现：

- 本轮已经补完最后一个 mask
- 但本轮仍然不算 complete
- 还需要再多一个 round 才真正输出

这会平白多出一轮 scheduler + forward + postprocess 成本。

所以一个很直接的方向是：

- 重新审视 FDFO 的 complete 判定
- 看能不能把“本轮补完最后一个 mask”的 block 直接在本轮完成

如果这个语义改动能做通，它的收益很可能比继续抠 `pick_tokens` 更大，因为它是在直接减少 round 数，而不是只减少单轮时间。

### 3. 如果不改语义，另一个方向是什么

如果不想先动完成语义，那么下一个方向就是继续压 `model_forward` 的无效轮次。

更具体地说，就是想办法让：

- 后期 mask 很少的 block
- 不要再按同样的 round 模式继续完整跑下去

这类优化本质上仍然是在减少“每个 block 需要多少轮”这个核心成本。

### 4. 优先级判断

所以我会这样排序后续优化方向：

1. 优先看能不能减少 block completion 的 round 数
2. 其次看 `accept_length` / complete 判定能不能去掉那一轮额外延迟
3. 再往后才是继续深挖 `pick_tokens`，例如 Triton 融合算子

也就是说，`pick_tokens` 这次已经证明了“抠 postprocess 还有收益”，但下一阶段如果想拿更大的提升，最可能的突破口已经转向 **减少 FDFO 的 round 数，而不是继续只优化单轮后处理**。

## 一个更真实的工程 debug 故事：为什么 FDFO 会故意多保留一轮

如果把这件事讲成一个更接近真实工程现场的故事，我认为大概率是这样发生的。

### 1. 最开始的直觉实现一定是“本轮补完就直接 complete”

开发者第一次做这个功能时，最自然的想法通常是：

- 这一轮 `_pick_tokens()` 之后，如果一个 block 已经没有 mask 了
- 那它看起来就已经完成了
- 既然已经完成，为什么不在本轮直接 complete？

这个想法非常合理，因为这样做看上去能直接省掉一轮：

- 少一次 scheduler 往返
- 少一次 forward
- 少一次 postprocess
- 吞吐理论上会更好

所以第一版实现，很可能就是按 `_pick_tokens()` 之后的状态来判 complete。

### 2. 刚开始的时候，这个改动甚至可能“看起来是对的”

这类问题最麻烦的地方在于，它往往不会立刻炸。

你一开始很可能会看到：

- 吞吐变好了
- 短 case 看起来正常
- 当前这一轮输出的 token 也没明显问题

也就是说，这种改动很容易给人一种错觉：

- “逻辑没问题”
- “性能还更好了”

所以开发者很可能不会第一时间怀疑这里。

### 3. 然后会出现一种很典型的坏味道：当前块没问题，但后续生成开始漂

真正开始暴露问题时，表象一般不是 crash，而是更隐蔽的行为回归：

- 长生成结果和 baseline 对不上
- 跨 block 之后输出开始变怪
- 某些 case 偶发失败，不稳定
- 当前 block 自己看起来是对的，错的是后面的 continuation

这类现象特别容易误导人，因为你会本能地先去怀疑：

- `_pick_tokens()` 的 threshold 逻辑是不是错了
- fallback 位置是不是选错了
- output processor 的 prefill/decode 边界裁剪是不是有 bug
- `dllm_incomplete_ids` 的恢复是不是有问题

但通常你把这些地方查一圈，会发现一个很尴尬的事实：

- token 看着没错
- block 在 `_pick_tokens()` 后也确实已经 fully-filled
- 可后续链路还是不对

### 4. 真正的突破点通常是：不再只看 token，而开始看 KV 是按什么输入算出来的

这个时候，真正关键的工程思路不是继续盯 token，而是把执行顺序完整摊开：

1. 这一轮 forward 开始时，block 里仍然还有 mask
2. 模型是按“带 mask 的输入”算出了 logits
3. 这一轮对应的 KV cache 也已经按这个输入生成好了
4. 之后 `_pick_tokens()` 才把 mask 替换成预测 token
5. 如果此时直接 complete，相当于：
   - output token 来自“填完后的 block”
   - KV 却来自“带 mask 的 block”

到这里问题就清楚了：

- 错的不是 token picker 本身
- 错的是 KV cache 语义和最终 token 状态不一致

### 5. 开发者通常会做一个 replay 对照实验，把这个问题钉死

一个很像真实开发现场的验证方式是：

- 找一个 block，让它在本轮 `_pick_tokens()` 后刚好补完最后一个 mask
- 记录这时的 fully-filled block
- 做两条路径对比

路径 A：

- 不 replay
- 直接把这个 block 当作完成块，沿用当前这一轮的 KV 往后跑

路径 B：

- 用 `_pick_tokens()` 之后那个 fully-filled block，再 fresh forward 一次
- 然后再往后跑

再去比较：

- 下一块 decode 的结果
- 下一轮 logits
- 或者和 full recompute / basic 版本的对齐情况

这类实验通常会给出非常明确的信号：

- 路径 A 的后续结果开始漂
- 路径 B 能和 reference 对齐

一旦看到这个结果，开发者基本就能确认：

- 当前块“看起来已经完成”并不代表这一轮的 KV 可以直接拿来用
- 真正缺失的是一次“按最终 token 输入执行的 forward”

### 6. 接下来通常还会再踩一个认知坑：不是修几个 KV 位置就能解决

知道问题在 KV 之后，下一反应可能是：

- 能不能只修补一下刚刚被替换掉的那些位置？

但很快就会意识到，事情没这么简单。

因为 transformer 的表示是有前后依赖的：

- 前面某个 token 从 mask 变成了真实 token
- 后面位置的 hidden state / K / V 也可能一起变化

所以这通常不是“修一个位置的 KV”能解决的问题，而更像是：

- 整个 block 都需要按最终 token 重新 replay 一次

到这一步，设计空间就会收得很紧。

### 7. 最后才收敛成现在这种保守设计

所以现在的设计，从外面看像是：

- `accept_length` 故意晚判一轮

但如果你是踩过这个坑的人，你心里真正想的其实是：

- 这一轮只是把 token 猜出来了
- 下一轮才是把“和最终 token 一致的 KV”真正补出来

于是系统最后会收敛成现在这种规则：

- 如果一个 block 在进入本轮 forward 时还有 mask
- 那不管本轮 `_pick_tokens()` 后变得多完整
- 这一轮都只能先算 incomplete
- 把这一段 KV free 掉
- 下一轮再用 fully-filled block 重跑一次 forward
- 只有这一轮产生的 KV，才能被安全保留下来并继续服务后续 decode

### 8. 为什么这是一个很真实的工程故事

因为这类设计通常不是一开始就想得这么保守，而是经历了下面这个过程之后被“逼”出来的：

1. 先追求少一轮完成
2. 改完吞吐看起来还变好了
3. 然后长链路行为开始出现偏差
4. 排查 token 逻辑查不出明显问题
5. 最后定位到 KV cache 一致性
6. 通过 replay 对照实验确认必须多补一轮
7. 最终接受一个更保守但正确的设计

所以，如果要把这段经历讲成一句面试里的工程总结，我会这样说：

“我们一开始也尝试过让 block 在本轮补完 mask 后直接完成，因为这样理论上能少一轮、吞吐更高。后来发现当前轮输出的 token 虽然对了，但 KV 还是按带 mask 的输入算出来的，后续 decode 会开始漂。最后通过 replay 和 full recompute 对照，确认 completed block 必须再按最终 token 形式过一轮 forward，才能拿到正确 KV，所以现在才会故意把 complete 判定后移一轮。” 
