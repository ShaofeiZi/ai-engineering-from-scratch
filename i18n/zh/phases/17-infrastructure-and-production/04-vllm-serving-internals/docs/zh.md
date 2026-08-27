# 服务引擎内部原理：PagedAttention、连续批处理与分块预填充

> 现代服务引擎的吞吐并不是靠某一个“黑魔法”提上去的，而是靠三个默认机制叠加出来的。PagedAttention 默认始终开启。连续批处理会在每一次 decode 迭代之间，把新请求插入当前活动 batch。分块预填充会把长 prompt 切成小块，让 decode token 不至于一直挨饿。三者一起打开后，一张 H100 SXM5 上运行的 Llama 3.3 70B FP8，在 128 并发下可以达到 2,200 到 2,400 tok/s，大约比 vLLM 自己的默认配置高出 25%，也比一个天真的 PyTorch 循环快 3 到 4 倍。本课会带你读 vLLM 的 scheduler 和 attention kernel。vLLM 正是这三项技术的参考实现。最终你会在 `code/main.py` 里看到一个玩具版连续批处理器，按 vLLM 的方式去调度 prefill 和 decode。

**Type:** 学习
**Languages:** Python（标准库，玩具级连续批处理调度器）
**Prerequisites:** 阶段 17 · 01（模型服务），阶段 11（LLM 工程）
**Time:** 约 75 分钟

## 学习目标

- 把 PagedAttention 解释成一个 KV cache 分配器，说明 block、block table，以及为什么在生产负载下碎片率能压到 4% 以下。
- 在迭代级别画出 continuous batching 的调度过程，说明已完成序列如何离开 batch，新序列如何在不清空 batch 的情况下加入。
- 用一句话说明分块预填充是什么，并指出它保护的是哪个延迟指标，提示：是 TTFT 尾延迟，不是平均吞吐。
- 说出 2026 年 vLLM v0.18.0 的一个关键坑点，也就是团队把所有优化同时打开时最容易踩到的问题。

## 问题

一个天真的 PyTorch 服务循环会一次处理一个请求：tokenize、prefill、decode 到 EOS，然后返回结果。只有一个用户时，这样也能跑；一百个用户时，它就变成一条排满耐心等待者的长队。一个直觉上的补救办法是做静态批处理，但静态 batching 会把每个请求 pad 到窗口里最长的 prompt，把每个 decode pad 到最长的预期输出，还会让整个 batch 被最慢的那条序列拖住。你为根本用不到的 padding 付出了代价，而快请求也要陪慢请求一起等。

vLLM 同时解决了三类问题。PagedAttention 避免了传统连续内存分配那种会吃掉 60% 到 80% GPU 显存的 KV cache 碎片。Continuous batching 允许请求在每一轮 decode 之间进出 batch，因此 batch 里始终装着真实工作。分块预填充则会把一个 32k-token 的长 prompt 切成约 512-token 的小片段，并与 decode 交错执行，这样超长 prompt 就不会把 GPU 上所有 decode token 一起卡死。

2026 年的生产默认值，通常就是这三项全开。你必须真的理解每一项在干什么，因为它们的故障模式都出现在 scheduler 上，而不是模型本身。

## 核心概念

### 把 PagedAttention 看成虚拟内存系统

对一条序列来说，KV cache 大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。以 Llama 3.3 70B、8192 tokens 为例，单条序列在 BF16 下大约要占 1.25 GB。如果你为每个请求都预先连续保留 8192 个 token 的位置，但平均请求实际上只用到 1500 tokens，那你就浪费了大约 82% 的预留 HBM。传统 batching 就是这么浪费的。

PagedAttention 借用了操作系统虚拟内存的思想。KV cache 不再按序列连续分配，而是按固定大小的 block 来分配，默认每个 block 16 tokens。每条序列都有一张 block table，用来把逻辑 token 位置映射到物理 block ID。序列增长时，就额外分配一个 block；序列结束时，block 再归还给池子。

碎片率会从传统方案的 60% 到 80%，降到 PagedAttention 下的 4% 以下。你不需要用某个 flag 去“开启” PagedAttention，它本来就是 vLLM 唯一提供的分配器。真正的控制旋钮是 `--gpu-memory-utilization`，默认值是 0.9，它告诉 vLLM 在加载完权重和激活值后，应该把多少 HBM 预留给 KV blocks。

### 迭代级连续批处理

旧式“动态批处理”的做法，是先等一个窗口，例如 10 ms，把 batch 攒满，再运行 prefill + decode + decode + decode，直到 batch 里的所有序列都结束。快序列会提前跑完，然后空在那儿等 GPU 把最慢的几条序列收尾。

Continuous batching 不是按请求调度，而是在每一次 decode step 之间调度。把当前运行中的序列集合记作 `RUNNING`。每一次迭代里会发生三件事：

1. `RUNNING` 中任何刚刚到达 EOS 或 max_tokens 的序列都会被移除。
2. 调度器检查等待队列。如果有空闲的 KV blocks，就把新的序列放进来，不管它是刚进入的 prefill，还是暂停后恢复的序列。
3. 前向计算会在当前 `RUNNING` 里的全部序列上执行，每条序列各产出一个新 token。

batch size 不再被 pad 到某个固定值。输出进度不同的序列，可以共享同一个融合后的 forward。在 2026 年的 vLLM 中，这一套就叫 `V1 scheduler`。最关键的约束是：调度器每个 decode iteration 都运行一次，而不是每个请求运行一次。

### 分块预填充保护 TTFT 尾延迟

Prefill 是 compute-bound 的。对 Llama 3.3 70B 来说，一个 32k-token prompt 在单张 H100 上大概要花 800 ms 纯 prefill 时间。只要 prefill 正在跑，batch 里其他序列的 decode token 就都得排队。在 serving loop 里，一个超长 prompt 的 first-token latency，也就是 TTFT，会变成几十个其他用户 inter-token latency，也就是 ITL 的突刺。

分块预填充的办法，是把 prefill 拆成固定大小的块，默认约 512 tokens，并把每个 chunk 当成一个调度单元。chunk 与 chunk 之间，调度器就有机会让 decode 序列先推进一个 token。你付出的代价，是每个 chunk 会多出几毫秒的绝对 prefill 开销；换来的收益，是 decode 侧抖动显著降低。在公开基准里，混合负载下的 P99 ITL 可以从大约 50 ms 降到约 15 ms。

### 三个默认值是互相配合的

这三项特性默认就假设彼此存在。PagedAttention 给调度器提供了足够细粒度的 KV 资源，让它能在 block 预算内权衡取舍。Continuous batching 依赖这种细粒度资源，否则接纳一个新序列就会逼出全局重排。Chunked prefill 也不是独立系统，而是同一个 `RUNNING` 列表上的另一条 scheduler policy。

你不需要记住所有 flag。你真正要理解的是：调度器到底在优化什么。答案是，在固定 KV block 预算下，尽量最大化 goodput，同时通过分块预填充切片约束长 prompt 对尾延迟的伤害。

### 2026 年 v0.18.0 的坑点

在 vLLM v0.18.0 里，`--enable-chunked-prefill` 不能和基于 draft model 的 speculative decoding，也就是 `--speculative-model`，一起用。文档中明确写出的例外，是 V1 scheduler 里的 N-gram GPU speculative decoding。很多团队喜欢把所有看起来能提速的 flag 一次性全开，结果不是性能轻微回退，而是服务在启动时直接报运行时错误。如果你本来就是为了 speculative gain 才想同时打开分块预填充，那 2026 年更常见的正确答案，往往是“用 EAGLE-3，不开分块预填充”，而不是“draft model 加分块预填充，最后根本跑不起来”。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三项全开时大约是 2,200 到 2,400 tok/s。
- 同一模型，用 vLLM 默认配置但不开分块预填充，大约是 1,800 tok/s。
- 同一模型，用天真的 PyTorch forward loop，大约只有 600 tok/s。
- PagedAttention 在生产负载下的 KV 碎片浪费低于 4%。
- 混合负载下的 P99 ITL，开分块预填充时约 15 ms，不开时约 50 ms。

### 调度器大概长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 本质上就是用 stdlib Python 实现了这一套循环，只不过 token 数量和 forward 延迟都是假的。运行它，你就能看到分块预填充是如何在长 prefill 期间，仍然让 decode 序列继续活着往前走的。

```figure
tensor-parallel
```

## 实际使用

`code/main.py` 会模拟一个带可切换特性的 vLLM 风格调度器。运行它时，你会看到：

- `NAIVE` 模式：一次只处理一个请求，没有 batching。
- `STATIC` 模式：先 pad 再等待，属于经典批处理。
- `CONTINUOUS` 模式：按迭代级别接纳和释放序列。
- `CONTINUOUS + CHUNKED` 模式：把 prefill 切片，并和 decode 交错运行。

输出里会给出总吞吐，也就是每虚拟秒生成多少 tokens，TTFT 均值，以及 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 这一行通常应该表现最好。

## 交付成果

本课会产出 `outputs/skill-vllm-scheduler-reader.md`。给定一份 serving 配置，例如 batch size、KV memory utilization、chunked prefill size、speculative config，它会输出一份 scheduler 诊断，指出三项默认机制里到底是哪一项成了瓶颈，以及下一步应该调什么。

## 练习

1. 运行 `code/main.py`。在短请求和长请求混合的负载下，比较 `STATIC` 和 `CONTINUOUS`。吞吐差距主要来自 prefill 效率、decode 效率，还是尾延迟？
2. 修改这个玩具 scheduler，加上 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，合理值应该是多少？提示：它取决于 KV block 大小和空闲 block 数量，而不是原始 HBM 总量。
3. 重读 vLLM v0.18.0 的 release notes。哪些 flag 组合彼此互斥？把它们列出来。
4. 对一条包含 1,000 个请求的 trace 计算 KV cache 碎片浪费，已知输出 token 平均值是 1,500，标准差是 600。分别计算：(a) 每请求按 8192 上限做连续分配；(b) 使用 16-token blocks 的 PagedAttention。
5. 用一段话解释，为什么分块预填充本身保护的是 P99 ITL，而不是单独直接提升吞吐。实践中的吞吐提升到底来自哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | “那个 KV 技巧” | 面向 KV cache 的固定大小 block 分配器，碎片率低于 4% |
| Block table | “页表” | 每条序列自己的映射表，把逻辑 token 位置映射到物理 KV block |
| Continuous batching | “终于做对了的动态批处理” | 每个 decode iteration 都重新做接纳和释放决策 |
| Chunked prefill | “把 prefill 切开” | 把长 prefill 切成约 512-token 的片段，并与 decode 交错执行 |
| TTFT | “首 token 时间” | prefill、排队、网络延迟之和；长 prompt 下通常主要受 prefill 影响 |
| ITL | “token 间延迟” | 连续两个 decode token 之间的时间；通常主要受 batch size 影响 |
| Goodput | “满足 SLO 的吞吐” | 每秒产出多少 tokens，同时每个请求仍满足 TTFT 和 ITL 目标 |
| V1 scheduler | “新版调度器” | vLLM 在 2026 年使用的调度器；N-gram spec decode 是与 chunked prefill 兼容的路径 |
| `--gpu-memory-utilization` | “显存旋钮” | 在加载完权重与激活值后，预留给 KV blocks 的 HBM 比例 |

## 延伸阅读

- [vLLM 文档：Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — vLLM 官方关于 chunked prefill 与 speculative decoding 兼容性的说明。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年版本节奏与具体版本行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 最初那篇定义了这一分配器思维方式的文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 关于碎片分析与调度设计的论文。
- [Aleksa Gordic — 深入 vLLM 内部](https://www.aleksagordic.com/blog/vllm) — 带火焰图的 V1 scheduler 深入讲解。
