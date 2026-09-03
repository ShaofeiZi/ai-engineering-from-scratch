# 前缀缓存服务：RadixAttention 与 KV 复用

> 把 KV cache 当作一种一等、可复用的资源，用 radix tree 来存储，并且连调度策略也要围绕它改变：不是像 vLLM 默认那样用 FCFS（first-come, first-served），而是用 cache-aware scheduler，优先处理共享前缀更长的请求。等价地说，它会沿着 radix tree 做一种深度优先遍历，让热点分支尽量常驻 HBM。SGLang 就是围绕这一思路构建服务引擎的代表。在 Llama 3.1 8B + ShareGPT 风格 1K prompt 上，SGLang 约为 16,200 tok/s，vLLM 约为 12,500，领先约 29%。在前缀高度重复的 RAG 负载上，优势可以到 6.4x；在语音克隆形态负载上，cache hit rate 超过 86%。到 2026 年，它已经部署在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等 400,000+ GPU 上。真正的陷阱在于：只要 prompt 的前缀顺序不稳定，6.4x 这个数字就会立刻蒸发。顺序设计，才是工程师真正能控制的杠杆。

**Type:** 学习
**Languages:** Python（标准库，玩具级基数树缓存与缓存感知调度器）
**Prerequisites:** 第 17 阶段 · 04（服务引擎内部原理），第 14 阶段（智能体 RAG）
**Time:** 约 75 分钟

## 学习目标

- 画出 RadixAttention 的工作方式：前缀如何被存进 radix tree，以及同一分支上的序列如何共享 KV blocks。
- 解释 cache-aware scheduling 的意义，以及为什么 FCFS 对前缀高度复用的流量是错误策略。
- 已知 prefix-cache hit rate 和 prompt 长度分布时，估算一个工作负载的预期加速比。
- 说出哪一种 prompt ordering discipline，决定了 6.4x 是真实收益还是白白错失的上限。

## 问题

传统 serving 系统把每个请求的 prompt 都当成不透明文本处理。即使 5,000 个 RAG 请求都以同一个 2,000-token system prompt 加上相同 retrieval preamble 开头，vLLM 也会把这 2,000-token 前缀重复 prefill 5,000 次。GPU 在一遍又一遍地做完全相同的工作。

关键观察是：在 agentic workload 和 RAG workload 里，prompt 几乎总是共享很长的前缀。system prompt、tool schema、few-shot example、retrieval header、conversation history，这些部分会在大量请求里重复出现。如果这个前缀的 KV cache 只算一次、后续直接复用，那么它就不需要反复 prefill。

RadixAttention 做的正是这件事。token 序列被索引进一个 radix tree；每个节点拥有从 root 到该节点路径上那段 token sequence 对应的 KV blocks。新请求进入时，会沿树向下走：凡是 token 匹配的节点，都直接复用已有 KV blocks。于是 prefill 成本不再与完整 prompt 成正比，而是只和“新增后缀”成正比。

真正的难点在于调度。如果两个请求共享 2,000-token 前缀，而第三个请求只共享其中 200 token，你希望前两个长共享请求被连续处理，这样长前缀才能留在 HBM 里。FCFS 恰恰会做相反的事：谁先到就先服务，结果很可能在下一个长前缀请求到来前，就把热点分支给换出去了。

## 概念

### 把 radix tree 当作 KV 索引

radix tree（压缩 trie）用来存 token sequence。每个节点拥有一个 token range，以及为这段范围计算好的 KV blocks。子节点会在此基础上继续扩展一个或多个 token。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

假设来了一个新请求，内容是 system prompt + "Context: <doc A>" + "Question: Carol"。调度器会沿树往下走：system 前缀匹配，复用 124 个 blocks；doc-A 分支也匹配，再复用 31 个 blocks；只有 "Question: Carol" 这一段需要新分配 blocks，比如 4 个。于是 prefill 成本只剩 4 个 block 的新 token。没有 radix tree 时，则要重新处理 160 个 block。单看 prefill，这就是约 40x 的节省。

### 缓存感知调度

如果 cache 会频繁抖动，radix-tree-backed KV 复用就没有意义。这里有两条关键策略：

1. **Depth-first dispatch。** 从队列里挑下一个请求时，优先选择和当前运行集位于同一分支上的请求。这样可以让热点分支继续常驻。
2. **LRU 以 branch 为单位，而不是以 block 为单位。** 从最少使用的叶子开始，整段分支一起驱逐，而不是零碎驱逐单个 block。这样 cache 形状才会和 radix 形状保持一致。

FCFS 同时违背这两点。一个共享 2,000 token 的请求，可能排在一个只共享 50 token 的请求后面；随后为了接纳后者，2,000-token 那个热分支反而被提前换出。

### 必须记住的基准数字

- Llama 3.1 8B、H100、ShareGPT 1K prompts：SGLang 约 16,200 tok/s，对 vLLM 的约 12,500 tok/s，领先约 29%。
- 前缀高度重复的 RAG（相同 system + 相同文档、问题不同）：SGLang 最高可以到 6.4x。
- 语音克隆类负载：prefix-cache hit rate 可到 86.4%。
- SGLang 客户在生产上的命中率：根据 prompt discipline 不同，大致在 50-99%。
- 到 2026 年，部署规模已超过 400,000+ GPU。

### 顺序陷阱

6.4x 这个数字有一个前提：prompt template 的顺序必须稳定。如果你的客户端有时把 prompt 拼成 `[system, tools, context, history, question]`，有时又拼成 `[system, context, tools, history, question]`，那么 radix tree 根本找不到共享前缀。对人类看起来明明“差不多”的 prompt，在 radix tree 看来却是两条完全不同的 token sequence。

工程师真正能控制的杠杆就是：prompt template 本身就是 cache key。顺序要固定。把不可变的部分放前面，例如 system、tools、schemas；然后是 retrieval context；最后才是 user question。不要把动态内容夹进本来应当可缓存的前缀里。

研究里的一个真实案例是：仅仅把动态内容移出可缓存前缀，就让一个部署的 cache hit rate 从 7% 提高到了 74%。

### RadixAttention 适合与不适合的场景

适合：
- RAG：retrieval preamble 相同，问题不同。
- Agents：tool schema 相同，查询不同。
- 带长 system prompt 的 chat。
- 带重复 preamble 的 voice / vision workload。

不适合，或者说会退回到接近 vLLM 水平的情况：
- 单次生成且 prompt 唯一，例如代码补全，或没有固定 system prompt 的开放聊天。
- 高动态 prompt，每个请求都会把独有内容插进前缀。

### 为什么这首先是调度器问题，而不只是 kernel 问题

你当然可以把 KV 复用做成一种 kernel trick。SGLang 真正有价值的洞见是：只有当 scheduler 能把热点分支留在内存里时，KV 复用才真正值钱。一个天真的“只要能复用就复用”策略，在混合负载下会把 cache 搅得一塌糊涂。真正把 kernel trick 变成 29% 生产优势的，是 radix-tree-indexed scheduler。

### 与 vLLM 的关系

这两套系统并不是绝对对立。到 2026 年，vLLM 也已经加入了 prefix caching（`--enable-prefix-caching`）以及 cache-aware router（Rust 写的 vLLM Router）。差距因此缩小了，但并没有完全消失。SGLang 整个 serving stack 是从 radix-first 的思路出发设计的；vLLM 更像是在已有体系上把这套能力加进去。对于 prefix reuse 非常强的工作负载，SGLang 仍然是默认答案。对于没有明显前缀模式的一般服务，vLLM 依然可能相当甚至更优。

```figure
roofline
```

## 动手用

`code/main.py` 实现了一个 toy radix-tree KV cache 和一个带两种策略的 scheduler：FCFS 与 cache-aware。它会把同一组工作负载分别跑一遍，报告 prefix-cache hit rate 和 throughput delta。然后再跑一个“顺序被打乱”的 workload，展示 6.4x 是如何崩掉的。

## 交付物

这一课产出 `outputs/skill-radix-scheduler-advisor.md`。给定一个工作负载描述，例如 prompt-template 结构、retrieval 模式、并发租户数量，它会给出 prompt ordering 方案，并判断是否值得采用 SGLang。

## 练习

1. 跑 `code/main.py`。比较同一工作负载下 FCFS 与 cache-aware 的差异。delta 主要来自哪里，是 prefill 节省、decode 节省，还是 queue delay？
2. 修改 workload，让 prompt 随机打乱 `[system, tools, context]` 的顺序。重新运行。hit rate 会发生什么变化？为什么？
3. 计算在 Llama 3.1 8B 上，把一个 2,000-token system prompt 常驻成一个 radix branch 所需的 HBM 成本。再和一个没有 prefix reuse 的 16-sequence batch 做对比。
4. 阅读 SGLang 的 RadixAttention 论文。用三句话解释，为什么 tree-shaped LRU eviction 在前缀重负载下优于 block-shaped LRU。
5. 某客户报告只有 8% 的 cache hit rate。列出三个最可能原因，并分别给出你会跑的诊断。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| RadixAttention | “SGLang 那套东西” | 把 KV cache 索引成 radix tree，从而让共享前缀复用 blocks |
| Radix tree | “压缩 trie” | 每个节点拥有一段 token range 及其 KV blocks 的树结构 |
| Cache-aware scheduler | “热点分支优先” | 优先调度共享当前常驻分支的请求 |
| 前缀缓存命中率 | “你的 prompt 有多少是白送的” | prompt token 中有多少比例来自复用的 KV blocks |
| FCFS | “先来先服务” | 会破坏前缀局部性的默认调度策略 |
| Branch-level LRU | “把叶子整段驱逐掉” | 与 radix 形状对齐的整分支驱逐策略 |
| 提示模板顺序 | “缓存键” | prompt 各组件的顺序，决定了 radix tree 能共享什么 |
| 系统提示词常驻 | “常驻前缀” | 让不可变 system 前缀常驻，以减少驱逐抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源码与文档入口
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention 与调度细节
- [SGLang 论文：Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计论文
- [LMSYS 博客：SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准数字与调度思路
- [vLLM：Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的 prefix caching 实现，可作对照
