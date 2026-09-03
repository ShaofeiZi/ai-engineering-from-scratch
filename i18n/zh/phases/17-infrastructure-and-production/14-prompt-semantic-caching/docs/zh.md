# 提示词缓存与语义缓存的经济学

> **价格快照记录于 2026-04。** 下文中的数字都来自本课发布时记录的供应商价格表；如果你要在别处引用，请先对照链接文档重新核实。

> 缓存分成两层。L2（provider-level）prompt/prefix caching 会复用重复前缀的 attention KV。Anthropic 的 prompt-caching 文档声称，在长 prompt 上可实现最多 90% 的成本下降和 85% 的延迟下降；以 Claude 3.5 Sonnet 为例，cache read 的价格是 $0.30/M，而 fresh input 是 $3.00/M，5-minute TTL 下如此，1-hour TTL 还会额外收取 2x write premium（docs.anthropic.com，2026-04）。OpenAI prompt caching 则会对 prompts ≥1024 tokens 自动生效，cached input 相比 fresh input 大约便宜 90%（platform.openai.com，2026-04）；但每个模型的确切 cached rate 仍以实时价目表为准。L1（app-level）semantic caching 则是在 embedding similarity 命中后直接跳过 LLM。供应商口中的“95% accuracy”指的是匹配正确率，不是 hit rate；生产环境里报告的 hit rate 可能从 10%（open-ended chat）到 70%（structured FAQ）不等，而官方并没有发布统一基线，所以这些数字更应该视作社区 telemetry，而不是保证。生产环境中最常见的两个坑是：并行化会杀死缓存效果（在第一次 cache write 完成之前就发出 N 个并行请求，会让花费膨胀数倍），以及前缀中混入动态内容会让 cache hit 完全消失。ProjectDiscovery 在 2025-11 报告说，仅仅通过把动态文本移出 cacheable prefix，就把 hit rate 从 7% 提升到了 74%。

**Type:** 学习
**Languages:** Python（标准库，玩具级两层缓存模拟器）
**Prerequisites:** 第 17 阶段 · 04（服务引擎内部原理），第 17 阶段 · 06（SGLang RadixAttention）
**Time:** 约 60 分钟

## 学习目标

- 区分 L2 prompt/prefix caching（在 provider 侧复用 KV）与 L1 semantic caching（对相似 prompt 直接绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记方式，以及两个 TTL 选项（5-min 和 1-hour）对应的价格倍率。
- 根据 hit rate、prompt/response 结构和 token 单价，计算预期的月度节省。
- 说出那个会让账单膨胀 5-10 倍的 parallelization anti-pattern，以及让 hit rate 崩塌的 dynamic-content anti-pattern。

## 问题

你给自己的 RAG 服务加上了 prompt caching。账单却没降。你去量 hit rate，只有 7%。你的 prompts 看起来很静态，其实并不是这样。系统提示里包含按分钟格式化的当前时间、request ID，以及为了“多样性”而做的随机示例重排。于是每个请求都会写出一条新的 cache entry，却没有任何一次 read。

另一方面，你的 agent 每次用户提问都会并行发出十个工具调用。这十个请求都在第一次 cache write 完成前就到了 provider。十次 write，零次 read。结果你的账单变成了“开了缓存之后应该有的价格”的 5-10 倍。

缓存是一种协议，不是一个开关。两层缓存，对应两种截然不同的失效模式。

## 概念

### L2：供应商侧提示词/前缀缓存

provider 会把可缓存前缀对应的 attention KV 存起来，当下一个请求命中相同前缀时直接复用。你只在第一次 write 时付出成本，之后的 reads 几乎等于白送。

**Anthropic (Claude 3.5 / 3.7 / 4 series)**：请求里需要显式打上 `cache_control` 标记，用来说明哪些 blocks 可以缓存。TTL 分成两档：5-minute（write 成本是基础价的 1.25x）和 1-hour（write 成本是基础价的 2x）。以 Claude 3.5 Sonnet 为例，cache read 是 $0.30/M，而 fresh input 是 $3.00/M，便宜约 10 倍（docs.anthropic.com，2026-04）。不同模型的价格会不同（Opus/Haiku 单独公布），所以始终要以实时价格页为准。

**OpenAI**：对 prompts ≥1024 tokens 自动启用缓存（platform.openai.com，2026-04），不需要显式开关。按当前 gpt-4o/gpt-5 价目表，cached input 大约比 fresh input 便宜 10 倍。无论官方文档还是 release notes，都没有给出统一的 hit-rate baseline；社区经验通常落在 30–60%，前提是 prompt 设计得足够稳定。你需要通过 `usage.cached_tokens` 自己测量真实命中情况。

**Google (Gemini)**：通过显式 API 提供 context caching；当上下文达到 1M tokens 量级时，缓存的收益更明显。

**Self-hosted (vLLM, SGLang)**：Phase 17 · 06 已经讲过 RadixAttention，本质上是同样的模式，只不过缓存运行在你自己的计算资源上。

### L1：应用层语义缓存

在调用 LLM 之前，先对 prompt 做哈希、做 embedding，然后查找是否存在一条相似的已缓存请求（通常要求 cosine similarity > 0.95）。命中就直接返回缓存响应；未命中才调用 LLM，并把结果写入缓存。

开源方案包括：Redis Vector Similarity、GPTCache、Qdrant。商业方案包括：Portkey Cache、Helicone Cache。

供应商宣称的 accuracy，指的是返回的缓存响应在语义上是否合适，而不是你有多高概率会命中。生产环境里更常见的 hit rate 区间是：

- 开放式聊天：10-15%。
- 结构化 FAQ / 支持问答：40-70%。
- 代码问题：20-30%（轻微变体就足以让命中消失）。
- 反复重复提示的语音 agent：50-80%（因为语音归一化后集合更稳定）。

### 并行化反模式

你的 agent 并行发起 10 个工具调用。它们都带着同一个 4K-token system prompt。Anthropic 的 cache write 是按请求单独完成的；provider 看到 prompt 后，大概 300 ms 左右，第一个 cache write 才会完成。于是第 2 到第 10 个请求在同一个毫秒窗口里抵达时，看到的全是 cache miss。你为 10 次请求都支付了 write premium，却一次 read discount 都没享受到。

修复方式是：sequential-first。先单独发出请求 1，等它把缓存写好，再并发触发 2-10。这样会给第一个工具调用增加大约 300 ms，但能把账单压回原来的 1/5 到 1/10。

### 动态内容反模式

你的 system prompt 可能长这样：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都不一样。每个请求都会触发 write。零命中。

修复方式是：把真正静态的部分前移到 cacheable prefix 中，把动态内容放到缓存边界之后：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 就是通过这个调整，把 cache hit rate 从 7% 提升到了 74%，并公开分享了具体拆解过程。

### 把 batch 和 cache 叠起来，用于夜间工作负载

Batch APIs（Phase 17 · 15）本身就提供 24 小时周转下的 50% 折扣。如果再把 cached input 叠加上去，通常还能再拿到约 10 倍收益。对夜间分类、批量打标、报告生成这类工作负载来说，stack batch + cache 之后，总成本可以降到同步未缓存方案的大约 10%。

### 你需要记住的数字

下面这些价格点记录于 2026-04，对应的是当时链接中的供应商文档；这类价格每几个月就可能变化一次，所以正式依赖前要重新核查。

- Anthropic cached read：Claude 3.5 Sonnet 上是 $0.30/M，大约比 fresh input 便宜 10 倍（docs.anthropic.com）。
- Anthropic cache write premium：1.25x（5-min TTL）或 2x（1-hour TTL）。
- OpenAI auto-cache：对 prompts ≥1024 tokens 生效；cached input 价格大约是 fresh input 的 10%（platform.openai.com）。
- Semantic cache hit rate（社区口径）：开放式聊天约 ~10%，结构化 FAQ 最高可到 ~70%；并不是供应商官方基线。
- ProjectDiscovery：通过把动态内容移出 prefix，hit rate 从 7% → 74%（项目博客，2025-11）。
- Parallelization anti-pattern：当 N 个并行请求都错过第一次 cache write 时，账单膨胀 5–10x 是常见现象。

```figure
semantic-cache-hit
```

## 用起来

`code/main.py` 会在混合工作负载上模拟 L1 + L2 caching，输出 hit rates、账单，并直观展示并行化带来的惩罚。

## 产出

这一课会产出 `outputs/skill-cache-auditor.md`。给定 prompt 模板与流量模式，它会审计 cacheability，并给出重构建议。

## 练习

1. 运行 `code/main.py`。切换 parallelization flag 后，账单变化了多少？
2. 你的 system prompt 带有日期字段。把它移出去，并给出前后 hit rate 的计算。
3. 根据你的请求到达率，计算 1-hour TTL（2x write）和 5-minute TTL（1.25x write）各自的 break-even。
4. Semantic cache 在 0.95 threshold 时命中 20%；降到 0.85 后命中 50%，但开始出现错误缓存响应。你会选哪个阈值？为什么？
5. 你对每个用户问题都批量发出 10 个并行子查询。请把这套流程改写得更利于缓存，同时不要增加端到端延迟。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| L2 prompt cache | “前缀缓存” | provider 保存可重复前缀的 KV |
| `cache_control` | “Anthropic 缓存标记” | 用于标记可缓存 blocks 的显式属性 |
| 缓存写入溢价 | “写入税” | 第一次 miss-to-cache 时需要额外支付的成本（1.25x 或 2x） |
| L1 semantic cache | “嵌入缓存” | 在调用 LLM 前做哈希和 embedding 的应用层缓存 |
| GPTCache | “LLM 缓存库” | 常见的 OSS L1 cache 库 |
| Cache hit rate | “命中数 / 总数” | 从缓存直接服务的请求占比 |
| Parallelization anti-pattern | “N 次写入陷阱” | N 个并行请求连续 N 次错过同一份缓存 |
| 动态内容陷阱 | “提示词内时间字段陷阱” | 前缀中的动态字节会直接杀死 hit rate |
| RadixAttention | “副本内缓存” | SGLang 的 prefix-cache 实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义与 TTL 说明。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存的行为与生效条件。
- [TianPan — 面向生产环境的 LLM 语义缓存](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — 用 Prompt Caching 将 LLM 成本降低 59%](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
