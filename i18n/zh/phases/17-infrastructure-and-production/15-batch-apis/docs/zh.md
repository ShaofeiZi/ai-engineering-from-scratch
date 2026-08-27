# Batch APIs：50% 折扣已成行业标准

> 几乎所有主要供应商都提供了异步 batch API，统一特点是 50% 折扣和大约 24 小时周转时间。OpenAI、Anthropic、Google，以及大多数推理平台（Fireworks batch tier、Together batch）都采用了同一模式。把 batch 和 prompt caching 叠起来后，夜间 pipeline 的成本通常能降到同步未缓存方案的约 10%。规则简单得近乎粗暴：只要它不是交互式任务，就应该进 batch。内容生成流水线、文档分类、数据提取、报告生成、批量打标、目录标注，只要能容忍 24 小时延迟，不进 batch 就是在白白烧钱。2026 年的生产实践，是把每个新的 LLM 工作负载先分到三条车道里：interactive（同步 + 缓存）、semi-interactive（异步队列 + fallback）、batch（过夜运行 + cached input 叠加）。那些“假装实时”、但其实可以容忍几分钟延迟的任务，浪费最严重。

**Type:** 学习
**Languages:** Python（标准库，玩具级批处理与同步调用成本模拟器）
**Prerequisites:** 阶段 17 · 14（提示词缓存与语义缓存）
**Time:** 约 45 分钟

## 学习目标

- 说出三类主要 provider batch APIs（OpenAI、Anthropic、Google），以及它们共同的 50% 折扣 + 24h turnaround 保证。
- 计算一个夜间分类工作负载在叠加 batch + cached-input 后的成本，并与同步未缓存基线比较。
- 把一个工作负载分到 interactive / semi-interactive / batch 中，并说明为什么这样分。
- 说出两个常见陷阱：partial interactivity（用户对 24h 之外还有更快预期）和 output-schema drift（不同 provider 的 batch 文件格式不一样）。

## 问题

你的团队上线了一条 nightly report generation pipeline。50,000 份文档，每份先做总结，再把总结聚类，最后写一份高管简报。同步跑完整条流程要 4 小时，每晚花费 $2,000。然后你听说了 batch APIs。

只开 batch，就能先省掉 50%。如果你再把 system prompt 的 prompt caching 打开，因为这段 prompt 会在全部 50k 调用里共享，叠加后账单可以降到每晚 $180，也就是基线的约 9%。同一条 pipeline，只改三处配置。

Batch 是整个 LLM 成本工具箱里最便宜、却最少团队会主动去拉的杠杆。原因通常不是技术，而是组织认知：团队脑子里想的是“实时”，而真实 SLA 其实只是“明天早上之前”。这节课要解决的，就是不要把 90% 的费用继续留在桌上。

## 概念

### 三种 batch API

**OpenAI Batch API**：通过 JSONL 文件上传一组请求。承诺 24 小时内返回，实际常见是 2-8 小时。输入和输出 tokens 都打 50% 折扣。入口是 `/v1/batches`。如果输入本身又满足缓存条件，还能继续叠加 cached-input pricing。

**Anthropic Message Batches**：同样是 JSONL 上传。24 小时 turnaround。50% 折扣。支持 `cache_control`，也就是说 cache write 需要显式标记，而 reads 会在 batch 内自动发生。

**Google Vertex AI Batch Prediction**：输入来源可以是 BigQuery 或 GCS。Gemini 也有类似的 50% 折扣，并且天然能和 Vertex pipelines 集成。

### 语义上是异步，不代表它很慢

Batch 的意思是“我承诺 24 小时内返回”，不是“它一定要跑 24 小时”。典型 P50 通常只有 2-6 小时。provider 会把 batch 调度到离峰时段，以便利用空闲 GPU 库存。

### 与缓存叠加

假设你有一条 50k-document summarization 任务，而且所有请求共享同一个 4K-token system prompt：

- Synchronous uncached：50000 × ($input × 4000 + $output × 200)，全部按原价计算。
- Synchronous cached：system prompt 在第一次 write 之后被缓存，剩余 49999 次请求获得 10 倍更便宜的输入价格。
- Batch cached：在以上基础上，再给 read 和 write 都加上 50% 的 batch 折扣。

结论就是：batch + cache 叠加后，成本大约只有 sync uncached 的 10%。只要一个工作负载会过夜运行，并且 system prompt 是共享的，就应该这样做。

### 工作负载分流

**Interactive**：用户在等待响应。TTFT 是关键。应走同步调用，并配合 prompt caching。不能用 batch。

**Semi-interactive**：用户提交任务，几分钟后回来查看。应走异步队列；如果 batch 不可用，再 fallback 到同步调用。中等规模的 RAG indexing 就是典型例子。

**Batch**：用户期望的是“明早给我”或者“下一小时给我”。内容流水线、大规模分类、离线分析，都属于这一类。总原则是：能 batch 就 batch，且要叠加 cache。

最常见的错误，是因为这个 pipeline 属于生产环境，就把它强行归为 interactive。生产不是延迟要求，SLA 才是。

### 部分交互性陷阱

有些功能看起来是交互式，但其实完全能容忍 5-10 分钟。例如一份 nightly customer health report 上加了一个“refresh”按钮。用户点了刷新，等 10 分钟其实完全可以接受。可团队却把它按同步接口来做。结果 50 个并发刷新，成本是“batch 后通过邮件回送”的 10 倍。

真正该问的问题是：“对这个用户来说，24 小时意味着什么？”如果答案是“他们根本不会察觉”，那就应该进 batch。

### 输出 schema 陷阱

不同 provider 的 batch 文件格式并不统一：

- OpenAI：JSONL，每行一个 request。
- Anthropic：JSONL，每行一个 message，响应格式直接嵌在消息里。
- Vertex：BigQuery table 或 GCS prefix，配合 TFRecord。

所以你要写一个“跨 provider 的统一 batch client”，本质上还是得为每个 provider 单独写 adapter code。那些宣传自己支持 multi-provider batch 的 gateway（例如某些层级的 Portkey、LiteLLM），本质上也只是对底层原始格式做了一层薄封装。

### 你需要记住的数字

- 各 provider 的 batch 折扣：输入 + 输出统一 50%。
- Turnaround SLA：保证 24 小时内返回；典型 P50 是 2-6 小时。
- Stacked batch + cached input：约等于 sync uncached 成本的 10%。
- Workload triage rule：如果 24h 延迟可以接受，就始终应该进 batch。

```figure
batch-lane-triage
```

## 用起来

`code/main.py` 会计算一条 50k-document 工作负载在 sync、sync+cache、batch、batch+cache 四种模式下的成本，并以美元和百分比形式展示节省幅度。

## 产出

这一课会产出 `outputs/skill-batch-triager.md`。给定工作负载特征，它会把任务分到 interactive / semi / batch，并估算节省空间。

## 练习

1. 运行 `code/main.py`。对一条包含 100k documents、3K-token system prompt 和 500-token output 的 pipeline，计算 full stack（batch + cache）相对于 sync baseline 的节省。
2. 从一个真实产品里挑三个你熟悉的功能，分别把它们分到 interactive / semi / batch。
3. 用户抱怨一份报告花了 3 小时才生成出来。这是 batch 误分流，还是合法的 interactive？请写出你的判定标准。
4. 如果你的 batch API 返回 SLA 是 24h，但 P99 是 20 小时，你要如何向用户沟通？边界情况时，下游系统应当怎么处理？
5. 计算 break-even：共享前缀长度达到多少后，batch + cache 会比在你自己的保留 GPU 上过夜运行更便宜？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Batch API | “异步折扣” | 以 24h 周转换取 50% 折扣 |
| JSONL | “批处理格式” | 每行一个 JSON 请求；OpenAI/Anthropic 的标准格式 |
| Message Batches | “Anthropic batch” | Anthropic 的 batch API 产品名 |
| Batch prediction | “Vertex batch” | Vertex AI 的 batch API 产品 |
| Turnaround SLA | “24 小时承诺” | 保证上限，不是典型耗时；典型是 2-6h |
| Workload triage | “交互性决策” | interactive / semi / batch 的分流决策 |
| Output schema | “响应格式” | 各 provider 自己的 JSONL 或表结构；不可直接通用 |
| Stacked discount | “批处理 + 缓存” | 两者叠加后，成本约为 sync uncached 的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL 格式与 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch 格式与 `cache_control` 的交互方式。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) — Gemini 的 batch 语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
