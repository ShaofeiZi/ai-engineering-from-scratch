# LLM 可观测性栈选型

> 2026 年的可观测性市场大致分成两类。开发平台（LangSmith、Langfuse、Comet Opik）把监控、evals、prompt 管理、session replay 打包在一起；网关和埋点工具（Helicone、SigNoz、OpenLLMetry、Phoenix）则聚焦于 telemetry。Langfuse 以 MIT 授权核心实现了较好的 OSS 平衡（云端每月免费 50K events）。Phoenix 原生支持 OpenTelemetry，采用 Elastic License 2.0，非常适合做漂移和 RAG 可视化，但并不是一个持久化的生产后端。Arize AX 通过零拷贝的 Iceberg/Parquet 集成，声称成本可以比单体式可观测方案低 100 倍。LangSmith 在 LangChain/LangGraph 场景下领先，价格为 $39/user/mo，且只有 Enterprise 才支持 self-host。Helicone 走 proxy 模式，15-30 分钟即可接入，每月免费 100K req，但对 agent trace 的深度较弱。常见的生产搭配是：Gateway（Helicone/Portkey）+ eval platform（Phoenix/TruLens），中间用 OpenTelemetry 粘合。

**Type:** 学习
**Languages:** Python（标准库，玩具级 trace 采样模拟器）
**Prerequisites:** 阶段 17 · 08（推理指标），阶段 14（智能体工程）
**Time:** 约 60 分钟

## 学习目标

- 区分开发平台（打包了 evals + prompts + sessions）与网关/telemetry 工具（只提供 traces + metrics）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到它们的许可证、定价和最适用场景。
- 解释 OpenTelemetry 粘合模式，以及它如何让你把网关工具和单独的 eval 平台组合起来。
- 说出 2026 年的成本分水岭（Arize AX 的零拷贝方案对比单体 ingest），并记住大约 100 倍这个量级。

## 问题

你上线了一个 LLM 功能。它能跑，但你完全看不到 prompt 失败、工具循环、延迟回归、成本尖峰，或者 prompt cache hit rate。你去搜“LLM observability”，结果看到八个工具，都说自己在解决同一个问题，只是价格各不相同。

它们其实解决的不是同一类问题。LangSmith 回答的是“为什么这个 LangGraph run 失败了？”；Phoenix 回答的是“我的 RAG pipeline 漂移了吗？”；Helicone 回答的是“哪个应用在烧 tokens？”；Langfuse 回答的是“我能不能把整套东西 self-host？”工具不同，受众也不同。

选型通常要看四个轴：技术栈（LangChain？原生 SDK？多供应商？）、许可证接受度（必须 MIT？Elastic 可以？商业也行？）、预算（free tier？$100/mo？$1000/mo？），以及 self-host 要求（必须、自带加分、还是完全不需要）。

## 概念

### 两大类别

**开发平台**会把 observability、evals、prompt management、dataset versioning、session replay 打成一套。你可以直接做实验，比较哪一个 prompt 更好，也可以用数据集对新 prompt 做 regression，对照旧版本结果。代表产品包括 LangSmith、Langfuse、Comet Opik。

**网关/telemetry 工具**专注于推理调用本身的埋点：prompt、response、tokens、latency、model、cost。Helicone、SigNoz、OpenLLMetry、Phoenix 都属于这一类。它们更轻、更专一，也更容易通过 OpenTelemetry 与独立 eval 工具组合使用。

### Langfuse：OSS 平衡点

- 核心采用 Apache / MIT 许可；可通过 Docker self-host。
- 云端 free tier：50K events/month。付费版：团队 $29/mo。
- 覆盖 evals、prompt management、traces、datasets，基本把开发平台四件套都补齐了。
- 最适合的情况是：你想要接近 LangSmith 的功能，但又必须 self-host，或者必须保持 OSS 许可证。

### Phoenix（Arize）：telemetry 优先、原生 OpenTelemetry

- Elastic License 2.0；self-host 难度低。
- 在 RAG 和 drift 可视化上非常强，embedding space scatter plots 是一等功能。
- 它不是按“持久化生产后端”来设计的，主要偏向开发阶段可观测性。
- 最适合 RAG pipeline 开发、漂移排查，以及和独立 gateway 搭配处理生产场景。

### Arize AX：规模化路线

- 商业产品。通过 Iceberg/Parquet 做零拷贝数据湖集成。
- 声称在大规模场景下比单体式可观测性方案（类似 Datadog）便宜约 100 倍。核心逻辑是：trace 数据存到你自己的 S3 Parquet 中，Arize 直接读取。
- 最适合每天超过 10M traces、已经有数据湖、又想要 LLM 专用 dashboard 但不想承受 Datadog 定价的团队。

### LangSmith：LangChain/LangGraph 优先

- 商业产品，$39/user/month。只有 Enterprise 才支持 self-host。
- 在 LangChain 和 LangGraph 栈上属于 best-in-class；如果你不在这两个生态里，它的吸引力会明显下降。
- 最适合已经明确押注 LangChain，并且愿意为此付费的团队。

### Helicone：proxy 模式的最低可行方案

- 只要把你的 `OPENAI_API_BASE` 切到 Helicone proxy，15-30 分钟就能完成接入。
- MIT 许可；每月免费 100K req，付费从 $20/mo+ 起。
- 同时提供 failover、caching、rate limits，因此本身也兼具 gateway 职能。
- 对 agent / multi-step trace 的深度不如开发平台。
- 最适合想快速启动、技术栈单一，又希望 gateway + observability 二合一的应用。

### Opik（Comet）：OSS 开发平台

- Apache 2.0，完全 OSS。
- 功能面与 Langfuse 相近，但带有 Comet 生态背景。
- 最适合已经在用 Comet 的 ML 团队，希望在同一个界面里查看 LLM observability。

### SigNoz：OpenTelemetry 优先的一体化 APM

- Apache 2.0。既能处理通用 APM，也能通过 OpenTelemetry 覆盖 LLM 调用。
- 最适合想把普通服务和 LLM 调用统一纳入同一套 observability 的团队。

### 粘合层：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年底发布了 GenAI semantic conventions，例如 `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`。凡是消费 OTel 的工具，就有了互操作能力。当前逐渐成型的生产模式是：

1. 所有 LLM 调用都发出带 GenAI conventions 的 OTel。
2. 日常运维流量先路由到 gateway（Helicone / Portkey）。
3. 同时双写到 eval 平台（Phoenix / Langfuse）做回归分析。
4. 长期归档则进入数据湖（Iceberg），再由 Arize AX 或 DuckDB 做长期分析。

### 陷阱：埋点埋在了错误的层

如果你把埋点放在 agent framework 内部，比如直接加 LangSmith traces，你就被这个框架绑定住了。把埋点放在 HTTP/OpenAI-SDK 这一层，比如通过 OpenLLMetry 或 gateway，则更具可移植性。

### 采样：你不可能全留

当请求量超过每天 1M 时，完整保存所有 trace 的成本甚至会高过 LLM 调用本身。更现实的做法是按规则采样：100% 保留错误，100% 保留高成本请求，5% 保留成功请求。聚合指标始终保留，原始 trace 只为长尾问题保留样本。

### 你需要记住的数字

- Langfuse 云端免费额度：50K events/month。
- LangSmith：$39/user/month。
- Helicone 免费额度：100K req/month。
- Arize AX 的宣传口径：在大规模场景下比单体式方案便宜约 100 倍。
- OpenTelemetry GenAI conventions：2025 年发布，2026 年已被广泛采用。

```figure
i4-otel-glue
```

## 用起来

`code/main.py` 会模拟一个 1M-trace 的一天，在不同保留策略下（100% ingest、sampling、sampling + errors）分别计算存储成本，并展示每种策略会丢掉什么。

## 交付成果

这一课会产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算和许可证约束，它会选出合适的工具组合。

## 练习

1. 你的团队使用 LangChain，但又想要 OSS self-hosted observability。在 Langfuse 和 Opik 之间做选择，并说明理由。
2. 如果每天有 5M traces，而 Datadog 的报价是 $150K/month，计算 Arize AX 的 break-even。
3. 设计一套 OpenTelemetry GenAI attribute set，作为你们组织要求每次 LLM 调用都必须打出的埋点字段。
4. 论证 Phoenix 单独使用时是否足以支撑生产环境。在哪些情况下它不够？
5. Helicone 额外引入 20 ms proxy 开销。如果你的 P99 TTFT 是 300 ms，这能接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| OpenLLMetry | “OTel for LLMs” | 面向 LLM 的开源 OpenTelemetry instrumentation |
| GenAI conventions | “OTel attributes” | LLM 调用的标准 OTel 属性名 |
| LangSmith | “LangChain observability” | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | “OSS LangSmith” | MIT OSS，功能集与之相近 |
| Phoenix | “Arize dev tool” | 原生 OpenTelemetry 的开发 / eval 平台 |
| Arize AX | “scale observability” | 商业化的零拷贝 Iceberg/Parquet 可观测方案 |
| Helicone | “proxy observability” | 收集 LLM telemetry 的 HTTP proxy，同时带 gateway 功能 |
| Opik | “Comet LLM” | 来自 Comet、采用 Apache 2.0 的 OSS 开发平台 |
| Session replay | “trace rerun” | 重放一次完整的 agent session，包括工具调用 |
| Eval | “offline test” | 在标注数据集上运行候选模型或 prompt |

## 延伸阅读

- [SigNoz — 2026 顶级 LLM 可观测工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 搭建 Langfuse、LangSmith、Helicone 与 Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
