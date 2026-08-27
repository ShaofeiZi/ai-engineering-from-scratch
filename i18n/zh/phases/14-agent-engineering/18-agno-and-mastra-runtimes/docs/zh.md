# 生产级 Agent Runtime：快速实例化与类型化工作流

> 生产级 agent runtime 会优化那些原型框架常常忽略的东西：实例化成本、类型化工作流接口，以及可直接服务化的后端。到 2026 年，常见的一组对照是：Agno（Python）主打微秒级 agent 实例化与无状态 FastAPI 后端；Mastra 则构建在 Vercel AI SDK 之上，提供 agents、tools、workflows、统一模型路由和组合式存储。

**Type:** 学习
**Languages:** Python, TypeScript
**Prerequisites:** 第 14 阶段 · 01（Agent Loop），第 14 阶段 · 13（LangGraph）
**Time:** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标，并理解它们在什么场景下真正重要。
- 说出 Mastra 的三个基本原语：Agents、Tools、Workflows，以及它支持的 server adapters。
- 解释为什么“无状态、按 session 划分的 FastAPI 后端”是 Agno 推荐的生产路径。
- 针对具体技术栈做出选择：Agno 还是 Mastra（Python-first vs TypeScript-first）。

## 问题

LangGraph、AutoGen、CrewAI 都偏框架化。那些只想要“agent loop，够快，而且能直接运行在我现有 runtime 里”的团队，通常会去看 Agno（Python）或 Mastra（TypeScript）。两者都用一部分框架原语的让渡，换来更高的原始速度，以及和周边技术栈更紧的贴合度。

## 概念

### Agno

- Python runtime，前身是 Phi-data。
- 官方口号是：“No graphs, chains, or convoluted patterns — just pure python.”
- 文档中给出的性能目标包括：约 2μs 的 agent 实例化、每个 agent 约 3.75 KiB 内存占用，以及约 23 个模型提供方。
- 推荐的生产路径是：无状态、按 session 划分的 FastAPI backend。每次请求都启动一个全新的 agent；session 状态保存在数据库中。
- 原生支持多模态（text、image、audio、video、file）与 agentic RAG。

这些速度目标在“每秒有成千上万个短命 agent”的场景里才真正关键，比如聊天聚合入口或评估流水线。如果一个 agent 每次都要跑 10 分钟，那它们的重要性就会明显下降。

### Mastra

- TypeScript，建立在 Vercel AI SDK 之上。
- 三个核心原语：**Agents**、**Tools**（带 Zod 类型）、**Workflows**。
- 统一模型路由器：跨 94 个 provider 提供 3,300+ 个模型（2026 年 3 月数据）。
- Composite storage：memory、workflows、observability 可以分别写入不同后端；在大规模观测场景下推荐使用 ClickHouse。
- 采用 Apache 2.0，但源码中的 `ee/` 目录使用 source-available 的企业许可证。
- 支持 Express、Hono、Fastify、Koa 等 server adapters，并且对 Next.js 与 Astro 提供一等集成。
- 自带 Mastra Studio（localhost:4111）用于调试。
- 在 1.0 发布时（2026 年 1 月）拥有 22k+ GitHub stars 与 300k+ 周 npm 下载量。

### 定位

二者都不是在试图成为 LangGraph。它们的竞争点主要在于：

- **语言匹配度。** Agno 适合 Python-first 团队；Mastra 适合 TypeScript-first 团队。
- **Runtime ergonomics。** Agno 追求接近零开销；Mastra 则更深地融入 Vercel 生态。
- **可观测性。** 两者都能接入 Langfuse / Phoenix / Opik（Lesson 24），但 Mastra Studio 是第一方产品。

### 何时选择各自

- **Agno**：Python 后端、短生命周期 agent 很多、性能要求强、团队本来就用 FastAPI。
- **Mastra**：TypeScript 后端、部署在 Next.js / Vercel、需要统一多 provider 模型路由、偏好 Zod 类型工具。
- **LangGraph**（Lesson 13）：如果持久状态与显式图推理比原始速度更重要。
- **OpenAI / Claude Agent SDK**：如果你想直接采用 provider 已经产品化好的 agent 形态（Lessons 16–17）。

### 这种模式会出错的地方

- **为性能而性能。** 因为“2μs 很厉害”就去选 Agno，但你的负载实际上只是“每个请求调用一次很慢的 agent”。这时瓶颈根本不在 runtime 开销上。
- **生态锁定。** Mastra 带有明显的 Vercel 风格，这在 Vercel 上是优势，在别处可能就是负担。
- **企业许可证误解。** Mastra 的 `ee/` 目录并不是 Apache 2.0。如果你计划 fork，需要先把许可证读清楚。

```figure
wb-runtime-spawn
```

## 动手构建

这一课主要是做对比分析，没有一个单独的代码产物能同时把两个框架讲清楚。可以看 `code/main.py` 里的并排 toy 实现：同样是一个最小化的“运行 agent、流式输出、持久化 session”流程，各写了一次（一次采用 Agno 风格，一次采用 Mastra 风格）。

运行它:

```
python3 code/main.py
```

你会看到两条在结构上不同、但功能上等价的 trace。

## 如何使用

- **Agno**：适合既要速度又要 FastAPI 形态的 Python 后端。
- **Mastra**：适合有很多 provider、又需要 workflow primitives 的 TypeScript 后端。
- 两者都自带第一方 observability hooks，也都能接入 Langfuse。

## 交付成果

`outputs/skill-runtime-picker.md` 会根据技术栈、延迟预算与运维形态，在 Agno、Mastra、LangGraph 和 provider SDK 之间做选择。

## 练习

1. 读 Agno 的文档，把 stdlib ReAct loop（Lesson 01）移植到 Agno。哪些东西消失了？哪些东西还在？
2. 读 Mastra 的文档，再把同一个 loop 移植到 Mastra。工具类型这件事发生了什么变化（Zod vs nothing）？
3. 做一次 benchmark：测量你自己技术栈上的 agent 实例化延迟。Agno 的 2μs 对你的负载到底有没有意义？
4. 设计一次迁移：如果你一直在 Python 里跑 CrewAI，那么迁移到 Agno 时会断在哪里？
5. 阅读 Mastra `ee/` 的许可证条款。哪些限制会影响一个开源 fork？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agno | “Fast Python agents” | 无状态、按 session 划分的 agent runtime |
| Mastra | “TypeScript agents on Vercel AI SDK” | Agents + Tools + Workflows + Model Router |
| 统一模型路由器 | “多提供商访问” | 一个客户端统一访问 94 个 provider 的 3,300+ 模型 |
| Composite storage | “Multiple backends” | memory / workflows / observability 分别落到不同存储 |
| Mastra Studio | “Local debugger” | localhost:4111 上用于观察 agent 的 UI |
| Source-available | “Not OSS” | 允许阅读源码，但对商业使用有额外限制 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 性能目标与 FastAPI 集成
- [Mastra docs](https://mastra.ai/docs) — primitives、server adapters、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图方案的替代路径
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成中提到的 observability 对比
