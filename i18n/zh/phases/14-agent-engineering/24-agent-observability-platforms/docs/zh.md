# Agent 可观测平台：Langfuse、Phoenix、Opik

> 到了 2026 年，三大开源 agent observability 平台已经形成主流格局。Langfuse（MIT）每月 SDK 安装量超过 600 万，提供 tracing、prompt management、evals 与 session replay。Arize Phoenix（Elastic 2.0）更强调 agent 专属评估、RAG relevancy 与 OpenInference 自动埋点。Comet Opik（Apache 2.0）则主打自动 prompt optimization、guardrails 与基于 LLM-judge 的幻觉检测。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 23（OTel GenAI）
**Time:** 约 45 分钟

## 学习目标

- 说出三大主流开源 agent observability 平台及其许可证。
- 区分它们各自最擅长的方向：Langfuse（prompt management + sessions）、Phoenix（RAG + auto-instrumentation）、Opik（optimization + guardrails）。
- 解释为什么到 2026 年有 89% 的组织已经部署 agent observability。
- 实现一个从 trace 到 dashboard 的 stdlib 管线，并加入 LLM-judge evaluation。

## 问题

OTel GenAI（Lesson 23）只解决了 schema 问题。你依然需要一个平台来 ingest spans、运行评估、保存 prompt 版本，并把回归问题可视化出来。当前这三家开源平台，实际上分别押注在 agent 生命周期里的不同环节。

## 概念

### Langfuse（MIT）

- 每月 6M+ SDK installs，GitHub 19k+ stars。
- 主要能力：tracing、带版本管理与 playground 的 prompt management、evaluations（LLM-as-judge、用户反馈、自定义评估）、session replay。
- 到 2025 年 6 月，原先商业化的模块，如 LLM-as-a-judge、annotation queues、prompt experiments、Playground，已经以 MIT 许可证开源。
- 最强项：从 tracing 到 prompt iteration 的端到端闭环。

### Arize Phoenix（Elastic License 2.0）

- 更强调 agent-specific evaluation：trace clustering、anomaly detection、面向 RAG 的 retrieval relevancy。
- 原生支持 OpenInference auto-instrumentation。
- 可以与托管版 Arize AX 配合，用于生产环境。
- 不负责 prompt versioning，更像是与其他平台并行使用的 drift / behavioral regression 分析工具。
- 最强项：RAG relevancy、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 支持通过 A/B experiments 自动做 prompt optimization。
- 提供 guardrails，例如 PII redaction、topic constraints。
- 支持 LLM-judge hallucination detection。
- 按照 Comet 自己公布的测量结果，Opik 的 logs + evals 用时 23.44s，而 Langfuse 为 327.15s，大约有 14 倍差距。不过这类 vendor benchmark 更适合作为方向参考，而不是绝对结论。
- 最强项：优化闭环、自动化实验、guardrail enforcement。

### 行业数据

根据 Maxim 在 2026 年的 field analysis，89% 的组织已经具备 agent observability；同时，质量问题仍然是生产落地的头号阻碍，有 32% 的受访者明确提到这一点。

### 如何选择

| 需求 | 选择 |
|------|------|
| 提供提示词管理的一体化方案 | Langfuse |
| 深度 RAG 评估与漂移检测 | Phoenix |
| 自动优化与护栏 | Opik |
| 开放许可证，不采用 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| 集成 Datadog / New Relic | 任意一个——它们都能导出 OTel 数据 |

### 这种模式常见的失败点

- **没有 eval strategy。** 只有 tracing 没有 evaluation，本质上只是更昂贵的日志系统。
- **自己手搓 LLM-judge，但没有 grounding。** Lesson 05 里的 CRITIC 模式在这里仍然适用，judge 需要外部工具做事实核验。
- **prompt version 没和 traces 绑定。** 一旦生产环境出现回归，你就没法追溯到底是哪版 prompt 引入了问题。

```figure
wb-trace-ingest
```

## 动手构建

`code/main.py` 实现了一个 stdlib trace collector + LLM-judge evaluator，包含：

- ingest GenAI 风格的 spans。
- 按 session 聚合，并标记失败运行，例如 guardrail trips、low-confidence evals。
- 一个脚本化的 LLM-judge，会按 rubric 给 agent response 打分。
- 一个类似 dashboard 的摘要视图：failure rate、top failure reasons、eval score distribution。

运行方式：

```
python3 code/main.py
```

输出会给出每个 session 的 eval 分数和失败分类，大致对应 Langfuse、Phoenix、Opik 这类平台实际会展示的信息。

## 如何使用

- **Langfuse**：可 self-hosted，也可用 cloud；通过 OTel 或官方 SDK 接入。
- **Arize Phoenix**：支持 self-hosted；适合直接利用 OpenInference 自动埋点。
- **Comet Opik**：可 self-hosted，也可用 cloud；适合做自动优化闭环。
- **Datadog LLM Observability**：适合已经大量使用 Datadog 的混合 ops + ML 团队。

## 交付成果

`outputs/skill-obs-platform-wiring.md` 用来帮助你在现有 agent 中选定一个平台，并接入 traces、evals 与 prompt versions。

## 练习

1. 把一周的 OTel traces 导出到 Langfuse cloud（free tier）。哪些 sessions 失败了？原因是什么？
2. 针对你的业务领域写一个 LLM-judge rubric，例如 factual correctness、tone、scope adherence，并在 50 条 traces 上测试。
3. 对比 Langfuse 的 prompt versioning 与 Phoenix 的 trace clustering。哪一个能更快告诉你到底哪里坏了？
4. 阅读 Opik 的 guardrail 文档，给你的一个 agent run 接入 PII redaction guardrail。
5. 在你自己的语料上 benchmark 这三者。忽略厂商公开数字，测你自己的结果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Tracing | "Spans collector" | 接收 OTel / SDK spans，并按 session 建索引 |
| Prompt management | "Prompt CMS" | 与 traces 绑定的版本化 prompts |
| LLM-as-judge | "Automated eval" | 用独立 LLM 按 rubric 评估 agent output |
| Session replay | "Trace playback" | 为了调试，逐步回放过去的运行过程 |
| RAG relevancy | "Retrieval quality" | 检索出来的上下文是否真的匹配查询 |
| Trace clustering | "Behavioral grouping" | 对相似运行聚类，用于检测漂移 |
| Guardrail enforcement | "Policy at log time" | 对日志内容做 PII / toxicity / scope 检查 |

## 延伸阅读

- [Langfuse docs](https://langfuse.com/) — tracing、evals、prompt management
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation、drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三个平台共同消费的 schema
