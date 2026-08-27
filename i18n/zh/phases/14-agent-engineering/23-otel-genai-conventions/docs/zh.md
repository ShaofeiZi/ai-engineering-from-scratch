# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了代理遥测的标准 schema。span 名称、attributes 以及内容捕获规则开始在不同厂商之间收敛，因此无论你把 trace 送到 Datadog、Grafana、Jaeger 还是 Honeycomb，看到的 agent trace 都能表达同一件事。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测平台）
**Time:** 约 60 分钟

## 学习目标

- 说出 GenAI span 的三类：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT span 和 INTERNAL span，并理解各自适用场景。
- 列出顶层 GenAI attributes：provider name、request model、data-source ID。
- 解释内容捕获的契约：默认不采集、通过 `OTEL_SEMCONV_STABILITY_OPT_IN` 选择启用，并推荐采用 external reference。

## 问题

几乎每个厂商都会发明自己的一套 span 名称。结果就是运维团队不得不为不同框架单独做仪表板。OpenTelemetry 的 GenAI SIG 试图修正这件事：不是再造一个厂商格式，而是定义整个生态都能对齐的一套标准。

## 概念

### Span 类别

1. **Model / client spans.** 覆盖最原始的 LLM 调用，通常由 provider SDK（Anthropic、OpenAI、Bedrock）或框架里的模型适配器发出。
2. **Agent spans.** 包括 `create_agent`（代理构造时）和 `invoke_agent`（代理运行时）。
3. **Tool spans.** 每次工具调用发一个 span，并通过父子关系挂在 agent span 下面。

### Agent span 命名

- Span name：如果代理有名字，则使用 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span kind：
  - **CLIENT**：用于远程 agent service，例如 OpenAI Assistants API、Bedrock Agents。
  - **INTERNAL**：用于进程内 agent framework，例如 LangChain、CrewAI、本地 ReAct。

### 关键属性

- `gen_ai.provider.name`：例如 `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`：请求使用的模型 ID。
- `gen_ai.response.model`：实际解析后的模型，可能因为路由与请求值不同。
- `gen_ai.agent.name`：代理标识符。
- `gen_ai.operation.name`：例如 `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`：在 RAG 场景中，表示查询命中了哪个 corpus 或 store。

此外，Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 也分别有技术栈特定的 conventions。

### 内容采集

默认规则是：instrumentation **不应该**默认采集输入和输出内容。内容采集必须显式 opt-in，相关字段包括：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

更推荐的生产模式是：把内容本身存放在外部系统里，例如 S3 或你的日志存储，只在 span 上记录引用信息，比如 pointer ID，而不是把完整 prose 直接塞进 trace。这和 Lesson 27 里讲的内容投毒防御是一脉相承的：可观测性系统不应该顺手变成敏感数据扩散器。

### 稳定性

截至 2026 年 3 月，大部分 conventions 仍然处于 experimental 状态。要显式选择稳定预览通道，需要设置：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 已经可以把 GenAI attributes 原生映射到自己的 LLM Observability schema。其他后端，如 Grafana、Honeycomb、Jaeger，则通常直接消费这些原始 attributes。

### 这种模式常见的失败点

- **把完整 prompt 都写进 span。** 这样 trace 里会直接包含 PII、secret、客户数据，而运维又通常可以直接读取这些内容。更稳妥的做法是外部存储。
- **没有 `gen_ai.provider.name`。** 一旦系统是多 provider 混合运行，缺失归因字段会让仪表板彻底失真。
- **span 没有父链路。** 工具调用成了 orphaned spans，后续排障时只看到零散碎片。上下文传播必须始终正确。
- **没设置 stability opt-in。** 后端升级后字段名可能发生变化，导致你原有面板悄悄失效。

```figure
ae-genai-span-tree
```

## 动手构建

`code/main.py` 实现了一个符合 GenAI conventions 的 stdlib span emitter，包含：

- 带有 GenAI attribute schema 的 `Span`。
- 支持 `Tracer` 和 `start_span` 的嵌套上下文。
- 一个脚本化 agent run，会发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具调用对应的 tool spans，以及 LLM 调用的 `chat` spans。
- 一个内容捕获模式：把 prompts 存到外部存储，只在 spans 上记录引用 ID。

运行方式：

```
python3 code/main.py
```

输出会展示一棵包含全部必需 GenAI attributes 的 span tree，以及一个“external store”，用于展示 opt-in 内容引用是如何关联回去的。

## 如何使用

- **Datadog LLM Observability**：从 v1.37+ 起已支持对这些 attributes 做原生映射。
- **Langfuse / Phoenix / Opik**（Lesson 24）：适合直接接入现成 agent observability 生态。
- **Jaeger / Honeycomb / Grafana Tempo**：直接接收原始 OTel traces，再基于 GenAI attributes 自己搭面板。
- **Self-hosted**：运行带有 GenAI processor 的 OTel Collector。

## 交付成果

`outputs/skill-otel-genai.md` 用来把 OTel GenAI spans 接到现有 agent 上，并提供内容捕获默认策略以及 external-reference 存储方案。

## 练习

1. 给你在 Lesson 01 写的 ReAct loop 加上 `invoke_agent`（INTERNAL）和每个工具调用的 spans，然后发到一个 Jaeger 实例里。
2. 实现“references only”模式的内容捕获：prompt 存进 SQLite，span attributes 里只保留 row ID。
3. 阅读 `gen_ai.data_source.id` 的规范说明，并把它接入你 Lesson 09 的 Mem0 search。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证 collector 不会把你的属性名改掉。
5. 只依赖 GenAI attributes，做一个“哪些工具错误与哪些模型相关”的仪表板。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI SIG | "OpenTelemetry GenAI group" | 负责定义这套 schema 的 OTel 工作组 |
| invoke_agent | "Agent span" | 表示一次 agent run 的 span 名称 |
| CLIENT span | "Remote call" | 调用远程 agent service 时使用的 span |
| INTERNAL span | "In-process" | 进程内 agent run 对应的 span |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | 某次检索命中了哪个 corpus 或 store |
| Content capture | "Prompt logging" | 对 messages 的 opt-in 采集；生产上应外部存储 |
| Stability opt-in | "Preview mode" | 用环境变量固定 experimental conventions 的版本 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 官方规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认就会发出 GenAI spans
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内建 OTel spans
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — 支持 W3C trace context propagation
