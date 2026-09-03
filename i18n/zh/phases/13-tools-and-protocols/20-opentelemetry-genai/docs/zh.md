# OpenTelemetry GenAI——端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要用一条 trace 串起整个过程。OpenTelemetry GenAI 语义约定（v1.37 及更高版本中的稳定属性）是 2026 年的标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 均提供原生支持。本课将介绍必需属性，梳理 span 层级（智能体 → LLM → 工具），并交付一个仅依赖标准库的 span 发射器，可接入任意 OTel exporter。

**Type:** 构建
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** 第 13 阶段 · 第 07 课（MCP 服务器）、第 13 阶段 · 第 08 课（MCP 客户端）
**Time:** 约 75 分钟

## 学习目标

- 说出 LLM span 和工具执行 span 必需的 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用与 MCP 客户端分发的 trace 层级。
- 判断哪些内容需要主动选择采集，哪些内容默认应脱敏。
- 无需重写工具代码，即可将 span 发送到本地 collector（Jaeger、Langfuse）。

## 问题

来看一个 2026 年 2 月的调试案例：用户反馈“我的智能体有时要 30 秒才响应，有时只要 3 秒”。系统没有 trace。日志记录了 LLM 调用，却没有工具分发、MCP 服务器往返或子智能体的信息。你只能猜测。最终才发现：某个 MCP 服务器偶尔会在冷启动时卡住。

缺少端到端追踪，就无法快速定位这种问题。OTel GenAI 正是为此而生。

这些约定由 OpenTelemetry semantic-conventions 工作组在 2025 至 2026 年间逐步稳定下来。它们定义了一组稳定的属性名，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析同一种 span。只需插桩一次，就能把数据发送到任何后端。

## 概念

### Span 层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个层级共享同一个 trace id。各个 span id 用来表达父子关系。

### 必需属性

根据 2025–2026 年的 semconv：

- `gen_ai.operation.name`——`"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name`——`"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model`——请求中指定的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model`——实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id`——用于关联的提供商响应 ID。

对于工具 span：

- `gen_ai.tool.name`——工具标识符。
- `gen_ai.tool.call.id`——本次具体调用的 ID。
- `gen_ai.tool.description`——工具描述（可选）。

对于智能体 span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### SpanKind

- 对跨进程边界的调用（LLM 提供商、MCP 服务器）使用 `SpanKind.CLIENT`。
- 对智能体自身的循环步骤和工具执行使用 `SpanKind.INTERNAL`。

### 主动选择内容采集

默认情况下，span 只携带指标和时序信息，不包含 prompt 或 completion。大载荷与 PII 默认关闭。若要包含内容，需要设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及具体的内容采集环境变量。在生产环境启用前务必仔细审查。

### Span 上的 Event

可以将 token 级事件添加为 span event：

- `gen_ai.content.prompt`——输入消息。
- `gen_ai.content.completion`——输出消息。
- `gen_ai.content.tool_call`——记录下来的工具调用。

事件在同一个 span 内按时间排序，可用于细粒度回放。

### Exporter

OTel span 可以导出到：

- **Jaeger / Tempo。** 开源，可部署在本地。
- **Langfuse。** 专注于 LLM 可观测性，可视化 token 用量。
- **Arize Phoenix。** 将评估与追踪结合起来。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 面向列式数据，便于查询。

它们都使用 OTLP 这一报文格式。你的代码无需关心具体后端。

### 跨 MCP 传播

MCP 客户端调用服务器时，应把 W3C traceparent header 注入请求。Streamable HTTP 支持标准 header。Stdio 无法原生携带 HTTP header；该规范的 2026 年路线图讨论了在 JSON-RPC 调用中增加 `_meta.traceparent` 字段。

在这项能力正式发布前：手动把 traceparent 放进每个请求的 `_meta`。服务器据此记录 trace id。

### 指标

除 span 外，GenAI semconv 还定义了以下指标：

- `gen_ai.client.token.usage`——直方图。
- `gen_ai.client.operation.duration`——直方图。
- `gen_ai.tool.execution.duration`——直方图。

这些指标适合构建不需要逐次调用明细的仪表盘。

### AgentOps 层

AgentOps（创立于 2024 年）专注于 GenAI 可观测性。它封装了常见框架（LangGraph、Pydantic AI、CrewAI），能够自动发出 OTel span。如果技术栈使用受支持的框架，这种方式很方便；否则应手动插桩。

```figure
t3-span-waterfall
```

## 使用它

`code/main.py` 会针对一个调用 LLM、分发两个工具并完成一次 MCP 往返的智能体，把符合 OTel 结构的 span 输出到 stdout（采用类似 OTLP-JSON 的格式）。示例不接入真实 exporter——本课关注 span 的形状与属性集合。你可以把输出粘贴到兼容 OTLP 的查看器中，也可以直接阅读。

阅读输出时请重点观察：

- 所有 span 共享同一个 trace id。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性均已填充。
- 内容采集默认关闭；其中一个场景会通过环境变量将其开启。

## 交付它

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能会生成插桩方案：应该在哪里添加 span、填充哪些属性，以及将哪些 exporter 作为目标。

## 练习

1. 运行 `code/main.py`。统计 span 数量，并判断哪些是 CLIENT、哪些是 INTERNAL。

2. 打开内容采集（环境变量），确认 `gen_ai.content.prompt` 与 `gen_ai.content.completion` event 出现。思考这对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并为每次调用发出一个直方图样本。

4. 将父智能体 span 的 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段中。验证 MCP 服务器能够看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。找出一个规范列出、但本课代码没有发出的属性，并将它添加进去。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| OTel | “OpenTelemetry” | trace、metric 和 log 的开放标准 |
| GenAI semconv | “GenAI 语义约定” | LLM / 工具 / 智能体 span 使用的稳定属性名 |
| `gen_ai.*` | “属性命名空间” | 所有 GenAI 属性共用此前缀 |
| Span | “有耗时的操作” | 有开始、结束和属性的一项工作 |
| Trace | “跨 span 的祖先关系” | 共享同一 trace id 的 span 树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 表示 span 方向的提示信息 |
| OTLP | “OpenTelemetry Line Protocol” | 导出器使用的报文格式 |
| Opt-in content | “采集 prompt / completion” | 默认关闭；通过环境变量开启 |
| traceparent | “W3C header” | 跨服务传播 trace 上下文 |
| Exporter | “特定后端的发送组件” | 将 span 发送到 Jaeger / Datadog 等后端的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——GenAI span、metric 与 event 的规范约定
- [OpenTelemetry — GenAI span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)——LLM 与工具执行 span 的属性列表
- [OpenTelemetry — GenAI 智能体 span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)——智能体层的 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI span](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)——托管在 GitHub 上的真相源
- [Datadog — LLM OTel 语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)——生产集成实战指南
