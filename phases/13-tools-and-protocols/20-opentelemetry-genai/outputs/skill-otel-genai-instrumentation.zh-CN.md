---
name: otel-genai-instrumentation
description: 为智能体代码库制定端到端发出 OTel GenAI span 的插桩方案。
version: 1.0.0
phase: 13
lesson: 19
tags: [otel, observability, gen-ai, tracing]
---

给定一个智能体代码库（LLM 调用、工具调度、MCP 客户端、子智能体），制定一份 OTel GenAI 插桩方案。

需要产出：

1. Span 层级结构。根 span 为 `agent.invoke_agent`（INTERNAL），子 span 包括：`llm.chat`（CLIENT）、`tool.execute`（INTERNAL）、`mcp.call`（CLIENT）、`subagent.invoke`（INTERNAL）。
2. 每个 span 的属性清单。`gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.usage.*`、`gen_ai.tool.name`、`gen_ai.agent.name`。
3. 传播规则。在每次远程调用上注入 W3C traceparent；对于 MCP stdio，使用 `_meta.traceparent` 作为过渡字段。
4. 内容捕获策略。默认关闭；记录启用所需的环境变量；说明 PII 风险。
5. 导出器选择。Jaeger / Tempo / Langfuse / Phoenix / Datadog / Honeycomb；以 OTLP 作为传输协议。

硬性拒绝：
- 任何缺少跨 MCP 或子智能体边界的链路传播的方案。
- 任何默认开启内容捕获的方案。会泄漏提示词和 PII。
- 任何在没有 `gen_ai.` 前缀或显式厂商前缀的情况下发出任意自定义属性的方案。

拒绝规则：
- 如果代码库使用的框架内置了 OTel 自动插桩（Pydantic AI、LangGraph、AgentOps），优先推荐框架钩子。
- 如果导出器后端是本地部署且团队没有 SRE 支持，推荐使用托管后端。
- 如果用户要求在生产环境中捕获内容用于调试，在没有明确的合规同意策略和 PII 脱敏流水线的情况下予以拒绝。

输出：一份单页方案，包含 span 层级结构、每个 span 的属性清单、传播规则、内容捕获策略和导出器选择。最后给出需要告警的首要指标（通常为 p95 `gen_ai.client.operation.duration`）。
