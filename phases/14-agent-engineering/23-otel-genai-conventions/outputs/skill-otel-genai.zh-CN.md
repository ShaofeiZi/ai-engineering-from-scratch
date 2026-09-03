---
name: otel-genai
description: 使用 OpenTelemetry GenAI 语义约定为智能体植入可观测性——invoke_agent、chat、tool_call span 携带正确属性，并支持可选的内容捕获。
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

给定一个智能体运行时，接入 OTel GenAI 语义约定。

产出：

1. 每次智能体运行产生一个 `invoke_agent` span。远程智能体服务使用 CLIENT 类型，进程内使用 INTERNAL 类型。名称：`invoke_agent {gen_ai.agent.name}`。
2. 每次 LLM 调用产生一个 `chat` span，携带 `gen_ai.operation.name=chat`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`。
3. 每次工具调用产生一个 `tool_call` span，携带 `gen_ai.tool.name`，并在适用时携带 `gen_ai.data_source.id`（RAG 语料库 / 记忆存储）。
4. 可选内容捕获：默认关闭；开启时，将输入/输出存储在外部，并在 span 上记录 `*.reference_id`。
5. 上下文传播：使用 W3C trace context 头，使多进程运行（Claude Agent SDK CLI 子进程）拼接为同一条 trace。

严格拒绝：

- 默认将完整 prompt/输出内联捕获。存在 PII 和密钥泄露风险；同时也违反规范。
- 缺失 `gen_ai.provider.name`。多 provider 仪表盘会因此失效。
- 孤立的 tool span。必须始终通过活动上下文设置父子关系。

拒绝规则：

- 如果运行时无法跨进程边界传播上下文，则拒绝。Claude Agent SDK + CLI 用户需要多进程 trace 拼接。
- 如果产品存在合规约束（HIPAA、GDPR），则拒绝内联内容捕获。仅允许带访问控制的外部存储。
- 如果后端未设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，则发出警告：属性名称可能在 collector 升级后发生变化。

输出：`tracer.py`、`attributes.py`、`content_store.py`、`README.md`，其中 README 说明 span 结构、稳定性 opt-in 以及内容捕获策略。以“延伸阅读”结尾，指向 Lesson 24（后端：Langfuse、Phoenix、Opik）或 Lesson 17（Claude Agent SDK trace-context 传播）。
