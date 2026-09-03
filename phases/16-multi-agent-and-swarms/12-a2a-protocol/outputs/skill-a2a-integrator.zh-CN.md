---
name: a2a-integrator
description: 设计两个智能体之间的 A2A 集成——Agent Card、任务 schema、认证、流式或轮询。
version: 1.0.0
phase: 16
lesson: 12
tags: [multi-agent, a2a, protocol, interoperability, google]
---

给定两个需要互操作的智能体系统，生成 A2A 集成方案：Agent Card 内容、任务 schema、认证、传输模式。

需产出：

1. **Agent Card。** 名称、版本、技能、端点、支持的模态（text、structured、image、audio、video）、protocol_version、认证声明。
2. **每个技能的任务 schema。** 输入 JSON schema + 产物 JSON schema。需明确——客户端会进行校验。
3. **认证选择。** Bearer token（OAuth2 或不透明令牌）、mTLS 或签名请求。根据威胁模型（公网、VPC、混合）给出理由。
4. **传输模式。** 轮询 vs SSE 流式 vs webhook 回调。长时运行或进度密集型任务用流式；短任务用轮询。
5. **速率限制。** 每客户端和每任务的限制。防止滥用。
6. **幂等性。** 处理重复 `POST /tasks` 请求的策略（客户端 task-key、服务端去重）。
7. **失败处理。** 超出 `failed` 状态的任务状态（可重试 vs 致命）、死信策略、错误产物 schema。
8. **MCP 与 A2A 的划分。** 若远端智能体内部使用 MCP，注明哪些工具暴露给外部、哪些保留在内部。

硬性拒绝条件：

- 未声明协议版本的 Agent Card。
- 当用例需要结构化数据时使用自由文本的任务 schema。
- 公网部署中认证方式为 none。

拒绝规则：

- 若两个智能体运行在同一进程内，拒绝 A2A 并推荐直接使用 Python/JS 调用。A2A 适用于跨系统边界场景。
- 若延迟要求往返低于 100ms，拒绝 A2A 并推荐使用共享 schema 的直接 RPC。
- 若远端智能体未声明 Agent Card，拒绝集成并推荐先发布一张。

输出：一页集成简报。以内联粘贴的 Agent Card JSON 收尾，便于工程团队直接放入 `/.well-known/agent.json`。
