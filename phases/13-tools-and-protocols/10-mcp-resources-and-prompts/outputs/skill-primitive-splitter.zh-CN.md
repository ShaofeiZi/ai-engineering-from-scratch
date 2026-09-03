---
name: primitive-splitter
description: 审查 MCP 服务器设计，并使用 2026-07-28 契约分离工具、资源、提示词、缓存和订阅。
version: 2.0.0
phase: 13
lesson: 10
tags: [mcp, resources, prompts, subscriptions, caching]
---

从消费者角度审查一个提议的 MCP 服务器。

产出：

1. 一个 `server/discover` 结果，通告修订版本 `2026-07-28` 以及确切的资源和提示词能力。
2. 一张包含 `name`、`chooser`、`primitive` 和 `reason` 的表格。
3. 稳定的资源 URI 方案以及任何有界资源模板。
4. 提示词名称、描述以及必需或可选参数。
5. 每个列表方法的确定性排序规则。
6. 针对每个可缓存结果的缓存策略，包含 `ttlMs` 和 `cacheScope`。
7. 一个针对需要更新的资源或列表变更的 `subscriptions/listen` 过滤器。
8. 一个返回 JSON-RPC `-32602` 的无效资源示例，加上一个返回 `-32022` 并附带 `supported` 和 `requested` 的不支持修订版本示例。

使用以下决策规则：

- 由模型选择的操作是工具。
- 主机可读的、URI 寻址的内容是资源。
- 由用户选择的消息工作流是提示词。
- 更新流通过 `subscriptions/listen` 由客户端打开。
- 监听请求 ID 成为 `io.modelcontextprotocol/subscriptionId`。
- 确认必须先于该订阅上的所有事件。
- 通知绝不会绕过对后续读取的授权。
- 即使客户端选择先调用其他方法，`server/discover` 也是强制性的。

在以下情况下拒绝设计：

- 列表因连接历史而变化。
- 私有结果被放入公共缓存。
- 资源 URI 在未经过解析、授权和边界检查的情况下被接受。
- 设计使用 `resources/subscribe` 或将订阅视为协议会话。
- 提示词被允许覆盖受信任的主机指令。

返回一页契约审查。以风险最高的原语、缓存或订阅错误以及最小修正作为结尾。
