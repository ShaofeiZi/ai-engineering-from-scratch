---
name: mcp-server-designer
description: 设计无状态 MCP 2026-07-28 服务器，并明确发现、状态、传输和安全契约。
version: 2.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, stateless, tool-use]
---

给定一个领域（内部 API、数据库、文件源）以及将挂载该服务器的主机，输出：

1. 原语映射。哪些能力成为 `tools`（动作），哪些成为 `resources`（只读数据），哪些成为 `prompts`（用户触发的模板）。每个原语一行。
2. 发现契约。起草 `server/discover`，包含实现所支持的确切版本、能力、服务器身份、说明、`ttlMs` 与 `cacheScope`。
3. 请求契约。在每个请求的 `params._meta` 中要求字符串协议版本和对象型客户端能力。建议携带客户端身份。对于缺失或类型不正确的必需元数据，返回 Invalid Params（`-32602`）。仅当提供了服务器未实现的版本字符串时，才返回 `UnsupportedProtocolVersionError`（`-32022`）并附带 `data.supported` 和 `data.requested`。
4. 结果契约。为每个适用结果添加 `resultType`、服务器身份元数据、确定性列表排序及缓存策略。
5. MRTR 计划。仅对 `tools/call`、`resources/read` 或 `prompts/get` 使用 `input_required`。至少包含 `inputRequests` 或不透明的 `requestState` 之一；以新的 JSON-RPC ID 重试原方法，并在被要求时提供对应的输入响应，以及在存在时提供确切的状态值。
6. 状态计划。对于每个多调用工作流，定义一个由服务器铸造的不透明句柄，作为普通工具参数传递。不要将状态隐藏在连接或协议会话背后。
7. 传输与鉴权计划。选择 stdio 或 2026-07-28 Streamable HTTP POST 端点。对于 HTTP，定义 Origin 校验和每请求鉴权。要求 POST 请求携带 `MCP-Protocol-Version`，JSON-RPC 请求携带 `Mcp-Method`，且仅对 `tools/call`、`resources/read` 和 `prompts/get` 携带 `Mcp-Name`。被接受的通知 POST 返回 HTTP 202 且无响应体。
8. Schema 草案。为每个工具参数编写 JSON Schema，描述需针对模型选择进行调优，并对不可信输入设置显式边界。
9. 破坏性动作清单。为每个会变更状态的工具标记 `destructiveHint: true` 并要求人工审批。
10. 验证计划。覆盖以下场景：通知不产生 JSON-RPC 响应、畸形信封与请求 ID、元数据被拒、发现机制、确定性列表、版本不匹配、缓存字段、请求头与响应体不一致、鉴权、审批，以及一个提示词注入案例。

拒绝使用 `initialize`、`notifications/initialized`、`Mcp-Session-Id`、独立 HTTP GET、HTTP DELETE 或 `Last-Event-ID` 作为其现代路径的设计。仅允许在面向 2025-11-25 及之前协议版本的、明确隔离的适配器内部使用这些机制。不要向新实现中添加已废弃的 Roots、Sampling 或 Logging；兼容性支持必须明确标注，且 Roots 或 Sampling 的输入必须使用 MRTR。拒绝在未设鉴权、校验与审批路径的情况下写磁盘或调用外部 API 的服务器。
