---
name: sampling-loop-designer
description: 将模型辅助的 MCP 工具迁移至直接推理或无状态 2026-07-28 MRTR，并采用有界的兼容性 Sampling。
version: 2.0.0
phase: 13
lesson: 11
tags: [mcp, mrtr, sampling, stateless, migration]
---

为面向协议版本 `2026-07-28` 的 MCP 服务器设计模型辅助行为。

首先做一个决策：服务器能否与模型提供商直接集成？Sampling 对于新设计已被废弃。除非将使用客户端的模型和凭证作为明确的产品需求，否则应优先采用直接集成。

产出：

1. 架构决策。选择直接推理或兼容性 Sampling，并说明理由。
2. 发现契约。展示 `server/discover`，包含精确的 `supportedVersions`、所通告的能力、`ttlMs` 和 `cacheScope`。如果通告了工具，则必须包含确定性的 `tools/list` 描述符，且具有有效的对象类型 `inputSchema`、`resultType: "complete"`、服务器身份元数据以及缓存提示。
3. 请求信封。在每次请求的 `_meta` 中包含协议版本和客户端能力。当版本缺失或非字符串时使用 `-32602`；当版本不受支持时使用 `-32022` 并附带精确的 `supported` 和 `requested` 数据；当 Sampling 缺失时使用 `-32021` 并附带 `requiredCapabilities` 对象。将客户端身份元数据仅视为信息性内容。对于无 id 的通知，绝不发出 JSON-RPC 响应；已接受的 HTTP 通知返回 `202` 且无响应体。
4. 轮次表。对于每个 MRTR 轮次，标明 `inputRequests` 键、内嵌的请求方法、预期响应模式、校验逻辑和预算。
5. 重试契约。要求原始方法和参数、全新的 JSON-RPC id、当前轮次的 `inputResponses`，以及逐字节精确的 `requestState`。
6. 状态保护。将 HMAC 或认证加密绑定到已认证主体、方法、参数摘要、阶段和较短的有效期。
7. 安全策略。定义审批、最大轮次、token 和字节限制、响应校验、日志记录和拒绝行为。
8. 移除计划。如果保留 Sampling，则命名以直接集成替换它的条件和日期。

硬性拒绝：

- 在没有文档化需求的情况下采用已废弃 Sampling 的新设计。
- 发送 `sampling/createMessage` 作为实时服务器到客户端请求的 2026-07-28 服务器。
- 任何对 `initialize`、`notifications/initialized`、`Mcp-Session-Id` 或隐藏协议会话状态的使用。
- 影响授权、资源访问或业务逻辑的未签名 `requestState`。
- 复用原始 JSON-RPC id 或更改原始参数的重试。
- 没有能力检查、审批策略、校验和硬性轮次上限的客户端模型循环。
- `includeContext: "allServers"` 或隐式跨服务器上下文。

拒绝规则：

- 拒绝隐蔽的模型调用，或任何向用户隐藏服务器意图的设计。
- 拒绝将模型输出作为身份、授权或用户同意的证明。
- 当一次确定性的工具调用即可满足需求时，拒绝多轮设计。
- 拒绝将客户端和服务器元数据称为已认证身份。

输出一页式架构，包含决策、通信流、轮次表、签名状态内容、安全预算、失败场景和迁移计划。以结论结尾：`direct inference`、`temporary MRTR compatibility` 或 `no model required`。
