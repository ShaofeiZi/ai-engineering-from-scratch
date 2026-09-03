---
name: elicitation-form-designer
description: 设计显式资源作用域与无状态的 MCP 2026-07-28 采集机制，包含授权、安全表单与签名重试状态。
version: 2.0.0
phase: 13
lesson: 12
tags: [mcp, elicitation, mrtr, scope, authorization]
---

为面向协议修订版 `2026-07-28` 的 MCP 操作设计一个用户输入步骤。

产出内容：

1. 作用域契约。将工作区、目录或资源 URI 置于可见的工具参数或服务器配置中。说明哪些已认证主体可以使用它。
2. 边界检查。定义 URI 规范化、路径组件包含关系、符号链接策略以及操作系统沙箱。
3. 触发条件。指明需要用户输入的确切歧义、确认或外部交互。
4. 发现与能力门控。从 `server/discover` 返回确切的 `supportedVersions`、capabilities、`ttlMs` 和 `cacheScope`。如果公告了工具，则包含强制性的确定性 `tools/list` 描述符，其中带有有效的对象 `inputSchema`、服务器身份元数据和缓存提示。将 `elicitation: {}` 和显式的 `elicitation.form` 视为表单支持。对于缺失或仅 URL 的支持，以 `-32021` 和 `data.requiredCapabilities.elicitation.form` 拒绝；对于不支持的版本，以 `-32022` 和确切的 `supported` 与 `requested` 数据拒绝。
5. MRTR 结果。返回 `resultType: "input_required"`，附带稳定的 `inputRequests` 键和 `elicitation/create` 请求。
6. 交互设计。对于表单模式，提供简洁消息和受限的扁平 schema。对于 URL 模式，展示 HTTPS 目标和带外完成规则。
7. 重试契约。要求全新的 JSON-RPC id、原始方法与参数、当前 `inputResponses`、每请求 `_meta` 以及精确的 `requestState` 回显。
   无 id 的通知永远不会收到 JSON-RPC result 或 error；被接受的 Streamable HTTP 通知收到 `202` 且无正文。
8. 分支处理。将 `accept`、`decline` 和 `cancel` 映射到不同的安全结果。
9. 状态保护。将 HMAC 或认证加密绑定到已认证主体、原始参数摘要、候选集、操作阶段、过期时间和一次性 nonce。在由每个处理程序实例共享的、有界且按 TTL 修剪的重放存储中原子地消费该 nonce。
10. 最终再验证。在变更前立即重新检查授权、实时记录状态和包含关系。

硬性拒绝：

- 将已弃用的 Roots 用作授权、包含或沙箱。
- 在新的 2026-07-28 设计中使用 `roots/list` 或 `notifications/roots/list_changed`。
- 发送反向的 `elicitation/create` 请求，而不是通过 MRTR 返回它。
- 在表单模式下收集密码、API 密钥、访问令牌或支付凭证。
- 发送当前每请求能力中不存在的采集模式。
- 将 `clientInfo` 视为已认证的用户身份。
- 在已验证的接受和最终授权检查之前执行破坏性操作。
- 携带候选者或权限相关数据的未签名 `requestState`。

拒绝规则：

- 在明确拒绝后拒绝重复提示。
- 对于服务器无需用户即可推导或验证的值，拒绝采集。
- 拒绝包含凭证、用户密钥或预认证 bearer 值的 URL。
- 拒绝使用隐藏协议会话状态、`initialize` 或 `Mcp-Session-Id` 的请求。

输出一页式设计，包含作用域、授权、包含、交互模式、schema 或 URL、MRTR 传输结构、状态字段、响应分支、重放策略和最终再验证清单。
