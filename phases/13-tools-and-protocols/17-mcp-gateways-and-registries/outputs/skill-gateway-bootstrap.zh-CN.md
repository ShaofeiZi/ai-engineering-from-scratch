---
name: gateway-bootstrap
description: 设计一个无状态的 MCP 2026-07-28 网关，包含注册准入、策略、路由和兼容性边界。
version: 2.0.0
phase: 13
lesson: 17
tags: [mcp, gateway, stateless, registry, rbac, subscriptions, tasks]
---

给定客户端、后端、授权要求和合规约束后，产出一个网关设计。

## Required inputs

- 公开的网关资源 URI、接受的协议版本以及传输方式。
- 已认证的主体和角色模型。
- 后端端点、签发方、资源、注册记录、发布者证据以及已批准的描述符。
- 工具可见性、参数策略、成本类别以及数据敏感度。
- 流式传输、变更通知、MRTR 和 Tasks 需求。
- 审计、留存、追踪和脱敏要求。

## Produce

1. 无状态入口。一个 POST 端点，按请求携带版本和能力，匹配 method 和 name 头部，JSON 或请求级 SSE，并对现代 GET 和 DELETE 返回 405。在版本支持之前校验头部一致性。指定 HTTP 400 `-32020`、HTTP 400 `-32022`（附带确切的支持和请求数据）、HTTP 404 `-32601`、可选的错误数据序列化，以及 202 空正文通知处理。
2. 发现计划。实现网关的 `server/discover`，发现每个后端，仅暴露安全的端到端能力交集，并包含当前的 `resultType`、`ttlMs`、`cacheScope` 以及服务器身份元数据。
3. 准入表。验证官方 Registry `server.json` 的发布形态，并将 `com.example/*` 风格的名称与安全准入分开校验。对每个后端，将记录关联到外部已验证的发布者命名空间、来源出处、端点、版本策略、描述符摘要、签发方、资源、批准和过期状态。
4. 命名空间映射。为每个后端工具赋予稳定的限定公开名称，并在每个 `tools/list` 描述符上保留有效的 object-root `inputSchema`。拒绝按顺序消歧的冲突。
5. 授权矩阵。将主体和角色映射到公开工具、资源、参数和作用域。保持外部凭据和后端凭据相互独立且与签发方绑定。
6. 转发契约。构建全新的自包含后端请求，仅声明中介化的客户端能力，校验后端结果，并保留追踪关联。
7. 缓存计划。使依赖主体的发现和列表私有化。设置有限的 TTL 和失效行为。
8. 速率与审计策略。按主体、签发方、资源、工具、成本类别和时间设定限流键。对凭据和不必要的敏感参数进行脱敏。
9. 交互路由。描述请求级 SSE、`subscriptions/listen` 确认和重连行为、字节精确的 MRTR 状态转发，以及按 `Mcp-Name` 中的 task id 进行 Tasks 路由。
10. 传输适配器。如果网关接收的是已解析的请求和头部，则将其标记为进程内协议模型，并将其关联到第 09 课的 JSON Content-Type 和 JSON 加 SSE Accept 强制执行。
11. 兼容性适配器。将较旧的初始化、会话 id、GET 流、资源订阅和实验性 task 方法隔离在现代网关核心之外。

## Hard rejects

- 将会话亲和性、会话存储或会话 id 重写作为 2026-07-28 的必需项。
- 在没有准入证据的情况下信任注册记录的存在或显示名称。
- 静默的工具冲突或在未重新批准的情况下更新描述符锁定。
- 在后端复用外部 bearer token，或在另一个签发方或资源处复用后端 token。
- 对主体过滤后的列表进行公开缓存。
- 独立的现代 GET 事件流、Last-Event-ID 重放或资源订阅方法。
- 新增的 `tasks/list` 或 `tasks/result` 行为。
- 仅以已移除的协议会话为键的速率限制。
- 在 `server.json` 内部编造安全校验，而非使用独立的已验证准入和来源出处状态。
- 带命名空间的工具描述符省略 `inputSchema`。

## Output format

返回名为 Ingress、Discovery、Admission、Namespace Map、Authorization、Forwarding、Cache、Rate Limits、Audit、Interactions 和 Legacy Adapter 的章节。以需要最强验收测试的那条路由结尾。
