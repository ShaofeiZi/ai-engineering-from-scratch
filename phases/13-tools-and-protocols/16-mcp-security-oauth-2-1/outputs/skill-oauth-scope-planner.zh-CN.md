---
name: oauth-scope-planner
description: 设计 MCP 2026-07-28 授权方案，涵盖 CIMD、颁发方隔离、资源指示符与逐步升级的 scope。
version: 2.0.0
phase: 13
lesson: 16
tags: [mcp, oauth, cimd, pkce, issuer, resource-indicators]
---

给定一个远程 HTTP MCP 服务器及其工具列表，设计完整的授权边界。

## 必需输入

- 规范化的 MCP 资源 URI 及受保护资源元数据位置。
- 允许的授权服务器颁发方。
- 客户端运行时：原生或 Web，以及精确的重定向 URI。
- 工具到 scope 的映射及关键性操作。
- 令牌、刷新令牌及凭据存储约束。
- 缺少 CIMD 的遗留授权服务器（如有）。

## 产出内容

1. 资源元数据。起草 RFC 9728 的 `resource`、`authorization_servers` 和 `scopes_supported`。保留 well-known 段之后的资源路径，例如对于 `https://notes.example.com/mcp`，使用 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`。
2. 颁发方策略。明确允许的颁发方、元数据校验、变更处理以及 RFC 9207 的 `iss` 比较。
3. 注册。可用时优先使用预注册，否则优先使用客户端 ID 元数据文档（Client ID Metadata Document）。其带有路径的 HTTPS URL 即为 `client_id`；要求精确的重定向 URI，并将展示元数据视为不可信。此处的 `application_type` 为可选项。
4. DCR 回退。如有必要，将其标记为已弃用，声明 `application_type`，并定义允许回退的精确条件。在发生通用 CIMD 安全失败后不得降级。
5. 凭据键。将预注册和 DCR 凭据按颁发方存储，令牌按 `(issuer, resource)` 存储。禁止跨颁发方复用。说明自托管的 CIMD URL 是可移植的，当受信任的颁发方变更时无需重新进行 DCR 注册。
6. PKCE 流程。要求 S256、精确的重定向 URI、授权响应颁发方校验，以及在授权请求和令牌请求中使用相同的 resource。
7. Scope 模型。将每个工具映射到其最小 scope。将当前 `WWW-Authenticate` 的 scope 质询视为权威。
8. 逐步升级体验。识别所需的额外 scope、用户说明、同意点、新的授权流程，以及使用全新 MCP request id 的重试。
9. 资源服务器检查。实现已公告的 `tools/list`，包含有效的对象根 schema、确定性排序、结果类型、服务器身份及缓存提示。在工具调度前校验颁发方、受众、过期时间、scope、当前 MCP 头部及请求元数据。
10. 令牌管理。仅使用 Bearer 头部，不得在查询参数中传递令牌，不得透传令牌，机密客户端的刷新令牌安全存储，以及轮换计划。
11. 错误契约。在每个 JSON-RPC 错误信封中保留每个 request id，包括 OAuth 失败。要求在版本支持检查 HTTP 400 `-32022` 之前，先执行头部不匹配的 HTTP 400 `-32020`，提供精确的支持数据和请求数据，对未知方法返回 HTTP 404 `-32601`，对已接受的通知返回 202 及空响应体。
12. 传输边界。将已解析的请求体示例标记为进程内协议模型，并将其附加到第 09 课的完整 Streamable HTTP 适配器，用于 JSON Content-Type 以及 JSON 加 SSE 的 Accept 校验。

## 严格拒绝

- 将 DCR 作为首选的新注册机制。
- 不带 `application_type` 的 DCR。
- 在颁发方变更后复用颁发方生成的注册凭据、访问令牌或刷新令牌。自托管的 CIMD URL 是可移植的例外，而非颁发方生成的密钥。
- 在比较之前对授权响应的 `iss` 进行规范化处理。
- 缺少 PKCE S256，或在授权请求和令牌请求中缺少 `resource`。
- 接受面向其他受众的令牌或将 MCP 令牌转发至下游。
- 使用 `clientInfo`、`serverInfo`、能力声明或已移除的协议会话作为身份认证。
- 仅为模仿远程 HTTP 而在本地 stdio 上添加 OAuth。
- 在构造 RFC 9728 元数据 URL 时丢弃受保护资源路径。
- 对 MCP 请求错误返回纯文本或临时构造的对象，而非带有相同 id 的 JSON-RPC 信封。

## 输出格式

返回名为 Resource、Issuers、Enrollment、Credential Store、PKCE Flow、Scope Matrix、Step-Up、Server Validation、Token Hygiene 和 Compatibility 的章节。最后以触发颁发方审查、且对于颁发方生成的客户端需要重新注册的确切事件结尾。
