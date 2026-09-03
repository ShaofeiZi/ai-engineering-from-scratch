---
name: mcp-auth-wiring
description: 设计 MCP 2026-07-28 授权，包含发行方绑定的注册、CIMD、受保护资源元数据、JWKS 刷新、受众锁定以及按请求校验。
version: 2.0.0
phase: 13
lesson: 18
tags: [mcp, oauth, cimd, dcr, jwks, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728, rfc9207]
---

给定一个 MCP 服务器配置和一组 IdP 能力集，输出构成生产级 MCP 授权层的授权面与拒绝规则。

输入：

- `mcp_resource_url` — 规范资源 URL（最具体的标识符；仅当路径能区分共托管的服务器时保留路径），用作 `aud` 以及受保护资源元数据的 `resource` 值。
- `idp_metadata_url` — IdP 的 `/.well-known/oauth-authorization-server`（或 OpenID Connect Discovery）URL。
- `idp_capabilities`：观测到的 `issuer`、`code_challenge_methods_supported`、`grant_types_supported`、`client_id_metadata_document_supported`、已弃用的 `registration_endpoint`、`response_types_supported` 以及 `authorization_response_iss_parameter_supported` 的值。
- `pre_registered_client_ids`：可选的发行方到客户端 ID 映射，由授权服务器操作员预先配置。优先使用此发行方作用域内的身份，其次使用 CIMD，最后仅在需要兼容时使用已弃用的 DCR。
- `application_type`：`native` 或 `web`，当选择已弃用的 DCR 兼容方式时为必填项。
- `credential_store`：以授权服务器发行方为键的客户端 ID 和注册凭证，访问令牌以 `(issuer, mcp_resource_url)` 为键。
- `tools`：MCP 工具列表，以及每个工具所需的 scope。

产出：

1. **拒绝门禁。** 若任何硬性条件不满足，则拒绝接入并停止：
   - `code_challenge_methods_supported` 中缺少 `S256`（PKCE 没有降级模式）。
   - `grant_types_supported` 中缺少 `authorization_code`。
   - `response_types_supported` 不是恰好为 `["code"]`。
   - 不存在任何注册路径：预注册的 `client_id`、`client_id_metadata_document_supported: true` 以及已弃用的 DCR 兼容端点均不可用。
   - 选择了 CIMD，但其 `client_id` 不是带路径的绝对 HTTPS 文档 URL，不匹配文档 URL，或文档缺少非空的 `client_name` 或 `redirect_uris` 数组。`application_type` 对 CIMD 是可选的。
   - 返回的 RFC 9207 `iss` 与重定向之前记录的发行方不同，或在服务器声明支持该参数时被省略。
   - 已弃用的 DCR 缺少 `application_type`，或其重定向 URI 策略与 `native` 或 `web` 冲突。

2. **受保护资源元数据文档**（RFC 9728），用于 MCP 服务器。对于带路径的资源，在该路径之前插入 well-known 段：`https://host/team/mcp` 映射为 `https://host/.well-known/oauth-protected-resource/team/mcp`。包含 `resource`、`authorization_servers`（发行方允许列表）、`scopes_supported` 以及 `bearer_methods_supported: ["header"]`。

3. **HTTP 端点。**
   - `GET /.well-known/oauth-protected-resource` — 返回（2）中的文档。
   - `POST /mcp`（无状态 MCP 传输）：在任何工具调用派发之前，校验本次请求的 bearer 令牌。
   - 仅 DCR 兼容：`POST /register`，在其之前执行 application-type 检查和速率限制检查。

4. **后台作业 + 例程。**
   - 一个定时 JWKS 刷新作业，将 `jwks_uri` 重新拉取到缓存 `{keys, fetched_at}` 中。幂等；绝不生成密钥。AS 负责轮换；资源服务器仅负责刷新。默认 `0 */6 * * *`；对于高轮换 IdP 可收紧为 `*/15 * * * *`。
   - 一个 `validate` 例程 — 检查 `iss` 允许列表、基于缓存 JWKS 的签名、`aud == mcp_resource_url`、`exp`、所需 scope。
   - 一个逐步提升的签发路径 — 仅当工具列表包含受用户初始未授予的 scope 所限制的操作时。

5. **缓存方案。** 每个已接受的发行方一个条目，以 `issuer` 为键，存储 `{keys, fetched_at}`。文档化读取模式：校验器读取缓存，在 `kid` 未命中时回退到一次同步刷新（重新拉取，而非轮换 — 重新拉取是幂等的，无法被转化为密钥创建型 DoS）。

6. **Scope 映射。** 将每个工具映射到其所需的 scope。输出表格：
   `| tool | required_scope | rationale |`。将破坏性工具归入其各自独立的 scope；绝不复用读取 scope 用于写入工具。

7. **运行时拒绝规则**（校验器必须编码这些规则）：
   - 当 `aud != mcp_resource_url` 时拒绝 → 401 `Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="<prm_url>"`。
   - 当 `iss not in authorization_servers` 时拒绝。
   - 当 `kid` 在一次重新拉取回退后仍不在缓存 JWKS 中时拒绝。
   - 当所需 scope 缺失时拒绝 → 403 `Bearer error="insufficient_scope", scope="<required>", resource_metadata="<prm_url>"`。
   - 拒绝任何没有 S256 `code_challenge` 的授权请求，并拒绝任何 `code_verifier`、客户端、重定向 URI 或 `resource` 与一次性授权码记录不匹配的令牌请求。
   - 拒绝任何发行方与其凭证存储键不匹配的凭证或令牌。发行方变更需要重新注册。

硬性拒绝（绝不接入以下任何一项 — 拒绝请求并记录原因）：

- 以明文存储 `client_secret`。公共客户端使用 `token_endpoint_auth_method: none`；机密客户端使用 `private_key_jwt`。静态存储或注册响应日志中不得存在明文共享密钥。
- 在校验器上跳过 `aud` 检查。受众绑定（访问令牌权限限制）正是 RFC 8707 + RFC 9728 的全部意义所在。
- 将 JWKS 缓存未命中回退连接到轮换并生成而非重新拉取。它永远不会产生缺失的 `kid`，反而允许攻击者控制的 `kid` 值强制无限制的密钥创建。回退必须是幂等的刷新。
- 允许无 PKCE 的授权码请求。OAuth 2.1 禁止此行为；校验器必须拒绝任何其存储的授权码记录缺少 `code_challenge` 的 `/token` 交换。
- 在没有刷新作业的情况下缓存 JWKS。要么定时刷新上线，要么授权面不部署。
- 在没有允许列表的情况下信任 `iss` 声明。任何接受来自任意 `iss` 令牌的校验器都允许攻击者建立自己的 IdP 并伪造令牌。
- 将入站 MCP 令牌转发到上游 API（令牌透传）。如果 MCP 服务器调用上游 API，它必须获取自己的独立令牌；透传会造成混淆代理问题。
- 以明文存储 `registration_access_token`。静态哈希存储；每次更新时要求提供明文。
- 将 MCP 请求元数据或已移除的协议会话视为授权状态。2026-07-28 传输是无状态的；对每个请求进行认证和授权。

输出：一份单页计划，包含受保护资源文档、以发行方为键的注册布局、以发行方和资源为键的令牌布局、所选的注册路径、HTTP 端点、JWKS 刷新作业、scope 映射以及运行时拒绝规则。以授权服务器实际元数据中发现的第一个未满足的部署门禁结尾。
