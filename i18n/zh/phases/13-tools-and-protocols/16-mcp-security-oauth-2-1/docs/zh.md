# MCP 授权：CIMD、签发方绑定、PKCE 与权限升级

> 远程 MCP 请求是无状态的，但其授权并非匿名。应把每份凭据绑定到创建它的签发方，并把每个令牌绑定到接收它的资源。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 09 课（传输）、第 13 阶段 · 第 15 课（安全 I）
**Time:** 约 90 分钟

## 学习目标

- 通过受保护资源元数据发现授权服务器。
- 优先采用 Client ID Metadata Documents，而不是已弃用的 Dynamic Client Registration。
- 在无法避免使用 DCR 兼容路径时，声明正确的 `application_type`。
- 验证授权响应中的 `iss`，并按签发方隔离凭据。
- 使用 PKCE、资源指示符、受众验证和增量权限范围。
- 在没有协议会话的情况下发送已授权的 MCP 2026-07-28 请求。

## 问题

远程 MCP 服务器可能读取私有记录、写入外部系统或触发成本高昂的工作。身份认证告诉服务器是谁提交了凭据；授权还必须回答：

- 哪个授权服务器签发了这份凭据？
- 令牌是签发给哪个 MCP 资源的？
- 哪个客户端和重定向 URI 完成了流程？
- 用户批准了哪些操作？
- 当前这个具体请求是否仍符合该批准？

2026-07-28 授权配置文件强化了客户端登记和签发方处理。它优先采用 Client ID Metadata Documents，弃用 Dynamic Client Registration，要求在 DCR 中提供正确的 `application_type`，验证 RFC 9207 签发方响应，并禁止跨签发方复用凭据。

这些规则是对无状态核心的补充，并不会恢复核心握手或 `Mcp-Session-Id`。

## 核心概念

### 理解三个角色

- **MCP 客户端：**代表资源所有者发送请求。
- **MCP 资源服务器：**接收访问令牌并提供 MCP 端点。
- **授权服务器：**验证资源所有者身份、征得同意并签发令牌。

资源服务器和授权服务器可以由同一方运营，但两者的标识符与验证职责仍应保持独立。

### 授权适用于 HTTP

MCP 授权规范适用于基于 HTTP 的传输。本地 stdio 服务器运行在进程与操作系统的信任边界内；不要仅为保持形式对称，就给 stdio 添加虚假的浏览器 OAuth 流程。

对于远程 Streamable HTTP，应在每次请求的 `Authorization` 请求头中发送 bearer token，绝不能把它放进 URL。

### 从受保护资源元数据开始

资源服务器发布 RFC 9728 元数据：

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

客户端从 MCP 资源 URL 出发，获取该文档，选择其中声明的授权服务器，然后获取该服务器的 OAuth 或 OpenID Connect 元数据。

构造 RFC 9728 well-known URL 时，应保留资源路径。对于资源 `https://notes.example.com/mcp`，本课使用 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`。丢弃 `/mcp` 后缀可能选中同一来源下另一个受保护资源的元数据。

不要根据主机名猜测授权服务器，也不要追随未验证错误正文中发现的签发方。客户端应维护一套可接受哪些签发方的信任策略。

### 验证授权服务器元数据

元数据应公开端点及其支持的控制措施：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

PKCE 必须使用 S256。记录精确的签发方字符串；该精确值将成为注册信息和令牌存储的键。

### 遵循注册优先级

如果客户端已经与所选签发方建立明确关系，则使用预注册客户端信息。否则，在授权服务器声明支持时，优先使用 Client ID Metadata Documents。仅将 DCR 作为已弃用的兼容性后备方案；如果以上机制均不可用，再提示用户提供客户端信息。

### 优先使用 Client ID Metadata Documents

Client ID Metadata Document 向授权服务器提供一个 HTTPS URL，该 URL 既是客户端标识符，也是其元数据所在位置：

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

授权服务器获取并验证该文档。`client_id` 必须是带路径的 HTTPS URL，且文档中的值必须与该 URL 完全相等。文档必需字段为 `client_id`、`client_name` 和 `redirect_uris`。示例中出现的 `application_type` 并不是 CIMD 的必需字段；新增的强制使用要求专门针对 DCR 路径。

获取该文档属于 SSRF 敏感操作。解析并验证目标；拒绝环回、私有、链路本地及其他禁止的地址；每次重定向和 DNS 变化后重新检查；限制重定向次数、字节数和耗时；要求返回 JSON；并且只按已验证的 HTTP 缓存控制进行缓存。将 `client_name` 及其他显示字段视为不可信文本。

CIMD 不再要求每次首次接触都签发新的动态标识符，但它不会取消重定向 URI 验证、签发方策略或用户同意。

### DCR 是兼容路径

Dynamic Client Registration 仍可用于较旧的授权服务器，但新的 MCP 实现已经不推荐使用它。

使用 DCR 时，声明 `application_type`：

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- 桌面、移动、命令行及环回客户端使用 `native`。
- 远程托管的浏览器应用使用 `web` 和远程 HTTPS 重定向。

在 OpenID Connect 注册实现中，省略该字段可能会默认为 `web`，从而导致合法的环回重定向失败。

将 DCR 代码放在显式后备决策之后。任意 CIMD 验证失败时都不要静默降级，否则可能把一次安全失败转化为较弱的登记路径。

### 将凭据绑定到签发方

将签发方签发的登记材料保存到精确的签发方名下：

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

如果受保护资源发现结果从 `https://auth-one.example` 变为 `https://auth-two.example`，应重新评估信任。绝不能把第一个签发方的客户端秘密、DCR 客户端 ID、注册访问令牌、刷新令牌或访问令牌发送给第二个签发方。预注册和 DCR 客户端都必须使用新签发方签发的凭据。

CIMD 客户端 ID 与此不同，因为它是自托管 HTTPS URL，而不是由授权服务器签发的凭据。同一个 CIMD URL 可以移植：新的可信签发方无需重新执行 DCR 注册，只需获取并验证文档。授权响应和令牌仍须在新签发方名下验证与存储。

### 使用 PKCE 的授权码流程

交互流程如下：

1. 生成高熵 `code_verifier`。
2. 派生 S256 `code_challenge`。
3. 发送授权请求，其中包含精确的 `client_id`、`redirect_uri`、`scope`、`code_challenge` 和 `resource`。
4. 接收包含 `code` 以及可选 `iss` 的授权响应。
5. 使用任何响应字段前，将 `iss` 与记录的精确签发方进行验证。
6. 使用 `code_verifier`、同一个重定向 URI 和同一个 `resource` 交换授权码。
7. 将得到的令牌存储在 `(issuer, resource)` 键下。

RFC 8707 的 `resource` 参数同时出现在授权请求与令牌请求中，用于标识规范的 MCP 服务器 URI。

### 精确验证 `iss`

RFC 9207 可以防止某一签发方的授权响应被误认为来自另一个签发方。

当 `iss` 存在时，应直接与已记录签发方比较，不得进行大小写折叠、末尾斜杠修改、默认端口移除或百分号编码规范化。若不匹配，不要使用授权码，甚至不要展示该响应中由攻击者控制的错误详情。

包含 `iss` 的授权服务器会声明 `authorization_response_iss_parameter_supported: true`。即使缺少这项声明，当前客户端也仍会验证实际出现的 `iss`。

### 在 MCP 服务器验证受众

资源服务器只接受签发给自身的令牌：

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

无效、已过期、签发方错误或受众错误的令牌都会得到 401。MCP 服务器不得接受或转交签发给其他服务的令牌。

### 请求当前所需的最小权限范围

从当前所需的权限范围开始。如果之后某个工具需要更多权限，服务器返回 403 以及权威的权限范围质询：

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

客户端解释新增权限，征得同意，使用合并后的权限范围集合执行新的授权流程，再用新的 JSON-RPC ID 重试 MCP 请求。

不要假设质询中的权限范围属于 `scopes_supported` 的子集。对于当前操作，质询才是权威依据。

### 授权与无状态 MCP 线协议

已授权的工具调用仍然携带完整的当前请求信封：

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

令牌授权主体，请求元数据协商协议行为。两者不能相互替代。

按固定顺序验证线协议：JSON-RPC 和元数据类型、请求头与正文相等性，然后是协议支持情况。路由或版本请求头不匹配时返回 HTTP 400 和 `-32020`。如果请求头与正文一致但版本不受支持，则返回 HTTP 400 和 `-32022`，且 `data` 必须精确为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 和 `-32601`。

每项请求错误，包括 401 无效令牌和 403 权限范围不足，都采用带原始请求 `id` 的 JSON-RPC 错误信封。结构化恢复信息放入可选错误 `data`；`WWW-Authenticate` 仍是 HTTP 响应头。通知没有 `id`，所以不会收到 JSON-RPC 正文。已接受的 HTTP 通知返回 202 和空响应体。

服务器实现 `server/discover` 并声明工具，因此也必须实现强制的 `tools/list` 方法。其工具描述符具有稳定的名称、描述和对象根级别的 `inputSchema`。列表具有确定性，并返回 `resultType`、服务器身份元数据、有界的 `ttlMs` 与 `cacheScope`。发现和不因用户而异的工具列表可以在授权前开放；如果任一者随主体变化，应采用正常策略与私有缓存。

### 禁止令牌透传

MCP 服务器不得把客户端的 MCP 访问令牌转发给下游 API。应获取受众正确的独立下游令牌，或使用显式的令牌交换设计。只有当服务拒绝签发给其他对象的令牌时，受众验证才有效。

### 刷新令牌

刷新令牌是可选的。签发后，应以保密方式存储，并按签发方与资源设置键。不要假设刷新令牌一定存在。授权服务器支持轮换时应执行轮换，并检测已失效值被重复使用的情况。

```figure
t3-scope-stepup
```

## 动手构建

`code/main.py` 是一个进程内协议与授权模拟器。它实现受保护资源发现、授权服务器元数据、CIMD 登记、按版本门控的 DCR 后备路径、应用类型检查、PKCE、签发方验证、资源绑定令牌、权限范围升级、`server/discover`、`tools/list` 和无状态工具请求。

模型接收已解析的请求正文和路由请求头。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或 `Accept`。应将它接入第 09 课的 Streamable HTTP 适配器；该适配器要求 `Content-Type: application/json`，并要求 `Accept` 值同时包含 `application/json` 与 `text/event-stream`。

运行：

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

输出依次显示发现、CIMD 登记、普通读取、两次独立权限范围升级，以及按签发方设置键的凭据存储。

## 实际使用

将模拟器对象映射到生产组件：

- `ResourceServer.protected_resource_metadata` 对应 RFC 9728 端点。
- `AuthorizationServer.metadata` 对应 RFC 8414 或 OpenID Connect 发现。
- `Client.enroll` 对应 CIMD 解析加一个显式 DCR 兼容分支。
- 签发方签发的客户端凭据和 `tokens_by_issuer_resource` 对应加密记录。CIMD URL 可以保持可移植，但其授权结果仍绑定到签发方。
- `ResourceServer.handle` 对应中间件：它在分派前验证当前 MCP 请求头、令牌和工具权限范围，同时将每项请求错误保留在匹配的 JSON-RPC 信封中。

## 交付成果

本课交付 `outputs/skill-oauth-scope-planner.md`。它现在会设计登记优先级、签发方绑定的凭据存储、应用类型、PKCE、资源指示符、权限范围质询与当前无状态请求边界。

## 练习

1. 添加刷新令牌轮换，并拒绝重复使用上一个刷新令牌。
2. 添加签发方允许列表。签发方变化后，只复用可移植的 CIMD URL；拒绝此前签发方签发的所有凭据与令牌。
3. 为授权码添加过期时间，并确认延迟交换会失败。
4. 构建使用远程 HTTPS 重定向的 Web 客户端变体，并比较其 DCR 元数据与原生客户端的差异。
5. 在同一签发方下添加第二个资源，确认其访问令牌无法用于第一个资源。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 受保护资源元数据 | 标识资源和授权服务器的 RFC 9728 文档 |
| CIMD | URL 同时作为 OAuth 客户端标识符的 HTTPS 元数据文档 |
| DCR | 为兼容性保留、但已弃用的动态客户端登记方式 |
| `application_type` | `native` 或 `web`，用于验证重定向 URI 规则 |
| PKCE | 使用 verifier 与 S256 challenge 来保护被截获授权码的机制 |
| `iss` | RFC 9207 授权响应的签发方标识符 |
| 资源指示符 | 将令牌请求绑定到 MCP 资源的 RFC 8707 参数 |
| 受众 | 令牌对其有效的资源 |
| 权限升级 | 为当前操作新增的权限范围重新征得同意并签发令牌 |
| 签发方绑定凭据 | 按授权服务器的精确签发方隔离的注册与令牌记录 |

## 延伸阅读

- [MCP 2026-07-28 授权规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728：OAuth 2.0 受保护资源元数据](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707：OAuth 2.0 资源指示符](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207：OAuth 2.0 授权服务器签发方标识](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document 草案](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
