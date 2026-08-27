# 生产环境中的 MCP 身份验证：绑定发行方的注册与令牌

> 第 16 课构建了 OAuth 2.1 状态机。本课将针对 MCP 2026-07-28 加固其生产边界：优先使用 Client ID Metadata Document，只将已弃用的动态注册保留作兼容方案，验证授权响应的发行方，按发行方隔离客户端凭据，刷新 JWKS，并在每个无状态请求中使用绑定受众的令牌。
>
> **规范说明（2026-07-28）：** Dynamic Client Registration 已弃用，Client ID Metadata Document 取而代之。DCR 仍可作为兼容机制。使用 DCR 时，客户端要声明正确的 `application_type`。客户端必须验证响应中出现的 RFC 9207 `iss` 值，并且绝不能跨授权服务器发行方复用凭据。

**Type:** 构建
**Languages:** Python (stdlib)
**Prerequisites:** 第 13 阶段 · 第 16 课（OAuth 2.1 状态机）、第 13 阶段 · 第 17 课（网关）
**Time:** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器，并验证其契约。
- 通过 Client ID Metadata Document 注册，同时将已弃用的 DCR 隔离为回退方案。
- 验证 RFC 9207 `iss`；按授权服务器发行方保存注册信息，并按发行方与资源保存受资源约束的令牌。
- 按计划缓存和刷新 JWKS 密钥，使签名验证能够经受密钥轮换。
- 使用 RFC 8707 resource indicator 将令牌绑定到单个 MCP 资源，并拒绝“混淆代理”式复用。
- 在 JWT 验证与令牌内省之间做出选择，定义吊销信息的新鲜度，并在身份依赖不可用时安全失败。
- 分离授权服务器、资源服务器和客户端，让每个角色只执行自己负责的检查。
- 根据部署检查清单审计授权服务器，拒绝不安全的注册或令牌复用。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个仅靠内存模拟器无法发现的运维缺口。

第一个缺口是注册与凭据隔离。真实组织可能运行数百台 MCP 服务器和数千个 MCP 客户端。2026-07-28 修订版推荐使用 **Client ID Metadata Document**：客户端把由自己控制路径的 HTTPS URL 直接用作标识符，授权服务器主动拉取该 URL 上的元数据。RFC 7591 动态注册现在仅作为已弃用的兼容路径。不得不用 DCR 时，请求必须声明正确的 `application_type`。客户端按授权服务器发行方保存注册信息，并按 `(issuer, resource)` 二元组保存访问令牌。发行方变化意味着必须重新注册；资源不同意味着必须另取一枚绑定相应受众的令牌。

第二个缺口是密钥轮换。JWT 验证依赖授权服务器发布为 JSON Web Key Set（JWKS）的签名密钥。授权服务器会按计划轮换密钥（通常每小时一次，事件响应期间有时更快）。如果 MCP 服务器只在启动时获取一次 JWKS，那么在进入轮换窗口前一切正常，之后所有请求都会失败，直到服务重启。生产系统应把 JWKS 作为缓存值，并用刷新任务在旧密钥过期前覆盖缓存；如果收到由比缓存更新的密钥签发的令牌，还要在缓存未命中时执行一次回退获取。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 resource indicator。在生产环境中，该 indicator 会成为每个请求都必须通过的强制 claim 检查。MCP 服务器将 `token.aud` 与自身的规范资源 URL 比较，不匹配就返回 HTTP 401。这是防止上游 MCP 服务器（或持有一台服务器专用令牌的恶意客户端）把该令牌重放到同一信任网格中另一台服务器上的唯一防线。

本课会把每个缺口映射到接口中的具体部分。元数据文档对应一个 HTTP 端点。JWKS 缓存刷新对应定时任务加键值缓存。JWT 验证则是资源服务器在分发任何工具前执行的例程。保持三个角色彼此分离，每个角色便只需落实自己拥有的检查：授权服务器负责签发与轮换密钥，资源服务器负责缓存与验证，客户端负责发现与注册。

## 范围：第 16 课之后的生产强制措施

[第 16 课：使用 OAuth 2.1 保护 MCP](../../16-mcp-security-oauth-2-1/docs/en.md)负责授权码状态机、PKCE、受保护资源发现、resource indicator 和 scope 决策。本课不会定义第二套 OAuth 流程。它从这些契约已经存在的地方开始，讨论已部署的资源服务器如何在密钥轮换、不透明令牌验证、吊销、依赖故障、发布与事件响应期间继续执行这些契约。

这里的生产边界更窄，也更偏向运维：

- JWT 路径在每个请求中验证固定的发行方、算法、签名密钥、受众、时间 claim 和 scope，同时安全刷新 JWKS。
- 不透明令牌路径调用发行方经过身份验证的内省端点，并验证返回的 active 状态、受众或资源、过期时间、主体和 scope。
- 吊销策略定义凭据最迟必须在多久后停止生效，以及哪个缓存可能延迟这一事实。
- 故障策略决定发现服务、JWKS、内省或吊销基础设施不可用时应如何处理。
- 证据记录哪些发行方元数据、密钥集或内省响应、令牌 claim、策略版本和拒绝原因促成了最终结果，同时不保存令牌本身。

这种区分让两课能够组合。第 16 课证明流程本身成立。第 18 课证明令牌进入真实 MCP 请求路径后仍然可信，或者会被拒绝。

## 概念

### RFC 8414——OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

当客户端拿到一个 MCP 资源 URL 后，会串联两次发现：RFC 9728 的 `oauth-protected-resource`（资源服务器文档）给出发行方，随后本 RFC 的 `oauth-authorization-server` 给出所有端点。客户端绝不硬编码授权 URL。

对于带路径的资源标识符，应把 well-known 段插入该路径之前。例如，`https://mcp.example.com/team/server` 的受保护资源元数据地址是 `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`。把 `/.well-known/...` 追加到资源路径之后是错误的。

在信任某个 IdP 用于 MCP 之前，需要验证以下契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 定义的 PKCE）。规范明确要求：如果该字段**缺失**，授权服务器就不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 至少存在一种注册路径：`client_id_metadata_document_supported: true`（首选 CIMD）、预注册客户端，或 `registration_endpoint`（已弃用的 RFC 7591 兼容路径）。
- 如果 `authorization_response_iss_parameter_supported` 为 true，客户端必须要求返回 RFC 9207 `iss`，并将它与重定向前记录的发行方做严格比较。
- 对于 OAuth 2.1，`response_types_supported` 必须恰好是 `["code"]`。

如果缺少 `S256`，MCP 服务器就拒绝与该 IdP 一同部署——PKCE 不存在降级模式。如果两种注册路径都没有公布，且你也没有预注册的 `client_id`，同样无法注册；错的是部署清单，而不是代码。

### RFC 9728（回顾）——受保护资源元数据

第 16 课已经介绍 RFC 9728。生产环境中的差异在于：客户端只能通过这份文档查找受**这台** MCP 服务器信任的授权服务器。单台 MCP 服务器可以接受来自多个 IdP 的令牌（例如员工一个、合作伙伴一个）。RFC 9728 声明这组 IdP；RFC 8414 则描述每个 IdP 支持的能力。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Document（推荐默认方案）

CIMD 将注册从“推送”倒转为“拉取”。客户端不再请求授权服务器生成 `client_id`，而是把自己控制的 HTTPS URL **直接作为** `client_id`。该 URL 解析为一份 JSON 元数据文档；授权服务器在 OAuth 流程中按需获取它。信任根落在 DNS 上：如果服务器运营方信任 `app.example.com`，就会信任由 `https://app.example.com/client.json` 提供的客户端。无需注册往返，不会耗尽 `client_id` 命名空间，也不必维护需要在多台服务器间同步的状态。

客户端托管的元数据文档如下：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于提供该文档的 URL（授权服务器会验证，若不匹配就拒绝）。授权服务器通过 RFC 8414 元数据中的 `client_id_metadata_document_supported: true` 声明支持该机制。

按当前 CIMD 契约，`client_id`、`client_name` 和非空 `redirect_uris` 数组是必填项。客户端标识符必须是带路径的绝对 HTTPS URL。可以包含 `application_type`，但它不是 CIMD 的必填字段。不要把 DCR 对 `application_type` 的要求照搬到首选 CIMD 路径。

规范对两个安全事实说得非常直白：

- **SSRF。** 授权服务器会获取攻击者提供的 URL，因此必须防范服务端请求伪造（不得访问内部或管理端点）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者冒用合法客户端的元数据 URL，并绑定任意 `localhost` 重定向。授权服务器在征求同意时**必须**清楚显示重定向 URI 的主机名，并且对于只使用 `localhost` 的重定向**应该**发出警告。

CIMD 不需要服务端状态，因此无需像 DCR 那样搭建注册服务。客户端侧是只读的：从静态 HTTPS 端点提供元数据文档，让授权服务器主动拉取即可。

如果授权服务器运营方已经预配了客户端标识符，应先使用该发行方作用域内的注册信息，再尝试自动注册。否则优先使用 CIMD。只有当发行方既不能使用预注册、也不能使用 CIMD 时，才使用已弃用的 DCR。

### RFC 7591：已弃用的兼容注册方式

DCR 已在 2026-07-28 修订版中弃用。只有授权服务器无法使用 CIMD 且预注册又不切实际时，才保留它。兼容客户端发送如下 POST 请求：

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id`，以及供后续更新使用的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` 不是装饰字段。使用 loopback 的桌面客户端应声明 `native`；托管在服务器上的客户端应声明 `web`，并使用 HTTPS 重定向 URI。对公共原生客户端来说，`token_endpoint_auth_method: none` 是正确默认值。它只获得 `client_id`，由 PKCE 提供持有证明。

生产环境中有三个陷阱：

- 注册端点必须按源 IP 限流。否则，恶意参与者可以编写脚本创建数百万条伪造注册，耗尽 `client_id` 命名空间。在注册服务处理请求前先执行限流检查。
- 某些企业 IdP 要求 `software_statement`（由签名 JWT 为客户端背书）。本课 mock 会跳过它；生产系统应加入验证步骤，拒绝所有没有签名且不只使用 localhost 重定向 URI 的注册。
- `registration_access_token` 必须以哈希形式存储，不能保存明文。一旦该令牌被盗，攻击者就能重写客户端的重定向 URI。

### RFC 8707（回顾）——Resource Indicator

第 16 课已经确定其结构。生产规则是：每次令牌请求都包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用中都验证 `token.aud` 与自己的资源 URL 匹配。规范 URI 是服务器**最具体**的标识符：scheme 和 host 使用小写，不含 fragment，并且按惯例不带末尾斜杠。规则**不会**删除 path component——当路径用于标识某台独立 MCP 服务器时，规范会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。为每台服务器选择一个，并让 `aud` 精确绑定到它。（为简洁起见，本课 mock 使用 `https://notes.example.com` 这类只有主机的受众；如果一次部署在同一 origin 下托管多台 MCP 服务器，则应通过路径区分。）

### RFC 7636（回顾）——PKCE

OAuth 2.1 强制要求 PKCE。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何缺少 verifier，或 verifier 的哈希与已保存 challenge 不一致的令牌请求。

### MCP 2026-07-28 授权配置文件

当前 MCP 修订版保留了 OAuth 资源服务器边界，同时让 MCP 传输保持无状态。协议会话中没有可缓存身份决策的位置，因此授权层必须独立验证每个请求：

- 实现 RFC 9728 受保护资源元数据，并在 401 响应的 `WWW-Authenticate: Bearer resource_metadata="..."` header 中提供其位置，**或者**使用 well-known URI `/.well-known/oauth-protected-resource`（SEP-985 允许省略该 header，并以 well-known 地址回退）。元数据的 `authorization_servers` 字段**必须**至少列出一台服务器。
- 在**每个**请求中，只通过 `Authorization: Bearer ...` 接收令牌——绝不能放在 query string 中，也绝不能只在会话开始时验证一次。
- 每个请求都要验证 `aud`、`iss`、`exp` 和所需 scope。服务器**必须**验证令牌确实是专为自己签发的（受众）；缺失或不匹配的 `aud` 必须被拒绝，不能视作通配符。
- 对 401/403 响应，返回 `WWW-Authenticate: Bearer`，其中携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，**不是**裸资源地址），以及 `scope="..."`（用于 `insufficient_scope`，即 403）。注意：参数名是 `resource_metadata`，它是一个发现指针；挑战中不存在 `resource` 参数。
- 授权服务器发现既接受 RFC 8414 OAuth 元数据，也接受 OpenID Connect Discovery 1.0；客户端必须按优先级尝试两个 well-known 后缀。
- 防御 **mix-up attack** 的是客户端（而不是服务器）：客户端在重定向前记录预期 `issuer`，并在兑换授权码前验证实际授权响应中返回的 `iss` 值（RFC 9207）。仅靠 PKCE 无法阻止 mix-up，因为客户端会把自己的 `code_verifier` 交给被引导至的任意 token endpoint。
- 一份客户端凭据只属于一个授权服务器发行方。如果发现结果指向另一发行方，客户端必须重新注册，而不能提交原有 `client_id`、注册令牌或访问令牌。
- CIMD 是首选注册机制。DCR 已弃用；兼容 DCR 请求仍须声明正确的 `application_type`。

OAuth 2.1 草案是底层基础；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 构成接口；MCP 规范则是配置文件。

### 部署能力检查清单

供应商功能表很快就会过时。应检查你实际准备部署的授权服务器返回的元数据。准入机制完全是机械式判断：

| 检查项 | 必需决策 |
|---|---|
| 发现的发行方 | 与策略预期完全一致的 HTTPS 发行方 |
| PKCE | 公布 `S256`；否则停止 |
| 注册 | 首选 CIMD，可接受预注册，DCR 仅作为已弃用的兼容方式 |
| 授权响应 | 当 RFC 9207 `iss` 出现或被声明支持时验证它 |
| 资源绑定 | 令牌请求携带 `resource`；资源服务器要求匹配的 `aud` |
| 凭据存储 | 按发行方保存客户端 ID 和注册凭据；按发行方加资源保存访问令牌 |
| DCR 兼容 | 声明 `native` 或 `web`；拒绝不符合所声明 application type 的重定向 URI |

不要根据产品名称或价格档位推断支持情况。把发现到的文档保存为部署证据；任何强制字段缺失时都应失败关闭。

### JWKS 刷新模式（授权服务器轮换，资源服务器刷新）

必须严格区分两个动词，因为混淆二者会造成真实的生产故障：

- **轮换（Rotate）**由*授权服务器*执行：生成新的签名密钥，将其发布到 JWKS，稍后再淘汰旧密钥。资源服务器既不参与、也无法执行轮换——它没有 IdP 的私钥。
- **刷新（Refresh）**由*资源服务器*执行：重新 `GET` 已发布的 JWKS 并写入缓存。这是资源服务器对 JWKS 所做的唯一操作。

生产故障模式是缓存陈旧。用定时刷新任务加键值缓存解决它。资源服务器按固定间隔运行任务（cron、timer 或运行时提供的其他机制），获取 `<issuer>/.well-known/jwks.json`，再覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取密钥。如果令牌中的 `kid` 不在缓存中，则触发**一次**同步刷新作为回退，然后重新检查。这样同时处理了两种情况：按计划刷新，以及由全新密钥签名的令牌先于下一次定时刷新到达的密钥重叠窗口。

回退动作**必须是重新获取，绝不能是轮换**。如果把缓存未命中路径接到“轮换并生成”，会同时破坏两件事：(1) 新生成密钥的 `kid` **仍然**与令牌不匹配，所以查找照样失败；(2) 攻击者只需批量发送包含随机 `kid` 的令牌，就能迫使系统无限创建密钥，形成自我制造的 DoS。重新获取具有幂等性，因此伪造的 `kid` 最多只浪费一次请求。

缓存结构如下：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两把密钥才是稳态。授权服务器会先引入下一把密钥（`k_2026_04`），过一段时间再淘汰上一把（`k_2026_03`），因此用旧密钥签发的令牌在过期前仍然有效。缓存保存二者的并集；验证器按 `kid` 选择。

### 验证例程

MCP 服务器在分发任何工具前运行验证。`code/main.py` 使用的形式如下：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（未命中时刷新一次），验证签名，再依次检查 `iss` 是否位于允许列表、`aud` 是否等于本服务器的规范资源、`exp` 是否有效，以及是否具备所需 scope；一旦失败，立即返回 `WWW-Authenticate` 挑战。将这些逻辑集中为资源服务器上的单个例程，可以确保每个入口（每次工具调用、每种传输）都经过相同检查，不会出现绕过验证直达工具的路径。

### 不透明令牌必须使用内省，不能靠猜测

并非所有访问令牌都是 JWT。如果发行方声明使用不透明令牌，资源服务器就无法把它解码成可信 claim。资源服务器必须通过经过身份验证的后端通道，把令牌发送给发行方的 RFC 7662 内省端点，并要求 `active: true`、符合预期的发行方上下文、精确匹配的 MCP 受众或资源、尚未过期的时间 claim、主体，以及具体工具所需的 scope。

内省缓存键应由发行方、令牌的单向摘要和 MCP 资源组成。切勿把明文令牌用作日志或缓存标签。正向缓存项的期限不得超过令牌过期时间、发行方缓存指引和部署的吊销新鲜度目标三者中的最早值。负向缓存要足够短，避免新签发令牌长时间被错误视为 inactive。即使不透明令牌字符串完全相同，针对一个资源得到的结果也不能授权另一个资源。

不要根据攻击者可控的令牌内容选择验证模式。JWT 或内省行为必须由已经验证的发行方元数据与部署配置固定决定。在 JWT 路径上，固定允许的算法和可信 `jwks_uri`；绝不能只根据令牌 header 选择算法或跟随密钥 URL。

### 吊销是一项新鲜度契约

RFC 7009 允许客户端请求授权服务器吊销令牌。该请求不会删除各资源服务器缓存中已有的副本。必须定义可接受的最大吊销延迟，并让所有缓存遵守它。

不透明令牌部署可以通过在每次高风险调用时内省，或使用较短的正向缓存，实现更及时的吊销。自包含 JWT 部署通常把短时效访问令牌与 refresh-token 吊销相结合；发行方级事件可以淘汰密钥；紧急情况下还可以使用可选的主体、会话或 token-id 拒绝列表在本地立即拒绝。除非资源服务器拥有最新外部吊销证据，否则已签名 JWT 在过期前仍然具有密码学有效性。

登出、账号停用、撤回同意和事件响应是不同触发条件，但最终都必须满足同一个可测量陈述：不超过已声明的吊销窗口后，每个副本都拒绝该凭据。应通过负载均衡器测试这项陈述，不能只测试某一个已预热的进程。

### 依赖故障需要预先声明决策

绝不要在异常处理器中临时决定可用性策略。

| 故障 | 安全的生产行为 |
|---|---|
| 定时 JWKS 刷新失败，但已知 `kid` 仍位于尚未超过有界有效期的缓存中 | 只允许在声明的 stale-on-error 窗口内继续，并发出服务降级的健康证据 |
| 令牌包含未知 `kid`，且唯一一次允许的刷新失败 | 拒绝；绝不能接受无法验证的签名 |
| 内省不可用 | 对受保护调用失败关闭；不得把网络故障转成 `active: true` |
| 受保护资源或发行方元数据发生意外变化 | 停止新的注册和令牌获取；在有界事件策略下，只保留明确固定且尚未过期的配置 |
| 吊销端点不可用 | 将登出或吊销报告为未完成；可能时在本地把凭据标记为不可用，并且不要声称全局吊销已经成功 |
| 时钟源或 claim 类型无效 | 拒绝，而不是持续扩大时钟偏差容忍度，直至令牌勉强通过 |

依赖故障与无效凭据必须分别分类。依赖中断是带有健康状态和重试策略的运维错误。签名、发行方、受众、有效期或 scope 错误则是授权拒绝。二者都不能进入工具处理器，也都不应把令牌内容泄漏到审计证据中。

### 受众重放演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都注册到同一授权服务器。服务器 A 被攻破。攻击者取出用户的 notes 令牌，并把它重放给服务器 B。

服务器 B 的验证器执行：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 检查 `iss` 是否位于受保护资源元数据的 `authorization_servers` 中。（通过——使用同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，并携带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众 claim 是协议层防御这种攻击的唯一手段。为了性能而跳过它，是最常见的生产错误；验证器必须在每个请求中运行，不能只在会话开始时运行。规范将其称为**访问令牌权限限制**：MCP 服务器 `MUST` 拒绝任何未在受众中指名自己的令牌。

> **命名说明。** 规范把 *confused deputy* 一词保留给一个相关但不同的问题：充当第三方 API OAuth **代理**的 MCP 服务器使用静态客户端 ID，在没有获得逐客户端用户同意的情况下转发令牌。受众绑定解决的是上述重放；混淆代理问题的解决方案则是逐客户端征求同意，**并且**绝不把入站令牌原样传给上游 API（MCP 服务器 `MUST` 获取自己独立的上游令牌）。

### Mix-up attack（服务器无法提供的客户端侧防御）

客户端在整个生命周期中会与多台授权服务器交互。恶意 AS 可能试图诱使客户端把诚实 AS 的授权码提交到攻击者的 token endpoint。受众绑定无法解决此问题——攻击发生在任何令牌产生之前。防御必须位于客户端（RFC 9207）：

1. 重定向之前，客户端从已验证的 AS 元数据中记录预期 `issuer`。
2. 收到授权响应时，客户端把返回的 `iss` 参数与已记录的发行方比较（简单字符串比较，不做规范化），然后才能把授权码发送出去。
3. 不匹配（或者响应中缺少 `iss`，而 AS 已公布 `authorization_response_iss_parameter_supported`）→ 拒绝，而且连 `error` 字段都不要展示。

仅靠 PKCE 无法阻止 mix-up，因为客户端会把自己的 `code_verifier` 交给被引导至的任意 token endpoint。因此，规范要求逐请求记录发行方，并将它与 PKCE verifier 和 `state` 一同保存。

### 故障模式

- **陈旧 JWKS。** AS 轮换密钥后，验证器开始拒绝有效令牌。解决方法是前述“定时刷新 + 缓存未命中时重新获取”模式。绝不能在没有刷新任务的情况下缓存 JWKS。
- **把轮换当作回退。** 将缓存未命中路径接到“轮换并生成”而不是重新获取，是真实的 bug：它永远无法生成缺失的 `kid`，还会把由攻击者控制的 `kid` 转变为密钥创建 DoS。回退动作必须是幂等的 `refresh-jwks`。
- **缺少 `aud` claim。** 部分 IdP 默认省略 `aud`，除非令牌请求中包含 `resource`。验证器必须拒绝缺少 `aud` 的令牌，不能把缺失视作通配符。
- **缺少 `iss` 检查导致 mix-up。** 如果客户端没有把 RFC 9207 授权响应 `iss` 参数与重定向前记录的发行方比较，就可能被引导到攻击者的 token endpoint，用诚实 AS 的授权码发起兑换。这是客户端侧故障，资源服务器无法弥补。
- **Scope 提升竞态。** 同一用户的两个并发 step-up 流程可能都成功，并生成 scope 不同的两枚访问令牌。验证器必须使用请求中实际提交的令牌，不能查询“用户当前的 scope”——后者会制造 TOCTOU 窗口。
- **注册令牌被盗。** 泄漏的 `registration_access_token` 会让攻击者重写重定向 URI。静态存储时应保存哈希；每次更新都要求客户端提交明文；怀疑泄漏时立即轮换。
- **未固定 `iss`。** 接受任意 `iss` 的验证器，会允许攻击者搭建自己的授权服务器，为目标受众注册客户端并签发令牌。受保护资源元数据中的 `authorization_servers` 列表就是允许列表；必须强制执行。
- **凭据或令牌缓存冲突。** 如果客户端只按资源索引注册信息，就可能把一个授权服务器的身份提交给另一个服务器。如果只按发行方索引访问令牌，则可能把令牌重放给错误受众。应按已验证发行方索引注册信息，按 `(issuer, resource)` 索引访问令牌，并在发行方变化时重新注册。

```figure
t3-jwks-rotate
```

## 使用它

`code/main.py` 使用标准库 Python 和三个角色——`AuthorizationServer`、`ResourceServer` 与 `Client`——走通完整的生产流程。流程如下：

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

第一条命令会打印绑定发行方的注册与令牌验证过程记录。第二条命令会报告十八项检查全部通过。两条命令都不会打开网络监听端口，也不会写入凭据。

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，检查可用的注册方案（CIMD 使用 `client_id_metadata_document_supported`，DCR 使用 `registration_endpoint`）以及对 `S256` PKCE 的支持。
3. 客户端先查找按发行方保存的预注册信息；若不存在，则使用其 HTTPS Client ID Metadata Document 注册。已弃用的 DCR 保留为一个可单独测试的兼容方法。
4. 客户端记录经过验证的发行方，生成 S256 challenge，收到一次性授权码及 `iss`，验证返回的发行方，再使用原始 verifier 和 RFC 8707 `resource` indicator 兑换授权码。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；定时刷新重新拉取 JWKS 并写入缓存。
8. 下一次调用无需重启，就能用刷新后的密钥验证；在重叠窗口内，上一枚令牌仍能通过验证。
9. 针对另一 MCP 资源的受众重放会收到 401，响应包含 `audience mismatch` 和 `resource_metadata` 指针。

这里的 JWT 使用 HS256 与共享 secret（使课程只依赖标准库即可运行）。生产环境会配合上述 JWKS 模式使用 RS256 或 EdDSA；其余验证逻辑相同。由于 IdP 与资源服务器运行在同一进程，`refresh_jwks` 会直接读取授权服务器的密钥列表；在线路上，这对应一次 HTTP `GET`，目标是 `jwks_uri`。

## 交付它

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和一组 IdP 能力，该技能会生成需要搭建的身份验证接口——受保护资源元数据、应采用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、scope 映射，以及 IdP 不支持完整 RFC 配置文件时要执行的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪完整流程。注意 IdP 如何在第 6 步轮换密钥，定时 `refresh_jwks` 如何重新拉取已发布的密钥集，以及旧令牌（处于重叠窗口）和新令牌如何在无需重启的情况下都通过验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新 IdP。签发一枚由新 IdP 签名的令牌，确认验证器接受它。再签发一枚由未列出 IdP 签名的令牌，确认验证器拒绝，并返回 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 向 `register_client` 添加限流检查，并确保它在注册服务接受请求前运行。按源 IP 实现 token bucket，保存在一个以 IP 为键的小型 dict 中。

4. 阅读 RFC 7591，找出本课 `/register` handler 未验证的两个字段，并补上验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 添加第二台授权服务器。确认客户端保存一份单独的、按发行方索引的注册信息，并拒绝复用第一台发行方的令牌或 `client_id`。

6. 证明 DoS 修复有效。向验证器发送一枚包含随机 `kid` 的令牌，确认 `refresh_jwks` 最多运行一次，并且授权服务器的密钥数量不会增长。然后故意把回退重新接到“轮换并生成”，观察每枚伪造令牌如何让密钥数量增长——完成观察后恢复为重新获取。

7. 分别以 `native` 和 `web` 客户端演练已弃用的 DCR。确认使用 HTTP 重定向 URI 的 web 客户端，以及没有精确 loopback 重定向的 native 客户端都会被拒绝。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| ASM | “OAuth 元数据文档” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | “客户端元数据 URL” | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；AS 拉取其中的 JSON。它是 MCP 2026-07-28 的首选注册方式 |
| DCR | “自助式客户端注册” | RFC 7591 `POST /register`；在当前 MCP 中已弃用，只为兼容而保留 |
| JWKS | “用于 JWT 验证的公钥” | 从 `jwks_uri` 获取、按 `kid` 索引的 JSON Web Key Set |
| Rotate 与 refresh | “更新密钥” | *Rotate* = AS 生成/淘汰签名密钥；*refresh* = 资源服务器重新获取已发布的密钥集。资源服务器只执行 refresh |
| Resource indicator | “受众参数” | RFC 8707 `resource` 参数，把令牌固定到一台服务器 |
| `aud` claim | “受众” | 验证器与规范资源 URL 比较的 JWT claim |
| Audience replay | “令牌重放” | 为服务器 A 签发的令牌被提交给服务器 B；通过受众验证防御（规范称为访问令牌权限限制） |
| Confused deputy | “代理令牌误用” | 使用静态客户端 ID 的 MCP 代理在未逐客户端征求同意时转发令牌；与受众重放不同 |
| Mix-up attack | “错误的 token endpoint” | 客户端被引导到攻击者端点兑换诚实 AS 的授权码；通过客户端侧 RFC 9207 `iss` 防御 |
| `iss` allow-list | “可信授权服务器” | 受保护资源元数据的 `authorization_servers` 中列出的集合 |
| `resource_metadata` | “PRM 文档的位置” | 401/403 响应中的 `WWW-Authenticate` 参数，指向 RFC 9728 元数据 URL |
| Public client | “原生或浏览器客户端” | 没有 `client_secret` 的 OAuth 客户端；由 PKCE 弥补 |
| `WWW-Authenticate` | “401/403 响应 header” | 携带 `Bearer error=...` 指令，驱动客户端恢复 |

## 延伸阅读

- [MCP 授权规范（2026-07-28）](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)——当前 MCP 授权配置文件
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)——CIMD、发行方验证、DCR 弃用与按发行方保存凭据等变化
- [OAuth Client ID Metadata Document（draft-ietf-oauth-client-id-metadata-document-00）](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00)——CIMD
- [RFC 8414——OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414)——发现契约
- [RFC 7591——OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591)——DCR（回退路径）
- [RFC 7636——Proof Key for Code Exchange（PKCE）](https://datatracker.ietf.org/doc/html/rfc7636)——公共客户端的持有证明
- [RFC 8707——Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707)——受众绑定
- [RFC 9728——OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728)——资源服务器发现
- [RFC 9207——OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207)——防御 mix-up attack 的 `iss` 参数
- [RFC 7662：OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009：OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
