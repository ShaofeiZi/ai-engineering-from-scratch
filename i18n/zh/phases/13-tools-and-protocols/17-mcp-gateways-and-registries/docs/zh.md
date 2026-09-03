# 无状态 MCP 网关与注册表准入

> 网关应让每条路由都清晰明确。2026-07-28 协议无需传输会话，就能为网关提供方法、名称、版本、能力、身份、缓存与追踪边界。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 15 课（工具投毒）、第 13 阶段 · 第 16 课（OAuth 2.1）
**Time:** 约 75 分钟

## 学习目标

- 在一个 2026-07-28 端点后聚合多个 MCP 服务器，无需会话亲和性。
- 在应用策略或转发前，验证逐请求元数据和路由请求头。
- 使用稳定命名空间、确定性顺序、描述符固定、RBAC 与私有缓存合并工具。
- 将注册表记录视为仍需经过准入策略的发现证据。
- 正确路由请求作用域 SSE、`subscriptions/listen`、MRTR 重试与 Tasks 扩展调用。
- 将旧版握手和会话支持与现代路径隔离。

## 问题

让一个客户端直接连接一台服务器很简单。更大规模的部署需要对更棘手的问题给出一致答案：

- 允许哪些服务器接入？
- 每个主体可以看到和调用哪些工具？
- 两个后端公开同名工具时怎么办？
- 如何审查描述符变更？
- 在哪里应用速率限制与审计事件？
- 下一次请求能否由任意实例处理？

网关位于客户端与后端 MCP 服务器之间。它提供单一 MCP 端点，应用横切策略，并转发已批准的请求。

较旧的网关设计经常把一个客户端会话多路复用到多个后端会话，并重写 `Mcp-Session-Id`。这属于旧版兼容设计；2026-07-28 核心协议没有协议会话。

## 核心概念

### 现代网关路径

对每个请求执行以下步骤：

1. 从传输授权信息中认证主体。
2. 验证 `MCP-Protocol-Version`、`Mcp-Method`、`Mcp-Name` 和 `params._meta`。
3. 授权主体、资源、方法、工具和参数。
4. 应用描述符、注册表、速率与数据策略。
5. 为所选后端创建全新且自包含的请求。
6. 验证后端结果并返回网关结果。
7. 记录不包含秘密的审计事件。

任何步骤都不需要隐藏的协议会话。应用状态仍可以存在于数据库、显式句柄、Tasks 或受完整性保护的 MRTR 状态中。

### 运行时策略是网关的主要决策

准入决定哪个后端版本可以进入网关，但它并不授权实时调用。对于每个请求，网关都要根据已认证主体、签发方与资源、租户、匹配的方法和名称、规范化参数、已准入的描述符固定值、当前后端健康状态、能力交集、数据分类、速率状态，以及任何绑定操作的审批，重新计算策略。

这个顺序很重要。即使用户角色已撤销，Registry 记录仍可能保持活跃；即使描述符仍被固定，目标参数也可能跨越租户边界；即使后端仍获批准，事件响应策略也可能隔离状态变更调用。因此，运行时策略才是主要的允许或拒绝决策，而 Registry 与描述符证据只是其输入。

不要用连接标识符或已移除的会话标识符缓存允许决策。策略不可用时，应按操作类别遵循已声明的失败策略。安全的默认做法是对状态变更和敏感读取采取故障关闭；只有风险模型允许时，明确批准的公共读取路径才能短暂使用最后已知策略。记录促成决策的策略版本和失败路径，并在返回后端结果前对其进行验证。

### 单一 POST 端点

现代 Streamable HTTP 通过 POST 发送每条 JSON-RPC 消息：

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

网关可以为该 POST 返回 JSON 或请求作用域 SSE。现代请求中的 GET 与 DELETE 返回 405。`Mcp-Session-Id` 和 `Last-Event-ID` 不会创建权限、亲和性或重放行为。

请求头与正文中的值必须一致。在查找后端之前，以 `-32020` 拒绝不匹配。这样负载均衡器、网关和速率限制器无须解析完整正文即可路由，同时仍能维持端到端完整性。

验证必须采用唯一且精确的顺序：JSON-RPC 和元数据类型、请求头与正文相等性，然后是匹配版本的支持情况。不匹配时返回 HTTP 400 和 `-32020`。如果请求头与正文一致但版本不受支持，则返回 HTTP 400 和 `-32022`，且 `data` 必须精确为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 和 `-32601`。

`ProtocolError` 携带可选的 `data`，网关会把它序列化进 JSON-RPC 错误对象。通知没有 `id`，因此永远不会收到 JSON-RPC 成功或错误响应。已接受的 HTTP 通知返回 202 和空响应体。

### 在每一层实现发现

网关为客户端实现 `server/discover`。它还会发现每个后端，以了解协议版本、能力和扩展。

网关结果示例：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

只声明网关能够端到端履行的能力交集。后端功能不一定适合直接公开；没有后端路径的网关功能也不值得声明。

`serverInfo` 是服务器自行声明的显示与诊断数据。不要把它用作注册表或发布者证明。

### 逐请求客户端能力

每个转发请求都需要当前的 `_meta` 信封：

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

不要盲目把外层客户端能力复制给后端。对于后端而言，网关就是客户端；只应声明网关能够正确中介的功能。

### 确定性命名空间

使用稳定的公开名称合并后端工具：

```text
notes.search
notes.create
issues.list
issues.open
```

维护从公开名称到后端及原始工具名的映射。绝不能在冲突时选择第一个或最后一个工具。公开名称是审批与审计合约的一部分，因此更名就是一次迁移。

`tools/list` 必须具有确定性。当可见性因主体而异时，返回 `cacheScope: private`。使用有界的 `ttlMs` 可以降低后端发现负载，同时避免用户特定列表跨授权上下文泄漏。

每个对外公开的工具描述符都包含稳定名称、描述和对象根级别的 `inputSchema`。命名空间不能省略必需的描述符字段。完整列表结果还包含 `resultType`、服务器身份元数据和缓存提示。

### 固定已批准的描述符

准入时，规范化完整描述符，并把其摘要保存在限定公开名称下。列举和调用时，将实时描述符与已批准摘要比较。

如果发生变化：

- 从 `tools/list` 中移除。
- 拒绝直接调用。
- 发出审计事件。
- 要求策略或人工重新批准后，才能更新固定值。

网关是非常有用的集中执行点，但它不会让首次发现的描述符自动变得安全。初始审查仍不可少。

### 注册表帮助发现，而不替你决策

Registry 的 `server.json` 提供发布元数据。一个由软件包支持的记录可能如下：

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

发布元数据不承载网关的安全决策。应在独立的准入状态中保存已验证的发布者和来源证据：

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

网关检查 `server.json` 的结构，并将其与该外部状态关联。即便如此，网关仍需要准入策略。

为每个获准后端记录：

- 精确的注册表与记录标识符。
- 已验证的发布者命名空间或域名证据。
- 允许的传输方式与端点。
- 固定版本或获批的升级策略。
- 工件或描述符摘要。
- 授权签发方与资源。
- 审查者、批准时间和过期时间。

不要因为服务器显示名称像某个熟悉产品就接受它，也不要把出现在注册表中当成运维安全审查。即使私有服务器从未出现在公共注册表中，也可以通过同一套证据模式准入。

本课实现网关接缝：先将发布证据与本地准入状态关联，后端才能变得可路由。[第 30 课：MCP 注册表供应链、准入、漂移与回滚](../../30-mcp-registry-supply-chain-and-drift/docs/zh.md)构建完整控制平面，覆盖精确的命名空间证明、工件来源、不可变固定值、实时描述符漂移、Registry 状态对账、防篡改准入台账，以及有证据支撑的回滚。应将这套供应链状态与前述逐请求运行时决策分开。

### 凭据中介

网关认证其调用方，并另外向后端进行认证。后端凭据绝不会提供给客户端。

显式维护以下绑定：

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

绝不能把外层网关令牌传给后端，也不能在不同签发方或资源处复用后端令牌。如果工具代表终端用户行事，应通过专门设计的交换或声明模型保留委托关系，而不是用共享服务凭据冒充用户。

### 无会话的速率限制

按已认证主体、签发方、资源、公开工具、成本类别和时间窗口设置限制键。会话 ID 已不存在；即使存在，也很容易轮换。

在消耗昂贵工作前应用低成本验证。明确被拒绝的调用计入滥用限制、业务配额，还是同时计入两者。

### 审计决策链

记录足以重建一次调用的信息：

- 请求与追踪标识符。
- 已认证主体与签发方。
- 公开工具与后端路由。
- 描述符固定版本。
- 策略决策及原因。
- 延迟与结果类别。
- 适用时记录 MRTR 轮次或任务标识符。

对 bearer token、授权码、刷新令牌、原始秘密和非必要的敏感参数进行脱敏。

### 请求作用域 SSE

当工作在单次请求期间进行流式传输时，普通 POST 可以返回请求作用域 SSE。关闭响应流会取消这个正在进行的现代 HTTP 请求。

不要创建单独的 GET 流，也不要承诺 Last-Event-ID 重放；这些属于旧版传输假设。

### 长期变更通知

对于列表和资源变更通知，当前客户端通过 POST 发送 `subscriptions/listen` 并接收 SSE 响应。通知过滤器使用精确的扁平字段 `toolsListChanged`、`promptsListChanged`、`resourcesListChanged` 与 `resourceSubscriptions`：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

首个事件确认受支持的子集。其订阅标识符就是打开该流的请求所使用的 JSON-RPC ID：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

之后网关只转发已确认的变更类型。该流上的每条通知都携带相同的 `io.modelcontextprotocol/subscriptionId`，该字段位于 `params._meta` 中。协议不会自动重放或自动重新监听。重连后，客户端重新打开订阅并刷新自己依赖的列表。服务器主动优雅关闭时，返回一个带同一订阅 ID 的最终 complete 结果。

现代路径取代了 `resources/subscribe`、`resources/unsubscribe` 和未经请求的独立 GET 流。仅在按版本门控的旧版路径中保留这些机制。

### 经网关传递 MRTR

当后端返回 `resultType: input_required` 时，只有外层客户端支持所需的输入请求，网关才能转发该结果。应逐字节保留 `requestState`，除非网关有意终止并重新签发这次交互。

客户端使用全新的 JSON-RPC ID 和 `inputResponses` 重试原始公开工具。网关重新授权重试、检查同一公开路由，再转发全新的后端请求。它不得假设较早轮次已经授予无限授权。

### Tasks 扩展路由

Tasks 是官方扩展，标识符为 `io.modelcontextprotocol/tasks`。它不是核心会话的替代品。

客户端在逐请求客户端能力中声明该扩展；只有能够端到端维持其生命周期时，网关才在发现结果中声明它。对于受支持的 `tools/call`，只有后端决定返回普通结果还是 `resultType: task`。任务结果直接携带 `taskId`、`status`、时间戳、`ttlMs` 和可选的 `pollIntervalMs`。发送结果前，必须已能持久读取该任务。

网关为不透明任务标识符记录已认证主体与后端路由。后续 `tasks/get`、`tasks/update` 和 `tasks/cancel` 调用使用 `params.taskId` 作为 `Mcp-Name`，为中间组件提供路由键。`tasks/get` 返回 `resultType: complete` 以及当前任务状态，并在终态中内联最终结果或协议错误。`tasks/update` 为待处理任务输入发送带键的 `inputResponses`，并返回空的 complete 确认。`tasks/cancel` 表示协作式意图并返回空的 complete 确认，不保证工作一定停止。

不要实现新的 `tasks/list` 或 `tasks/result` 方法；它们属于旧的实验模型。需要输入的任务通过 `tasks/get` 公开完整的嵌入式请求，客户端经由 `tasks/update` 回答，而不是重试原始工具调用。客户端仍按建议间隔轮询；任务创建仍由服务器决定。

持久任务路由状态是以任务句柄为键的应用数据，而不是协议会话。

### 兼容性边界

如果网关必须服务旧版客户端或后端：

- 显式检测协议时代。
- 将初始化、传输会话、GET 流、资源订阅和旧版任务词汇限制在旧版适配器内。
- 绝不让旧版会话 ID 泄漏到现代路由或授权中。
- 优先使用有界发现探测和显式后备策略，而不是静默降级。

```figure
t3-gateway-funnel
```

## 动手构建

`code/main.py` 实现一个进程内协议网关和两个后端服务器。每个后端都会收到一项全新的当前协议请求。网关提供发现、按用户过滤且具有确定性的 `tools/list`、命名空间路由、Registry `server.json` 加外部准入状态、描述符固定、RBAC、按主体设置键的速率限制、审计决策，以及建模后的 `subscriptions/listen` SSE 确认。

模型接收已解析的请求正文、路由请求头和经过认证的 bearer 身份。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或完整的 `Accept` 合约。应将它接入第 09 课的 Streamable HTTP 适配器；该适配器要求 `Content-Type: application/json`，并要求 `Accept` 值同时包含 `application/json` 与 `text/event-stream`。

运行：

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

演示会打印外层请求 ID 和全新的后端请求 ID，让无状态跳转清晰可见。

## 实际使用

将进程内后端对象替换为真正的当前协议客户端，并保持以下接缝不变：

- 连接前先有准入记录。
- 暴露能力前先发现后端。
- 授权前先得到限定公开名称。
- 列举或调用前先验证描述符固定值。
- 转发前先生成全新的逐请求元数据。
- 返回前先验证结果。

## 交付成果

本课交付 `outputs/skill-gateway-bootstrap.md`。它会产出一个现代网关设计，覆盖入口、发现、准入、命名空间、授权、缓存、流式传输、订阅、MRTR、Tasks、可观测性和旧版隔离。

## 练习

1. 在外层与转发请求元数据中添加追踪上下文，并在审计事件中记录其关联关系。
2. 添加支持 Tasks 的后端，并路由 `tasks/get`，其任务 ID 位于 `Mcp-Name` 中。
3. 更改一个后端描述符，证明发现和直接调用都会被阻止。
4. 添加主体特定的服务器能力，并解释为何发现结果必须采用私有缓存。
5. 编写旧版适配器接口，但不向现代 `Gateway` 类添加任何旧版状态。

## 关键术语

| 术语 | 含义 |
|------|---------|
| MCP 网关 | 位于客户端与后端 MCP 服务器之间的策略与路由服务器 |
| 准入记录 | 允许某个后端进入网关的证据和策略决策 |
| 限定工具名 | `notes.search` 这样的稳定公开路由 |
| 描述符固定 | 在发现和分派期间检查的已批准摘要 |
| 私有缓存作用域 | 限制在一个授权上下文内的缓存结果 |
| 请求作用域 SSE | 附着于单个 POST 请求的流式响应 |
| `subscriptions/listen` | 由客户端打开、用于选定长期变更通知的 SSE 流 |
| 任务路由 | 从不透明任务 ID 到其后端的应用映射 |
| 旧版适配器 | 为旧握手与会话行为提供的显式版本门控边界 |

## 延伸阅读

- [Streamable HTTP 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [官方 Registry server.json 要求](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks 扩展](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
