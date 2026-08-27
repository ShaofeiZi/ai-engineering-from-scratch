# 模型上下文协议（MCP）

> MCP 为 AI 宿主提供一套统一协议，用于发现和调用工具、资源与提示词。2026-07-28 修订版使该协议变为无状态：能力与版本上下文随每个请求一同传递，而不是保存在与连接绑定的握手中。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 11 · 09（函数调用）、阶段 11 · 03（结构化输出）
**Time:** 约 75 分钟

## 学习目标

- 区分 MCP 宿主、客户端、服务器、传输机制与服务器原语。
- 构建包含 MCP 2026-07-28 所需元数据的 JSON-RPC 请求。
- 使用 `server/discover` 检查版本、身份与能力。
- 从工具、资源和提示词返回类型明确、支持缓存的结果。
- 解释现代无状态 MCP 如何与握手时代的服务器互操作。
- 为服务器选择安全的状态、传输与审批边界。

## 问题

你的应用需要执行数据库查询、日历操作和文件读取。如果没有共享协议，每个 AI 宿主都要为这些相同能力分别编写发现、调用、错误处理、传输与授权的适配代码。

MCP 缩小了这一集成矩阵。服务器发布标准 JSON-RPC 接口；合规客户端无须服务器专用适配器，就能发现该接口、将其呈现给模型或用户、执行调用，并解释结果。

这里有一个很容易忽视的重要边界：MCP 标准化的是通信。它并不决定模型应调用哪个工具，不会让不可信内容自动变得安全，也不会把无状态请求转化为持久应用状态。这些决策仍由宿主和服务器负责。

## 概念

![MCP 宿主、无状态请求与服务器原语](../assets/mcp-architecture.svg)

### 三种服务器原语

1. **工具**是可调用的操作。每个工具都有名称、描述、JSON Schema 输入与处理器。
2. **资源**是带名称、通过 URI 寻址且可供客户端读取的内容。
3. **提示词**是宿主可向用户提供的可复用模板。

宿主就是 AI 应用。宿主内部的一个 MCP 客户端会与一台服务器通信；传输层负责在双方之间传递 JSON-RPC 消息。

### 无状态请求取代握手

MCP 2026-07-28 移除了 `initialize` 与 `notifications/initialized`，同时移除了协议层会话。每个请求都在 `params._meta` 中携带解释该请求所需的上下文：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

协议版本和客户端能力是必需字段，建议提供客户端身份。缺少 `_meta`、缺少必需字段，或必需字段类型错误，都属于格式错误，服务器会返回 Invalid Params（`-32602`）。如果版本字符串格式正确，但服务器不支持该版本，则返回 `UnsupportedProtocolVersionError`（`-32022`）。服务器处理有效请求时，无须恢复之前的协商记录。

无状态并不意味着应用永远不能维护状态，而是状态不会隐藏在 MCP 连接或 `Mcp-Session-Id` 背后。如果工作流需要连续性，服务器应签发一个不透明句柄，客户端在后续调用中把它作为普通工具参数传回。每个请求仍然必须进行授权检查。

### 发现与版本选择

每台现代服务器都实现 `server/discover`。结果会公布支持的版本、能力与服务器身份：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

客户端可以直接调用其他方法，再处理版本错误；但发现操作会让能力展示与版本选择变得明确。不受支持的版本会返回 `UnsupportedProtocolVersionError`，错误码为 `-32022`。其 data 包含 `supported`（服务器修订版数组）和 `requested`（被拒绝的修订版）。

在 stdio 上，兼容新旧两个时代的客户端会先尝试 `server/discover`。收到发现结果，或 `UnsupportedProtocolVersionError` 等可识别的现代错误，都能确认对方是现代服务器。只有错误或超时无法识别为现代协议时，才允许回退到 2025-11-25 的 `initialize` 流程。旧版行为是兼容代码，而不是现代默认行为。

### 结果具有明确状态

2026-07-28 的每个核心结果都有 `resultType`：

- `complete` 表示操作已经完成。
- `input_required` 表示服务器需要通过多轮往返请求模式（Multi Round-Trip Requests）再进行一次交互。核心服务器只能从 `tools/call`、`resources/read` 或 `prompts/get` 返回该状态。

客户端必须把省略 `resultType` 的旧版结果视为已完成。

服务器应在每个结果中包含 `io.modelcontextprotocol/serverInfo`，并将其放在 `_meta` 内。这个身份由服务器自行声明，仅用于展示、日志与调试，不能用于安全决策。

列表与读取结果还会携带 `ttlMs` 和 `cacheScope`。确定性的 `tools/list` 顺序配合新鲜度提示，可以让客户端安全缓存发现结果，并提升提示缓存的稳定性。`cacheScope: public` 允许共享缓存，`private` 则把复用范围限制在当前调用上下文内。

### 报文格式与传输

MCP 通过 stdio 或 Streamable HTTP 使用 JSON-RPC 2.0。

- 请求包含 `jsonrpc`、`id`、`method` 与 `params`。
- 响应包含与请求一致的 `id`，以及 `result` 或 `error`。
- 通知不含 `id`，也不期待响应。

现代 Streamable HTTP 暴露一个接受 POST 的端点。每条 JSON-RPC 消息各自使用一次 POST。请求型 POST 会收到单个 JSON 对象，或一条以最终响应结束、仅属于该请求的服务器发送事件流。服务器接受通知型 POST 后返回 HTTP 202，不带响应正文；这个核心修订版没有定义 Streamable HTTP 上从客户端发送给服务器的通知。

2026-07-28 中不存在独立的 MCP GET 流、DELETE 会话端点、`Mcp-Session-Id` 或 `Last-Event-ID` 重放。长期变更通知使用 `subscriptions/listen` POST，其响应会保持为开放的 SSE 流。

### 不再依赖服务器发起请求的客户端输入

旧版本允许服务器通过数据流发送 `sampling/createMessage`、`roots/list` 或 `elicitation/create` 等请求。当前协议改用多轮往返请求。符合条件的工具调用、资源读取或提示词获取会返回 `resultType: input_required`，并至少包含 `inputRequests` 或 `requestState` 之一。客户端收集所需输入后，使用新的 JSON-RPC ID 和对应的 `inputResponses` 重试原方法；若服务器提供了 `requestState`，客户端必须原样回传。如果结果中没有 `inputRequests`，重试时就省略 `inputResponses`。

Roots、Sampling 和 Logging 仍可使用，但已经弃用，因此新实现不应采用它们。现有的 Roots 或 Sampling 请求应放在 MRTR 的 `inputRequests` 中传递，绝不能作为独立的服务器到客户端 JSON-RPC 请求。应优先使用显式文件或目录参数、资源 URI、服务器配置，以及直接集成模型提供商。stdio 诊断写入 stderr，生产遥测使用 OpenTelemetry。

```figure
mcp-nxm-collapse
```

## 动手构建

### 第 1 步：注册服务器接口

尽管请求契约发生了变化，注册过程仍然简单：

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

`code/main.py` 中随课交付的实现还注册了一个资源和一个提示词。它特意只使用标准库，让你看清每个消息的封装结构，而不是把协议细节交给 SDK 隐藏起来。

### 第 2 步：为每个请求附加元数据

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

不要只把这些元数据缓存到连接对象中。服务器会在每个请求上验证它们。

### 第 3 步：列出能力前可先执行发现

调用 `server/discover`，选择一个受支持版本，再调用 `tools/list`。如果已经知道版本，也可以直接调用 `tools/list` 并处理 `-32022`。

演示会按名称顺序返回工具列表，并附带 `ttlMs`、`cacheScope`、`resultType` 和服务器身份。工具调用返回完整且不可缓存的结果，因为它的输出可能依赖当前状态。

### 第 4 步：把同一请求映射到 HTTP

远程 `tools/call` POST 包含与 JSON-RPC 正文相呼应的标头：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

`MCP-Protocol-Version` 标头必须与 `_meta` 中的版本一致。每个 JSON-RPC 请求都必须包含 `Mcp-Method`，而且它必须与 `method` 一致。`Mcp-Name` 只在调用 `tools/call`、`resources/read` 和 `prompts/get` 时为必需字段，并且必须与工具名称、资源 URI 或提示词名称一致。缺少必需标头或内容不一致时，服务器会返回 HTTP 400，错误为 `HeaderMismatch`，代码是 `-32020`。

### 第 5 步：在协议状态之外落实安全

- 对每个 HTTP 请求验证授权与受众。
- 将本地服务器绑定到 localhost，并在 Streamable HTTP 上验证 `Origin`。
- 为会改变状态的工具标注 `destructiveHint: true`，并要求宿主批准。
- 显式传递目录与文件范围，不依赖已弃用的 Roots。
- 把资源和工具输出视为不可信数据。
- 使用 stdio 时将 stdout 专用于 JSON-RPC，把诊断信息写入 stderr。

## 投入使用

在课程目录中运行：

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

第一行应报告发现 `demo-server`，其协议版本为 `2026-07-28`。然后检查 `MCPClient.request`：它会为每次调用重新构造 `_meta`。从一个请求中移除元数据，观察服务器拒绝该请求。

## 交付成果

`outputs/skill-mcp-server-designer.md` 会把一个领域转化为无状态 MCP 设计。它的验收门禁要求具备发现结果、逐请求元数据策略、确定且可缓存的列表、显式状态句柄、传输标头、授权与审批规则。

## 继续深入学习 MCP

本课为你建立协议模型。阶段 13 将四个生产边界拆成独立的“构建并验证”课程：

1. [MCP 工具契约与内容](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md)介绍封闭输入 Schema、结构化内容、路由元数据、不透明分页、补全授权，以及协议错误与工具领域错误之间的区别。
2. [MCP 可靠性、取消与流量控制](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md)介绍请求取消、持久任务取消、截止时间、幂等性、背压、代理缓冲与重连行为。
3. [MCP 注册表供应链、准入、漂移与回滚](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md)介绍命名空间证明、制品来源、不可变固定版本、实时漂移、注册表状态、准入证据与回滚。
4. [MCP 一致性工程](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md)介绍标准与反例报文记录、严格的版本阶段、SDK 差异测试、代理证据、脱敏、健康门禁与发布回滚。

当服务器将跨越团队或信任边界时，请按顺序学习这些课程。它们共同推动系统从“方法可以工作”走向“契约在部署过程中始终安全且可诊断”。

## 练习

1. 添加一个 `subtract` 工具，并确认 `tools/list` 仍按字母顺序排列。
2. 删除协议版本键，验证 Invalid Params（`-32602`）。然后发送格式正确但不受支持的版本 `2025-11-25`，验证 `-32022`，确认 `requested` 原样返回该修订版，再从 `supported` 中选择版本。
3. 让服务器为创建操作签发一个 `draftId`，并要求更新操作把它作为参数传回。解释这为什么属于应用状态，而不是协议会话。
4. 从需要用户确认的工具返回 `input_required`。用新的 ID、一个 `inputResponses` 条目，以及完全相同的 `requestState` 重试原调用，而不是自行发明服务器到客户端 JSON-RPC 请求。
5. 勾勒一个兼容新旧两个时代的 stdio 客户端。把发现结果或可识别的现代错误视为现代协议，只有发生不可识别的错误或超时时，才允许回退到 `initialize`。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| MCP | “大语言模型工具协议” | 用于服务器发现、工具、资源、提示词及扩展的 JSON-RPC 协议 |
| 宿主 | “AI 应用” | 拥有模型与用户界面，并挂载一个或多个 MCP 客户端 |
| 客户端 | “连接器” | 代表宿主与一台服务器进行 MCP 通信 |
| 无状态 MCP | “没有会话” | 每个请求都携带版本与能力；协议状态不与连接绑定 |
| `server/discover` | “能力探测” | 必需的服务器方法，用于公布版本、能力与身份 |
| `resultType` | “结果状态” | 将结果标记为 `complete` 或 `input_required` |
| 状态句柄 | “工作流 ID” | 由服务器签发、作为普通参数传递的应用标识符 |
| Streamable HTTP | “远程传输” | 使用一个 POST 端点，返回 JSON 或请求级 SSE 响应 |
| MRTR | “询问并重试” | 在结果中嵌入输入请求，然后重试原操作 |

## 延伸阅读

- [MCP 2026-07-28 关键变更](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 已弃用功能](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
