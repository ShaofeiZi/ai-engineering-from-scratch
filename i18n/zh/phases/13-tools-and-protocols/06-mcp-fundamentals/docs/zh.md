# MCP 基础：无状态请求与 JSON-RPC

> 现代 MCP 没有握手，也没有协议会话。每个请求都必须自行携带足够的元数据，以便被理解、授权、路由和重试。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 01～05 课（工具接口与函数调用）
**Time:** 约 55 分钟

## 学习目标

- 区分 MCP 的服务器原语与客户端侧功能。
- 为 MCP `2026-07-28` 构建有效的 JSON-RPC 2.0 请求与响应。
- 为每个请求附加协议版本、客户端能力和客户端身份。
- 使用 `server/discover`，并在没有握手的情况下处理 `UnsupportedProtocolVersionError`。
- 追踪一个独立请求从验证到返回完整结果的全过程。

## 问题

一个 MCP 服务器可以在同一个进程或 HTTP 工作器中连续接收来自不同客户端、具有不同能力的两个请求。如果服务器记住了前一个请求所声明的内容，就可能应用错误的权限或返回错误的报文格式。

MCP `2026-07-28` 消除了这种歧义。协议核心是无状态的。服务器必须根据当前请求本身决定如何处理当前请求，而不能依赖连接历史。

这改变了心智模型。旧流程是先连接、再握手、最后执行操作。现代流程更加简单：

1. 客户端发送一个能够自我描述的请求。
2. 服务器验证该请求的版本与能力。
3. 服务器处理对应方法。
4. 服务器返回带类型的结果或 JSON-RPC 错误。

下一个请求会从头重复同样的过程。

## 概念

### 服务器原语

MCP 服务器公开三种主要原语：

1. **工具**是由模型控制的动作，通过 `tools/list` 发现，通过 `tools/call` 调用。
2. **资源**是通过 URI 寻址的数据，通过 `resources/list` 发现，通过 `resources/read` 获取。
3. **提示词**是可复用模板，通过 `prompts/list` 发现，通过 `prompts/get` 渲染。

Roots、Sampling 和 Logging 为兼容性仍保留在 `2026-07-28` Schema 中，但已经弃用。新实现应当使用显式的工具或资源输入来提供根目录，直接使用模型提供商 API 进行采样，并通过 stderr 或 OpenTelemetry 记录日志。Elicitation 仍可通过多轮往返请求使用：服务器返回输入请求，客户端随后重试原始操作。现代服务器绝不会主动发起独立的 JSON-RPC 请求。

### JSON-RPC 信封

MCP 使用 JSON-RPC 2.0：

- 请求：`{jsonrpc, id, method, params}`
- 响应：`{jsonrpc, id, result}` 或 `{jsonrpc, id, error}`
- 通知：`{jsonrpc, method, params}`，不包含 `id`

请求的 `id` 用于关联一次响应，并不会创建协议会话。

### 必需的请求元数据

每个现代请求都会携带一个 `_meta` 对象，它位于 `params` 内：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

协议版本和客户端能力是必需字段，建议提供客户端身份。客户端身份是自行声明的展示与调试数据，不是安全凭证。

服务器不得仅从先前请求、stdio 进程、HTTP 连接或传输标头推断这些值。

### 完整结果与服务器身份

每个成功的现代结果都包含 `resultType`。普通最终结果使用 `"complete"`。服务器还应在结果元数据中声明自身身份：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`、`resources/list`、`prompts/list`、`resources/templates/list`、`resources/read` 与 `server/discover` 的结果都可以缓存，并包含 `ttlMs` 与 `cacheScope`。安全的默认值是 `ttlMs: 0` 和 `cacheScope: "private"`。列表项应采用确定性顺序，使等价响应产生稳定的缓存键和稳定的模型上下文。

### 无须握手的发现

每台现代服务器都必须实现 `server/discover`。客户端可以在调用其他方法之前使用它获取：

- `supportedVersions`
- 服务器 `capabilities`
- 可选的使用 `instructions`
- 结果 `_meta` 中的服务器身份
- 缓存提示

发现很有用，但它不是门禁。客户端可以首先发送 `tools/list`，因为该请求本身已经携带协议版本与能力。

如果请求的版本不受支持，服务器会返回 JSON-RPC 代码 `-32022`，并附带：

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

客户端选择双方共同支持的现代版本，再使用新的 JSON-RPC 请求 ID 重试。

### 单个请求的生命周期

现代请求按以下顺序处理：

1. 解析一个 JSON-RPC 信封。
2. 确认 `jsonrpc` 为 `"2.0"`、存在 `id`、`method` 是字符串、`params` 是对象。
3. 要求 `params._meta` 中存在版本字符串与能力对象；元数据格式错误或缺失时返回 `-32602`。
4. 在 HTTP 边界，将版本、方法与适用的名称标头同正文比较。即使两处版本值中有一个不受支持，只要二者不一致，就返回 `-32020`。
5. 确认二者相等后，若匹配的版本不受支持，则返回 `-32022`。
6. 检查必需能力，再按 `method` 路由，并验证方法专用参数。
7. 在处理器运行前，对具体操作进行身份认证与授权。
8. 返回包含服务器身份的完整结果。
9. 丢弃仅属于本请求的协议元数据。

这个顺序可以防止两个组件把同一个请求解释成不同调用。网关不得授权 `Mcp-Name: notes.read`，却让源站执行 `params.name: notes.delete`。它还让格式错误输入、标头混淆、版本协商、能力失败、授权失败与处理器失败保留为彼此不同的证据。

关闭 stdin 或结束 HTTP 响应只会终止传输活动，并不会终止协议会话，因为现代 MCP 根本没有协议会话。

### 显式旧版兼容

截至 `2025-11-25` 的版本使用 `initialize`、`notifications/initialized`、连接范围内的能力，以及更早 Streamable HTTP 中可选的协议会话。兼容新旧两个时代的客户端与旧服务器通信时，仍需要这种行为。

必须把两个时代隔离开。现代请求通过必需的逐请求元数据识别；只有通过文档规定的回退路径，才会选择旧版连接。不要把发送 `initialize` 作为 `2026-07-28` 服务器的默认行为。

因此，“无状态”具有特定时代含义。在 `2026-07-28` 中，它是协议不变量：每个普通请求都可独立解释，不存在 MCP 会话。对于截至 `2025-11-25` 的版本，初始化与协商能力属于连接，因此兼容适配器可以保留旧版连接状态。兼容新旧两个时代的实现，不是一个宽松的状态机，而是一个无状态现代核心旁边放置一个隔离的旧版适配器；在任一解析器运行前，必须显式决定选择哪一个。

两种含义都不会禁止持久应用状态。工作流、任务或草稿可以存放在共享存储中，并使用不透明句柄引用。客户端把该句柄作为普通输入发送，每个副本都对它进行身份认证与授权。协议上下文不得泄漏到该存储中，充当已经移除的会话的替代品。

```figure
mcp-tool-call
```

## 投入使用

`code/main.py` 在不使用框架的情况下构建、验证、追踪并分派现代 MCP 消息。运行：

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

观察输出中的三个不变量：

- 每个请求都会重复其 `_meta` 字段。
- 每个成功结果都有 `resultType: "complete"`，并包含服务器身份。
- 列表结果按确定性顺序排列，并带有显式缓存提示。

## 交付成果

本课交付 `outputs/skill-mcp-handshake-tracer.md`。历史文件名保持不变，但这个产物现在是无状态请求追踪器。它会独立审计每条消息，并且只在确实出现旧版握手流量时才进行标记。

## 练习

1. 把一个请求的协议版本改为 `2027-01-01`。确认错误码为 `-32022`，而且 data 会公布受支持版本。
2. 从第二个请求中移除 `io.modelcontextprotocol/clientCapabilities`。确认服务器不会复用第一个请求的能力。
3. 反转内存工具注册表。确认 `tools/list` 仍以相同的确定性顺序返回。
4. 把 `cacheScope` 从 `public` 改为 `private`。解释每种情况下哪些授权上下文可以复用响应。
5. 添加可选的 `clientInfo` 缺失测试。请求仍应有效，因为客户端身份是建议字段，而非必需字段。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 无状态协议 | 每个请求都提供解释自身所需的元数据 |
| 请求元数据 | `params._meta` 中的版本、客户端能力与建议提供的客户端身份 |
| `server/discover` | 公布版本、能力、说明和身份的必需服务器方法 |
| `resultType` | 每个成功现代结果上的判别字段 |
| 可缓存结果 | 包含必需 `ttlMs` 与 `cacheScope` 提示的结果 |
| 协议时代 | 现代逐请求元数据，或旧版连接范围内初始化 |
| 传输生命周期 | 进程、连接或响应流的生命周期，而非协议会话状态 |
| `-32022` | 不受支持的协议版本错误，包含所请求和所支持版本 |

## 延伸阅读

- [MCP 架构](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP 基础协议](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
