# MCP 传输：stdio 与无状态可流式 HTTP

> 传输层负责承载 MCP 消息，却不会补齐缺失的协议状态。在 `2026-07-28` 中，本地 stdio 与远程可流式 HTTP 承载的都是自描述请求。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 07、08 课（MCP 服务器与客户端）
**Time:** 约 65 分钟

## 学习目标

- 为本地子进程选择 stdio，为网络服务选择可流式 HTTP。
- 实现现代的单端点、仅 POST 可流式 HTTP 契约。
- 对照 JSON-RPC 正文镜像并验证 MCP 版本、方法和名称请求头。
- 正确传递请求范围内的 SSE，以及长连接 `subscriptions/listen` 数据流。
- 迁移基于会话和旧版 HTTP+SSE 的部署，同时不把旧版行为包装成现代协议。

## 问题

早期的可流式 HTTP 修订版把协议协商与连接、会话行为混在一起。服务器可以签发 `Mcp-Session-Id`、公开独立的 GET 数据流、接受 DELETE 来终止会话，还可以使用 `Last-Event-ID` 恢复 SSE。

MCP `2026-07-28` 从现代线路协议中移除了这些机制。每个请求的正文都携带协议版本和客户端能力，因此可以落到任意健康工作器。HTTP 请求头会镜像部分选定字段，供路由和策略使用；但服务器必须在执行前，对照正文验证这些请求头。

由此得到的系统更容易扩展，也更容易推理。这还意味着：如果服务器仍把 2025 年的传输方式当作当前协议来讲解，它所教授的故障模型和安全模型就是错误的。

## 概念

### stdio

stdio 绑定用于客户端启动的子进程：

- 客户端向 stdin 写入 UTF-8 编码的 JSON-RPC 消息，每行一条。
- 服务器向 stdout 写入 UTF-8 编码的 JSON-RPC 消息，每行一条。
- 服务器把诊断信息写入 stderr。
- stdin 到达 EOF 时，服务器立即退出。
- 每个现代请求都在 `params._meta` 中携带版本与客户端能力。

进程可以存活并处理很多次调用，但它并不是现代协议会话。若进程意外退出，所有在途请求都会丢失。应重启进程、重新发现、重新列举、重新打开订阅，并为可安全重试的操作使用新的请求 ID。

### 2026-07-28 中的可流式 HTTP

现代服务器公开一个接受 POST 的 MCP 端点，例如 `/mcp`。

每个 JSON-RPC 请求或通知都是一次新的 HTTP POST，正文中只包含一条 JSON-RPC 消息。客户端不会向服务器发送 JSON-RPC 响应。

对于请求，服务器会返回以下两者之一：

- `Content-Type: application/json`，其中包含一条 JSON-RPC 响应；或
- `Content-Type: text/event-stream`，先包含与该请求相关的通知，最后给出 JSON-RPC 响应。

对于已接受的通知，服务器返回不带正文的 `202 Accepted`。

客户端通过以下请求头表明支持两种响应类型：

```http
Accept: application/json, text/event-stream
```

### 仅 POST 就是只允许 POST

现代可流式 HTTP 没有独立的 GET 数据流，也没有用于会话的 DELETE 端点。

- `GET /mcp` 返回 `405 Method Not Allowed`。
- `DELETE /mcp` 返回 `405 Method Not Allowed`。
- `Mcp-Session-Id` 会被忽略，服务器既不签发也不回显它。
- `Last-Event-ID` 会被忽略，因为现代数据流不支持恢复。

如果请求范围内的数据流在最终响应到达前中断，客户端就失去了该在途请求。重试安全时，可以使用新的 JSON-RPC ID 发起新请求，但绝不能尝试恢复原数据流。

### Origin 验证

服务器应验证传入连接的 `Origin`，以防止 DNS 重绑定。如果该请求头存在但未被明确允许，则返回 `403 Forbidden`。非浏览器客户端可以省略 `Origin`，官方传输规则允许这样做。

本地服务器应绑定到 `127.0.0.1`，而不是所有网络接口。网络服务仍需要在每个请求上执行身份认证与授权。Origin 验证不等于身份认证。

完成规范化配置后，应精确匹配来源。诸如 `origin.startswith("https://trusted.example")` 的前缀检查并不安全，因为它可能接受由攻击者控制的后缀。

### 必需的 HTTP 元数据请求头

每个现代 POST 请求都包含：

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

请求头规则如下：

- `MCP-Protocol-Version` 是必需字段，且必须等于 `params._meta.io.modelcontextprotocol/protocolVersion`。
- `Mcp-Method` 是必需字段，且必须等于 JSON-RPC `method`。
- `Mcp-Name` 对于 `tools/call`、`resources/read` 和 `prompts/get` 是必需字段。
- `Mcp-Name` 等于 `params.name` 或 `params.uri`，后者用于 `resources/read`。
- 请求头名称不区分大小写，但请求头值区分大小写。

不安全或非 ASCII 的 `Mcp-Name` 值使用精确的 UTF-8 Base64 哨兵格式：

```text
=?base64?{Base64EncodedValue}?=
```

服务器在对照正文之前，应先解码该值。

缺失、格式错误或不匹配的镜像请求头，应返回 HTTP `400`，并使用 JSON-RPC 错误码 `-32020`。如果请求头与正文中的版本一致，但服务器不支持该版本，则返回 HTTP `400` 和 `-32022`，同时提供精确错误数据，例如 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。

未知的现代方法返回 HTTP `404` 与 JSON-RPC `-32601`。JSON-RPC 正文很重要，因为兼容两个时代的客户端会用它区分现代协议错误与旧版端点缺失。

### 请求范围内的 SSE

服务器可以为单个长时间运行的请求选择 SSE：

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

服务器不得在这条数据流上发送独立的 JSON-RPC 请求。采样、信息征询和根目录交互使用多轮往返请求（Multi Round-Trip Request）结果。关闭响应数据流就会取消该请求。

不要为了重放而添加 SSE 事件 ID。现代修订版不包含 `Last-Event-ID` 恢复机制。

### 长期变更使用 subscriptions/listen

变更通知使用客户端打开的请求，而不是独立 GET：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
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

该 POST 的响应是一条长连接 SSE 数据流。其第一条协议消息是 `notifications/subscriptions/acknowledged`。确认通知、每条变更通知和最终结果都会携带 `io.modelcontextprotocol/subscriptionId`（位于 `_meta` 中），其值等于监听请求 ID。服务器可以发送 SSE 注释作为保活消息。数据流中断后，客户端应使用新的请求 ID 再次发出 `subscriptions/listen`，并重新获取受影响的数据。

`resources/subscribe` 和 `resources/unsubscribe` 属于旧版协议时代，不要在现代连接上使用它们。

### 显式应用状态

移除协议会话并不禁止有状态工作流。服务器可以签发一个不透明状态句柄，把它作为普通工具结果返回；客户端再将该句柄作为后续调用的显式参数。

句柄应绑定到已认证主体，并做到不可猜测、可过期，且每次使用都要授权。这样，状态会明确存在于应用层，而不是隐藏在传输亲和性中。

隐藏的副本状态会以一种必然的方式引发故障：

1. 请求 A 到达副本 1，并在该进程的内存中创建一份草稿。
2. 响应没有返回草稿句柄，因为实现假定连接本身能够标识草稿。
3. 请求 B 是一次新的 POST，因而到达副本 2。
4. 副本 2 拥有有效协议元数据，却无法命名或加载草稿，于是工作流失败或读取了错误的本地对象。
5. 粘性路由看似修复了症状，但重启、发布、重新调度或故障转移一旦把下一个请求移走，问题就会再次出现。

正确边界由两部分组成：协议上下文始终留在每个请求中；持久应用状态存入共享存储，并通过服务器签发、返回给客户端的句柄访问。下一次调用提交该句柄，任何副本都可以加载同一条记录，而授权会把记录绑定到已认证主体与租户。副本内存可以缓存记录，但正确性所需的数据不能只有这一份。

应根据生命周期选择状态机制。请求局部变量只服务一次调用；短期 MRTR 延续可以使用受完整性保护的 `requestState`；草稿或持久任务则需要显式句柄、共享持久化、过期机制、并发控制和幂等性。这些对象都不是 MCP 协议会话。

### HTTP 双时代兼容

同时支持现代和旧版服务器的客户端，应先尝试现代 POST。如果收到 HTTP `400`、`404` 或 `405`，就检查正文：

- 已识别的现代 JSON-RPC 错误能证明服务器属于现代协议。应修正请求，或用服务器公布的版本重试，不要降级。
- 空正文或未识别的响应可能表明这是旧版 HTTP+SSE 服务器。只有此时才尝试旧 GET 端点，并等待旧版 `endpoint` 事件。

迁移期间，服务器可以同时支持两个时代：将带现代元数据的请求路由到现代的仅 POST 实现，同时为旧客户端保留独立的旧版端点。绝不能把旧版 GET、DELETE、会话 ID 或重放行为描述成 `2026-07-28` 的组成部分。

```figure
tp-transport-handshake
```

## 投入使用

`code/main.py` 使用 Python 标准库实现一个有限运行的现代可流式 HTTP 服务器。它验证 Origin 与镜像请求头、忽略已经移除的会话请求头、为普通调用返回 JSON，并演示一条有限的 `subscriptions/listen` SSE 数据流。

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

探测会检查：

- 非法 Origin 会被拒绝；
- 无需会话 ID 即可成功发现；
- `Mcp-Session-Id` 和 `Last-Event-ID` 会被忽略；
- 请求头不匹配会返回 `-32020`；
- 版本不受支持会返回 `-32022`，并带有精确的 `supported` 和 `requested` 数据；
- 已接受且不带 ID 的通知会返回无正文的 HTTP `202`；
- GET 与 DELETE 都返回 `405`；
- `subscriptions/listen` 是一条 POST 响应流，其中的确认通知、变更通知和最终结果都会携带其订阅 ID。

## 交付成果

本课交付 `outputs/skill-mcp-transport-migrator.md`。它会移除现代协议会话、添加请求头与正文的相互验证、用 `subscriptions/listen` 取代独立 GET，并把所有旧版桥接清晰隔离。

## 练习

1. 从 POST 中移除 `Mcp-Method`。确认收到 HTTP `400` 和错误 `-32020`。
2. 在请求头与正文中发送一致的版本 `2027-01-01`。确认收到 HTTP `400`、错误 `-32022`，以及精确数据 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。
3. 为非 ASCII 资源 URI 发送采用 Base64 哨兵格式的 `Mcp-Name`。确认解码后的值会与 `params.uri` 比较。
4. 在最终响应前中断有限监听流。使用新的 JSON-RPC ID 再次发出请求，并重新获取工具。
5. 为 ping 工具添加显式工作流句柄。将它绑定到授权主体，但不要使用连接亲和性。

## 关键术语

| 术语 | 含义 |
|------|---------|
| stdio | 在客户端启动的子进程上，以换行符分隔的 JSON-RPC |
| 可流式 HTTP | 每条现代消息都作为一次新 POST 发往同一端点的传输 |
| 请求范围内的 SSE | 包含相关通知和最终响应的 POST 响应流 |
| `subscriptions/listen` | 为选择接收的变更通知建立的长连接 POST 请求 |
| 请求头不匹配 | 镜像请求头与正文不一致时返回的 HTTP `400` 和 JSON-RPC `-32020` |
| Origin 验证 | 传入连接的 DNS 重绑定防护，并非身份认证 |
| 显式状态句柄 | 作为普通参数传递、用于取代隐藏会话状态的应用令牌 |
| 旧版桥接 | 仅为兼容而保留、与现代实现分离的早期协议时代行为 |

## 延伸阅读

- [MCP 传输概览](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP 可流式 HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP 订阅](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
