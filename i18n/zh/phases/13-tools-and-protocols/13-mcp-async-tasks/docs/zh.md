# MCP Tasks 扩展：无状态核心之上的持久任务

> MCP 无状态并不意味着每项操作都必须在单次请求内完成。官方 Tasks 扩展为长时间运行的工作提供了明确、持久的句柄。服务器可从 `tools/call` 返回该句柄，任意实例都能响应 `tasks/get`，客户端则通过 `tasks/update` 提交输入，无须恢复协议会话。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 09 课（传输）、第 13 阶段 · 第 11 课（无状态 MRTR）、第 13 阶段 · 第 12 课（信息征询）
**Time:** 约 90 分钟

## 学习目标

- 区分无状态的协议传输与持久的应用任务状态。
- 在逐请求能力中协商 `io.modelcontextprotocol/tasks` 扩展，并通过 `server/discover` 发现它。
- 仅在任务已持久化创建后，返回由服务器决定的 `CreateTaskResult`，其中带有 `resultType: "task"`。
- 使用 `tasks/get` 轮询，通过 `tasks/update` 补充任务输入，并通过 `tasks/cancel` 请求协作式取消。
- 移除旧有的 `tasks/status`、`tasks/result` 和 `tasks/list` 假设。
- 通过 POST 响应的 SSE 流，在 `subscriptions/listen` 上订阅可选的任务通知。
- 正确建模任务过期、重启恢复、输入键去重以及执行错误。

## 为什么 Tasks 是扩展

Tasks 最早在 2025-11-25 作为实验性核心功能出现。2026 年 7 月的重新设计将其移入官方 `io.modelcontextprotocol/tasks` 扩展，这样客户端和服务器可以自行选择是否启用额外的生命周期，而不必为所有使用者扩张核心协议。

尽管 Tasks 当前已归属官方扩展，但扩展规范本身仍处于草案阶段。请固定 SDK 所支持的扩展版本，运行一致性场景，并将线协议适配器与工作器及存储领域隔离开来。

当一项操作具备以下一个或多个特征时，应使用任务：

- 它的运行时间可能超过普通请求的超时时间。
- 已有工作队列或外部作业系统负责执行。
- 客户端需要在自身重启后恢复操作。
- 操作会在执行期间暂停，以等待用户或模型输入。
- 产品要求支持取消和持久化结果检索。

不要为成本很低的确定性查询创建任务。句柄、持久化、轮询、过期和取消都会带来真实的复杂度。

## 无状态核心，有状态应用

MCP 2026-07-28 移除了 `initialize`、`notifications/initialized`、协议会话以及 `Mcp-Session-Id`。这并不禁止产品本身维护状态。

任务 ID 是显式的应用状态：

- 服务器在返回任务之前将其持久化。
- 客户端可以保存它，并在重启后再次轮询。
- 只要副本由同一持久化存储支持，该 ID 就可以路由到任意副本。
- 每次调用任务方法时都要检查授权。
- 过期与删除由任务字段定义，而不是由传输生命周期决定。

从运维角度看，这与绑定在某个连接上的隐式状态截然不同。

应将以下四种生命周期分开：

| 状态 | 生命周期 | 应归属的位置 |
|---|---|---|
| 协议元数据 | 一次请求 | `params._meta`，每次调用时都重新验证 |
| 传输层工作 | 一次 stdio 请求或 HTTP 响应 | 具有有界截止时间的进行中协调器 |
| MRTR 延续 | 一次重试序列 | 受完整性保护的 `requestState`，并在需要时配合重放控制 |
| 持久任务 | 跨请求、副本、重启与重连 | 以已授权 `taskId` 为键的共享应用存储 |

把任务记录放进进程内存并不会让 MCP 变成有状态协议，只会让应用变得不可靠。协议仍是无状态的，但后续 `tasks/get` 若被路由到另一个副本，就无法找回该记录。应先持久化再返回句柄，并让每个任务方法都在租户与主体检查之后解析到同一条共享记录。

## 能力协商

客户端在每个符合条件的请求中声明支持：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

服务器返回精确的 `supportedVersions`、能力、`ttlMs` 和 `cacheScope`；这些内容来自 `server/discover`，且能力中包含同一个扩展。服务器既然声明支持工具，也必须实现必需的 `tools/list`。该结果返回确定性的 `generate_report` 描述符、有效的对象 `inputSchema`、`resultType: "complete"`、服务器身份元数据以及公共缓存提示。

如果客户端没有声明该扩展却调用任务方法，服务器返回 `-32021`（Missing Required Client Capability），并将 `data.requiredCapabilities` 设为 `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`。不受支持的协议字符串返回 `-32022`，其中包含精确的 `supported` 和 `requested` 数据；缺少版本或版本不是字符串则返回 `-32602`。

不带 JSON-RPC `id` 的信封是通知。接收方可以处理它，但不会发出 JSON-RPC 结果或错误。Streamable HTTP 适配器对已接受的通知返回无响应体的 `202 Accepted`。

目前只有 `tools/call` 支持任务增强型执行。设计内部抽象时，应保证未来新增请求类型不必重写存储层。

## 由服务器决定的任务创建

旧的客户端标志 `params._meta.task.required` 已被移除。客户端声明支持扩展，随后由服务器决定某次具体的 `tools/call` 是否转为任务。

请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

在针对该 ID 的 `tasks/get` 已能成功解析之前，服务器不得返回此句柄。使用最终一致性存储时，应在响应前等待读取可见性；否则客户端可能收到一个看似有效的 ID，却立即遇到“未找到”。

任务响应在某种意义上是未经请求的，因为客户端并未要求进入任务模式。但它并非未经协商：当前请求仍必须声明该扩展。

## 任务结构

每项任务都包含：

- `taskId`：由服务器生成的稳定标识符；
- `status`：`working`、`input_required`、`completed`、`cancelled` 或 `failed`；
- `createdAt` 和 `lastUpdatedAt`：ISO 8601 时间戳；
- `ttlMs`：从创建时起计算的过期时长，若未声明限制则为 `null`；
- 可选的 `pollIntervalMs`：服务器当前建议的最短轮询间隔；
- 可选的 `statusMessage`：面向用户或模型的上下文。

仅在相关状态下才会出现特定字段：

- `input_required` 包含 `inputRequests`。
- `completed` 包含与原始请求相同结构的 `result`。
- `failed` 包含一个 JSON-RPC `error` 对象。

客户端应遵守 `pollIntervalMs`。服务器可以对更频繁的轮询实施速率限制，也可以在任务生命周期内调整该间隔。

## 使用 `tasks/get` 轮询

客户端请求当前快照：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` 本身已经完成，所以其结果始终带有 `resultType: "complete"`。嵌套的任务仍可能处于 `status: "working"` 或 `status: "input_required"`。

这种区分可避免一种常见的解析器错误：

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

不存在 `tasks/result` 调用。任务完成后，下一次 `tasks/get` 响应会把原始 `CallToolResult` 内联到 `result` 下：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

外层 `resultType` 表示 `tasks/get` RPC 已完成；嵌套的 `result.resultType` 表示原始工具调用已完成。这个嵌套判别字段是必需的。嵌套的 `CallToolResult` 还应该携带自己的 `io.modelcontextprotocol/serverInfo`；本课选择包含它，而不是存储一段无类型的负载。

不存在 `tasks/list`。无会话服务器无法安全推断哪些任务应属于某个连接作用域的列表。需要历史记录的应用应公开经过授权的领域工具，并提供明确的过滤条件和所有权规则。

## 任务执行期间的输入

任务输入与核心 MRTR 看起来相似，但使用的是不同的延续机制。

### 创建任务前需要输入

返回核心协议的 `resultType: "input_required"`，该结果来自原始的 `tools/call`。客户端满足输入请求后，重试原始调用。只有这些同步 MRTR 轮次完成后，才创建任务。

### 创建任务后需要输入

将任务设为 `input_required`。`tasks/get` 会公开尚未满足的 `inputRequests`，客户端通过 `tasks/update` 发送响应，而不会重试原始的 `tools/call`。

快照：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

更新：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

成功响应是一个空确认，并带有 `resultType: "complete"`。状态变更可能最终一致，因此客户端应继续轮询或监听。

每个 `inputRequests` 键在任务的整个生命周期内都必须唯一。重复的 `tasks/get` 快照可能显示同一个待处理键；客户端应在 UI 中去重，服务器则忽略针对未知、已取代或已完成键的响应。部分更新可以让任务继续保持 `input_required`，直至所有必需键都得到响应。

## 取消是协作式的

`tasks/cancel` 表达取消意图，并返回一个空的 complete 确认。该确认并不保证工作器已经停止；工作可能抢先完成、忽略取消，或稍后才发生状态转换。

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

对于全部三个任务方法，`Mcp-Name` 都映射 `params.taskId`，而不是重复 JSON-RPC 方法名。`code/main.py` 在 `make_http_request` 中集中实现了这条规则。

本课的工作器会立即响应取消，因此重复调用具有幂等性。但生产环境客户端仍须把取消视为协作式行为，不能根据这条确认就推断任务已处于最终状态。

不要用 `notifications/cancelled` 取消任务。该通知用于取消请求，而非取消持久 Tasks。

这一区别在路由边界上至关重要。请求取消针对的是某个进行中的 JSON-RPC 操作或其请求作用域内的 HTTP 响应。若 `tools/call` 已返回 `resultType: "task"`，该请求就已经结束，关闭其传输连接既无法指明、也无法停止持久作业。`tasks/cancel` 是一项新的、经过授权的 RPC：它携带 `params.taskId`，在 `Mcp-Name` 中映射该 ID，解析任务所属的后端，记录协作式取消意图，并在不宣称工作器已经停止的前提下返回确认。

因此，网关必须把请求协调器和任务路由保存在不同的表中。请求表可以在响应完成时消失，任务路由则必须存活到终态以及保留期结束。[第 29 课：MCP 可靠性、取消与流量控制](../../29-mcp-reliability-cancellation-and-flow-control/docs/zh.md)为两条路径建立竞态、超时、幂等性、背压与重试规则。

## 可选通知

轮询是基线方案。希望获得推送更新的客户端可携带任务 ID 发送 `subscriptions/listen`。对于 Streamable HTTP，这是一个 POST 请求，其响应为请求作用域内的 SSE 流。这里不存在独立的 GET 事件流，也没有需要保持活跃的协议会话。

服务器使用 `notifications/subscriptions/acknowledged` 确认已接受的 ID，随后可通过 `notifications/tasks` 发送完整快照。确认通知和每条任务通知都携带 `io.modelcontextprotocol/subscriptionId`，该字段位于 `_meta` 中，其值等于 `subscriptions/listen` 的请求 ID。除此之外，每条任务通知都等同于该时刻 `tasks/get` 会返回的内容。

客户端仍必须声明 Tasks 扩展。客户端应依靠持久任务 ID 重连并恢复，而不是依赖事件重放或 `Last-Event-ID`。

## 失败语义

应正确区分并使用两层错误语义。

### 协议错误

无效的方法参数或未知任务 ID 会返回 JSON-RPC 错误，通常为 `-32602`。缺少扩展支持时返回 `-32021`，并附上所需的能力对象。

### 任务执行结果

- 带有 `isError: true` 的普通工具结果仍对应 `completed` 任务，因为工具调用已经产出了其定义的结果。
- 延迟执行期间发生 JSON-RPC 错误会使任务进入 `failed`，并把该 JSON-RPC 错误存入 `error`。
- 用户拒绝可以产生 `cancelled`、一个表示拒绝但已完成的结果，或其他领域特定的安全结果；应明确记录所选语义。

## 持久性、过期与所有权

至少应持久化任务 ID、状态、时间戳、TTL、轮询间隔、原始操作的所有权、结果或错误、待处理输入请求，以及发放过的全部输入键。

存储键必须包含权威的租户与主体信息，或能够解析出这些信息。仅知道任务 ID 不应获得访问权限。每次 `tasks/get`、`tasks/update`、`tasks/cancel` 和订阅操作都要检查所有权。

`ttlMs` 从创建时起计算，且可以变化。当任务不再产生可观察更新时，客户端可以把它视作最后保障。服务器可能发生故障，并在恢复后删除已经过期的任务。不要将其描述为“任务完成后承诺保留结果这么多毫秒”。

使用原子写入或事务。本课先写临时文件，再通过原子重命名替换。多副本服务应使用共享持久化存储，并配合工作器租约或等效的并发控制。

```figure
tp-task-lifecycle
```

## 动手构建

`code/main.py` 实现了一个确定性的任务服务：

- `server/discover` 返回 `supportedVersions`、缓存提示和 Tasks 扩展。
- `tools/list` 返回一个确定且可缓存的 `generate_report` 描述符，并附带有效的输入模式。
- `tools/call` 在返回 `resultType: "task"` 之前创建并持久化任务。
- 新服务实例重新加载同一任务，以演示重启恢复。
- `tasks/get` 返回完整的任务快照。
- 工作器将状态从 `working` 推进到 `input_required`。
- `tasks/update` 接收表单响应并返回空的 complete 确认。
- 工作器存储嵌套的 `CallToolResult`（其中包含自身的 `resultType` 和服务器身份），随后转换到 `completed`。
- 本实现中的 `tasks/cancel` 具有幂等性。
- HTTP 构造器将 `Mcp-Name` 设为 `params.taskId`，这条规则适用于 `tasks/get`、`tasks/update` 和 `tasks/cancel`。
- 通知辅助函数使用 `notifications/subscriptions/acknowledged` 和 `notifications/tasks`，二者都标记监听请求的 ID。
- 不带 ID 的通知不会产生 JSON-RPC 响应。

工作器通过显式调用推进，而不是在后台线程中休眠。这样每次状态转换都具有确定性，同时也让协议示例与队列机制彼此分离。

## 实际运行

从仓库根目录执行：

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

预期结果序列：

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

还要验证 `tasks/status`、`tasks/result` 和 `tasks/list` 在现代服务中都会返回“method not found”。
验证 `tools/list` 是确定性的，并且当前的每个 HTTP 任务方法都通过 `Mcp-Name` 映射其任务 ID。

## 交付成果

`outputs/skill-task-store-designer.md` 现在会产出理解扩展语义的设计，覆盖能力协商、返回前持久化创建、当前方法、输入更新流程、所有权、过期、取消、订阅，以及从已移除实验方法迁移的方案。

## 练习

1. 添加第二个待处理输入键。发送一次部分 `tasks/update`，证明在两个键都得到响应前，任务仍保持 `input_required`。
2. 为存储添加租户所有权，并拒绝由错误认证主体提交的有效任务 ID。
3. 添加带过期时间的工作器租约，证明两个服务实例不能并发完成同一任务。
4. 为 `subscriptions/listen` 实现 POST 响应 SSE 适配器。不要添加 GET、`Last-Event-ID` 或会话头。
5. 添加过期清理。在不泄露跨租户存在性的前提下，区分过期任务和格式错误的任务 ID。

## 关键术语

| 术语 | 在当前扩展中的含义 |
|------|----------------------------------|
| Tasks 扩展 | 用于持久异步工作的可选 `io.modelcontextprotocol/tasks` 能力 |
| `CreateTaskResult` | 服务器针对符合条件的请求主动返回的 `resultType: "task"` 响应 |
| `tasks/get` | 轮询完整的当前任务快照，包括终态结果或待处理输入 |
| `tasks/update` | 为任务中尚未满足的 `inputRequests` 提交响应 |
| `tasks/cancel` | 确认协作式取消意图 |
| `input_required` | 表示客户端仍需提供输入的任务状态 |
| `pollIntervalMs` | 服务器建议的再次轮询前最短等待时间 |
| `ttlMs` | 从任务创建时开始计算的过期时长 |
| 返回前持久化 | 发送句柄前必须确保任务 ID 已可解析的规则 |
| `notifications/tasks` | 通过已订阅 SSE 响应传递的可选完整任务快照 |

## 旧版兼容性

2025-11-25 的实验接口采用由客户端请求的任务增强，并使用 `tasks/status`、`tasks/result` 和可选的 `tasks/list`。这些名称只能保留在固定版本的旧版适配器中。当前客户端应使用扩展能力，接受服务器主动返回的句柄，轮询 `tasks/get`，通过 `tasks/update` 提供输入，并从任务快照读取最终结果。

## 延伸阅读

- [官方 MCP Tasks 扩展](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
