# MCP 资源与提示词：无状态服务器中的可寻址上下文

> 工具执行操作，资源公开可寻址内容，提示词则封装由用户选择的消息模板。优秀的 MCP 服务器会让这些契约彼此分离且行为可预测。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 07 课（构建 MCP 服务器）、第 13 阶段 · 第 09 课（MCP 传输）
**Time:** 约 60 分钟

## 学习目标

- 根据消费者意图，在工具、资源和提示词之间做出选择。
- 通过必需的 `server/discover` 公布资源与提示词接口面。
- 构建顺序确定的 `resources/list` 与 `prompts/list` 结果。
- 应用 `ttlMs` 和 `cacheScope`，同时避免泄漏用户专属数据。
- 对无效或未知的资源 URI 返回 JSON-RPC 错误 `-32602`。
- 打开一条 `subscriptions/listen` POST 响应流，并通过订阅 ID 关联每个事件。
- 将资源内容和提示词模板视为不可信的服务器输出。

## 从消费者开始

误用 MCP 最容易发生在一开始就编写实现代码的时候。数据库查询因为函数形式很熟悉而被做成工具；可复用工作流因为存储在文件中而被做成资源；提示词则因为宿主可以注入而变成隐藏策略。

设计时应先问：由谁做选择？选择者期待得到什么？

| 原语 | 主要意图 | 选择方 | 典型结果 |
|---|---|---|---|
| 工具 | 执行操作 | 模型或应用 | 结构化操作结果 |
| 资源 | 读取某个 URI 下的内容 | 宿主、应用或用户 | 文本或二进制内容 |
| 提示词 | 启动可复用的消息工作流 | 用户通过宿主 UI | 一条或多条提示消息 |

位于 `notes://note-1` 的笔记是资源，因为它是可寻址内容。`delete_note` 是工具，因为它会改变状态。`review_note` 是提示词，因为用户会选择一个预先准备好的审阅工作流。

不要仅仅为了显得功能完整，就把同一项操作同时公开为三种原语。每增加一个接口面，都需要相应的发现、授权、缓存、错误处理、测试和文档。

## 2026-07-28 无状态信封

本课面向 MCP 协议修订版 `2026-07-28`。该配置中不存在初始化握手或协议会话。每个请求都通过保留的 `_meta` 键携带协议版本与客户端能力。

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

服务器必须实现 `server/discover`。它的结果会公布受支持的版本、资源与提示词能力、实现身份及缓存提示。客户端可以直接调用其他方法，但发现操作能在客户端构建 UI 之前提供一份稳定快照。

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

普通结果会声明 `"resultType": "complete"`。响应 `_meta` 使用 `io.modelcontextprotocol/serverInfo` 标识提供服务的实现。这些信息有助于诊断，但不是身份认证标识。请求携带不受支持的修订版时，应返回 `-32022`，并同时包含请求的修订版与服务器支持的修订版。

无状态契约会改变设计直觉：列表不能依赖同一连接上的前一次调用。由于凭证属于请求输入，授权可以改变可见集合，但连接历史绝不能产生这种影响。

## 资源是稳定的 URI 契约

资源是由 URI 标识的内容。请先设计 URI，再编写处理器。

良好的 URI 应具备以下属性：

- 足够稳定，可以加入书签或在请求之间传递。
- 使用服务器所属领域的命名空间。
- 不依赖进程 ID 或连接。
- 在访问存储之前完成验证。
- 每次读取都执行授权。

`notes://note-1` 优于 `note-1`，因为前者明确带有命名空间。文件服务器可以使用 `file://` URI，但解析符号链接和相对路径段之后，仍必须检查是否超出配置的目录边界。

`resources/list` 返回调用方当前可见的资源。应按 URI 等稳定键排序。确定性顺序可以避免无意义的缓存未命中、不断变化的快照，以及宿主 UI 每次刷新时项目跳动。

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` 返回一个或多个内容项。未知 URI 不能被当作一次成功的空读取。当前资源规范把无效或未知资源 URI 归为 JSON-RPC 无效参数，错误码为 `-32602`。

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

这种区分让客户端能够把资源不存在与有效的空文档分开，也可以防止意外回退到范围更宽的查找。

### 资源模板

资源模板描述一组参数化 URI。当列出所有具体项目代价过高或数量没有上限时，应使用模板。例如，`notes://projects/{project}/decisions/{decision}` 告诉客户端如何构造有效地址，而无需返回每一条决策。

模板不会放宽验证要求。应解析变量、应用授权、强制执行长度与字符限制，并使用类型化参数构造存储查询。绝不能把任意 URI 尾部直接拼接到文件系统路径或数据库语句中。

### 内容不是可信指令

资源文本可能包含提示注入、秘密、误导性命令或格式错误的标记。宿主应保留来源，并把资源内容视为数据。服务器应限制内容大小、返回准确的 MIME 类型、删去调用者无权访问的字段，并避免返回无关记录。

## 提示词是由用户控制的模板

MCP 提示词面向用户的显式选择而设计。宿主可以把它们呈现为斜杠命令、菜单项或工作流按钮；协议不要求使用某一种 UI。

对于相同的请求授权，`prompts/list` 应保持确定性。每条提示词都需要稳定名称、有用描述，以及能让宿主在调用 `prompts/get` 之前收集输入的参数声明。

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` 会把参数解析为消息，但不会取代宿主的系统指令。宿主决定如何把返回的消息加入模型上下文，并让自己的可信策略保持更高优先级。

应在服务器边界验证提示词参数。提示词中的 URI 必须通过与直接读取资源相同的授权检查，不能让提示词成为绕过资源访问控制的侧信道。

## 缓存提示关系到正确性

`ttlMs` 告诉客户端某个结果可以复用多长时间。`cacheScope` 描述谁可以共享该缓存值。

| 作用域 | 含义 | 典型用途 |
|---|---|---|
| `public` | 授权允许时，可跨用户复用 | 公共提示词目录 |
| `private` | 绑定到发起请求的用户或凭证上下文 | 用户拥有的笔记内容 |

应根据数据变化频率和内容过期造成的损害选择 TTL。五分钟可能适合公共提示词目录，私有笔记读取则可以使用一分钟。

MCP 只把 `public` 和 `private` 定义为 `cacheScope` 的合法值。对于包含秘密或快速变化的结果，应返回 `cacheScope: "private"` 与 `ttlMs: 0`，然后在宿主缓存策略中实施更严格的禁止存储规则。`no-store` 本身不是 MCP `cacheScope` 值。

缓存提示永远不能代替授权。缓存键必须包含会改变可见性的每个请求维度，包括租户、用户、作用域、语言区域和分页游标。如果共享缓存无法安全表达这些维度，应使用 `private`、零 TTL 和宿主层禁止存储策略。

## 订阅使用客户端打开的响应流

现代订阅模式取代了以前的 `resources/subscribe` RPC 和旧 HTTP GET 事件端点。

客户端把 `subscriptions/listen` 作为普通 JSON-RPC 请求发送。在可流式 HTTP 上，它是一次 POST，其响应会作为 SSE 数据流保持打开。`notifications` 对象是一份允许名单，服务器不得传递未被请求的通知类型。

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

请求 ID 就是订阅 ID。在发送任何请求的事件之前，服务器先发送 `notifications/subscriptions/acknowledged`。其中的过滤器只包含服务器实际接受的子集。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

该数据流上的后续每个事件都携带相同元数据。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

这条通知只说明资源已经改变。客户端要在当前授权下，通过 `resources/read` 再次读取它，不能假定事件本身包含新文档。

多个订阅可以共享同一条 stdio 通道，订阅 ID 让客户端能够将它们分流。在 HTTP 上，关闭响应流就会取消订阅。服务器正常结束数据流时，会返回与原请求关联的最终 `resultType: "complete"` 响应。

不要把订阅流当作协议会话。后续读取仍然是完整请求，可以到达任何健康服务器实例。

```figure
t3-primitive-sort
```

## 交互实验

使用图示对项目跟踪器的五项能力进行分类：问题详情、创建问题、迭代审阅模板、项目策略和关闭问题。然后判断哪些列表可以公开缓存、哪些读取必须保持私有，以及哪些资源值得发送更新通知。

每次分类都应指出选择方。模型执行操作时使用工具；宿主读取由 URI 寻址的内容时使用资源；用户启动预先准备好的消息工作流时使用提示词。

## 实践实验

从仓库根目录运行模拟器：

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

按以下顺序检查记录：

1. 确认 `server/discover` 公布当前修订版和两类能力。
2. 确认两个列表结果都经过排序，并使用 `resultType: "complete"`。
3. 确认列表和读取结果携带有明确意图的缓存提示。
4. 将读取 URI 改为 `notes://missing`，观察 `-32602`。
5. 确认订阅确认通知出现在资源事件之前。
6. 确认事件与正常关闭都携带订阅 ID `5`。

这个 Python 模型不会建立真实 HTTP 连接，而是表示 SDK 必须放到请求范围响应流中的消息。生产环境应使用官方 SDK 处理帧与传输。

## 交付成果

`outputs/skill-primitive-splitter.md` 是一份可复用的 MCP 原语选择设计审查清单。它现在会检查确定性发现、缓存作用域、无效 URI 行为和现代订阅过滤器。

本课还交付 `assets/primitive-split.svg`，作为原语与订阅边界的静态版本，供离线学习使用。

## 验证方法

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

预期结果：主程序打印一份 JSON 记录，测试命令报告至少十二项测试通过。

## 与综合项目的联系

当综合项目服务器需要在操作旁边公开可寻址知识时，请使用这份契约。应包含一份确定性目录快照、一次已授权的资源读取、一次提示词解析、一个无效 URI 用例和一份订阅记录。

证据应表明：任何列表都不依赖连接历史，而且订阅事件绝不会自行授予对底层资源的访问权。

## 练习

1. 添加 `notes://projects/{project}/notes/{id}` 资源模板，并验证两个变量。
2. 为 `resources/list` 添加分页，同时保持确定性顺序。
3. 将一项资源改为 `cacheScope: "private"` 与 `ttlMs: 0`，添加宿主层禁止存储策略，并解释为何需要同时使用这两项控制。
4. 添加提示词列表变更订阅，并证明过滤器省略 `promptsListChanged` 时不会发送事件。
5. 同时创建两个订阅，并证明每个事件都携带正确的请求 ID。
6. 为读取处理器添加授权主体，并证明缓存项不能跨主体复用。

## 关键术语

- **资源：** MCP 服务器公开的、通过 URI 寻址的内容。
- **提示词：** MCP 服务器公开的、由用户控制的消息模板。
- **确定性列表：** 在请求输入相同的情况下，成员与顺序都保持稳定的发现结果。
- **`ttlMs`：** 以毫秒为单位的缓存新鲜度时长。
- **`cacheScope`：** 缓存结果的共享边界。
- **`subscriptions/listen`：** 一种长连接请求，其响应流会传递经过显式过滤的通知。
- **订阅 ID：** 原始监听请求的 ID，会在通知元数据中重复出现。
- **无效参数：** JSON-RPC 错误 `-32602`，用于无效或未知的资源 URI。
- **不受支持的协议版本：** JSON-RPC 错误 `-32022`，其中包含 `supported` 和 `requested` 修订版。
- **`server/discover`：** 返回受支持修订版、能力、身份和可选缓存提示的必需服务器方法。

## 延伸阅读

- [MCP 2026-07-28 资源](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 提示词](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 订阅](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 缓存](https://modelcontextprotocol.io/specification/2026-07-28/basic/utilities/caching)
