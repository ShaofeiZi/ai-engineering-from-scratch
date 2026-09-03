# 显式作用域与无状态信息征询

> Roots 在 MCP 2026-07-28 中已被弃用，而且从来都不是安全沙箱。应把作用域放进可见的工具参数或资源 URI，在服务器上执行授权，并在工具确实需要用户输入时使用 MRTR。用户看到决策，模型看到句柄，而任意服务器实例都能处理重试。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 07 课（MCP 服务器）、第 13 阶段 · 第 11 课（无状态 MRTR）
**Time:** 约 60 分钟

## 学习目标

- 用显式工作区参数、资源 URI 或服务器配置取代已弃用的 Roots。
- 将作用域提示与授权、路径包含性检查及操作系统沙箱区分开。
- 通过 MRTR 传递表单模式的 `elicitation/create`，并返回 `input_required` 结果。
- 在逐请求客户端能力中公布信息征询支持，并拒绝不受支持的模式。
- 将 `accept`、`decline` 和 `cancel` 验证为三种不同结果。
- 将破坏性确认绑定到经过身份认证的主体、原始参数、候选集合和过期时间。

## 两个看似相同的问题

一个笔记工具收到这样的请求：“删除旧的 TPS 报告。”

服务器必须回答两个不同的问题。

1. 这项操作可以触及哪个工作区？
2. 三篇匹配的笔记中，用户指的是哪一篇？

第一个问题涉及作用域与授权，第二个问题涉及交互式消歧。把二者混为一谈会导致危险的设计，例如把客户端提供的文件夹当成调用方有权删除其中一切内容的证明。

## Roots 是一个迁移接口面

较早的 MCP 修订版允许客户端公布 Roots，并在列表变化时通知服务器。Roots 只是信息性指引。它们不会限制服务器进程能够读取的内容，不会授权调用方，也不会创建操作系统沙箱。

MCP 2026-07-28 不再建议新设计使用 `roots/list` 和 `notifications/roots/list_changed`。应优先选择以下显式替代方案之一：

- 当作用域随每次调用而变化时，使用 `workspaceUri` 或 `directory` 工具参数。
- 当操作本来就以某个资源为目标时，使用资源 URI。
- 当一次部署只拥有一个固定工作区时，使用服务器配置。
- 当代码必须在技术上无法越界时，使用进程沙箱或受限文件系统。

如果现有的 2026-07-28 集成在弃用过渡期内仍需要 `roots/list`，服务器应将它嵌入 MRTR `inputRequests`，不得发送实时反向请求。这只是迁移适配器；新的处理器应改为接受显式作用域。

模型能够看到并复述显式句柄。隐藏在传输会话中的作用域则更难检查、重放、审计和路由。

### 三层规则

显式 URI 仍然不能自证授权。必须落实以下三层控制：

1. **授权：** 这个经过身份认证的主体是否可以使用该工作区？
2. **包含性：** 规范化后的目标 URI 是否仍位于获授权的工作区边界内？
3. **沙箱：** 即使服务器遭到入侵，操作系统能否阻止它越界？

可运行服务器会维护一份已授权工作区 URI 的允许名单，规范化百分号编码路径，检查真实的路径组件边界，并在删除前立即再次检查包含关系。

简单的字符串前缀检查是错误的：

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

这两个恶意路径都以具有迷惑性的字符串开头。应先规范化，再比较路径组件。生产环境中的文件系统服务器还必须防范符号链接竞态以及特定平台的路径语义。

## 信息征询仍然存在，但传递方式已经改变

信息征询是当前客户端在 `tools/call`、`prompts/get` 或 `resources/read` 执行期间收集用户输入的功能。方法名称仍是 `elicitation/create`，改变的是线路流向。

2026-07-28 服务器不会发送反向 JSON-RPC 请求，而是返回一个 `InputRequiredResult`：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

宿主负责渲染表单。用户可以接受、明确拒绝或关闭表单。随后，客户端使用新的 ID 重试原始 `tools/call`：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

两次调用之间不存在协议会话。服务器验证回显的状态，按照预期模式验证响应，检查所选笔记是否位于已签名的候选集合中，重新授权工作区，再次检查包含关系，然后才执行删除。

## 能力协商逐请求进行

支持表单模式信息征询的客户端应声明：

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

为了兼容性，空的信息征询能力 `"elicitation": {}` 仍然等同于仅支持表单。显式声明 `"elicitation": {"form": {}}` 也支持表单模式；仅声明 URL 的 `"elicitation": {"url": {}}` 则不支持。即使先前请求曾公布某种模式，服务器也不得嵌入当前请求能力中不存在的模式。

每个请求还会携带 `io.modelcontextprotocol/protocolVersion`。版本缺失或不是字符串时返回 `-32602`。版本字符串不受支持时返回 `-32022`，并附带准确的 `supported` 与 `requested` 数据。缺少信息征询支持或仅支持 URL 时返回 `-32021`，并将 `data.requiredCapabilities` 设为 `{"elicitation":{"form":{}}}`。

没有 JSON-RPC `id` 的信封属于通知。可以处理它，但不要发出 JSON-RPC 成功或错误响应。在可流式 HTTP 上，已接受的通知会收到不带正文的 `202 Accepted`。

应包含 `clientInfo` 以便诊断，但它由客户端自行声明，不能用于识别需要授权的用户。

服务器实现 `server/discover`，并返回 `supportedVersions`、能力、`ttlMs` 与 `cacheScope`，同时使用 `resultType: "complete"`。此现代设计不会公布 Roots。因为服务器公布了工具能力，所以还要实现必需的 `tools/list`。该结果返回顺序确定的 `notes_delete` 描述符、有效的对象型 `inputSchema`、服务器身份元数据和公共缓存提示。

## 表单模式

表单模式使用一种为可用对话框设计的受限 JSON Schema。根节点是对象，其属性为扁平的原始类型字段或受支持的枚举数组。深层嵌套对象和通用文档模式不适合放进确认对话框。

表单模式适用于：

- 从多个候选项中选择一个；
- 确认破坏性操作；
- 收集非敏感偏好；
- 收集少量必须由用户而非模型决定的值。

不要使用表单模式收集密码、API 密钥、访问令牌或支付凭证。这些秘密会经过 MCP 客户端，并可能进入日志或模型上下文。

服务器必须再次验证返回内容。客户端侧表单验证可以改善用户体验，但不会建立信任。

## URL 模式

URL 模式会发送一个安全的 Web URL，用于带外交互：

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

当敏感信息必须直接进入服务器控制的 Web 流程时，例如第三方授权，应使用这种模式。客户端会显示完整目标地址，并在打开前征得同意；不得预取该 URL。

`accept` 响应表示用户同意打开 URL，并不能证明外部流程已经完成。重试时，服务器检查自己的状态，然后完成操作，或者再返回一个 `input_required` 结果。

URL 信息征询不能代替 MCP 客户端与 MCP 服务器之间的授权。它用于完成 MCP 服务器需要代表用户执行的外部交互。服务器必须把浏览器用户绑定到发起 MCP 操作的同一个已认证主体。

## 响应分支

应把各个动作视为不同的产品决策，而不是彼此的别名：

| 动作 | 含义 | 安全的服务器行为 |
|--------|---------|----------------------|
| `accept` | 用户提交了交互 | 验证内容并继续 |
| `decline` | 用户明确拒绝 | 返回完整、非错误的拒绝结果 |
| `cancel` | 用户关闭交互或无法完成 | 安全停止，并允许以后重试 |

绝不能把缺少内容解释为同意，也不能把拒绝变成不断重复提示的循环。

## 保护破坏性 MRTR 状态

候选列表不能只存在于提示中，也不能放在未签名的 Base64 值里。客户端能够控制自己发回的一切内容。

本课签名的状态载荷包含：

- 经过身份认证的主体；
- 发起请求的方法；
- `workspaceUri` 和 `title` 的摘要；
- 表单中展示的允许笔记 ID；
- 操作阶段；
- 较短的过期时间。

在变更数据之前，服务器还要检查实时笔记记录。这样可以发现删除竞态，以及目标在表单展示后被移出工作区的情况。

对于一次性的财务操作或不可逆操作，仅靠 HMAC 无法阻止有效状态在过期前被重放。应在所有处理器实例共享的重放存储中保存 nonce，并确保只消费一次。本课注入了一个有界、按 TTL 清理的存储，并在执行内存删除期间持有其原子认领。生产数据库应在同一事务或等价的条件写入边界内完成 nonce 认领与数据变更。

应先验证交互，再认领 nonce。格式错误的响应或 `cancel` 不会执行任何变更，而且状态在过期前仍可重试。显式 `decline` 是终止性结果，因此本课会消费 nonce，但不会删除任何内容。

```figure
t3-roots-boundary
```

## 构建它

`code/main.py` 演示了一个现代 `notes_delete` 工具：

- `tools/list` 返回顺序确定且可缓存的描述符，其中包含必需的工作区和标题模式。
- 作用域是一个显式的 `workspaceUri` 参数。
- 服务器配置为本课主体授权了该工作区。
- URI 规范化会拒绝前缀混淆与编码后的路径穿越。
- 每次破坏性删除都需要表单模式信息征询。
- 信息征询通过 `resultType: "input_required"` 传递。
- 已签名的 `requestState` 绑定准确的候选列表与原始参数。
- 注入的重放存储会跨服务器实例拒绝相同的已接受或已拒绝状态。
- 重试使用新的请求 ID，并返回 `resultType: "complete"`。

数据存储位于内存中，便于检查协议行为。改用数据库后，安全规则仍然相同。

## 使用它

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

预期检查点：

- 发现操作会公布工具能力，但不公布 Roots。
- 工具发现返回 `notes_delete`，并带有 `resultType`、服务器身份及缓存提示。
- 请求 ID `1` 在 `inputRequests.delete_choice` 中返回表单。
- 请求 ID `2` 回显已签名状态并完成删除。
- 前缀路径与编码后的路径穿越都会因包含性检查而失败。
- 更改标题后，无法复用原始确认状态。
- 用户拒绝时，笔记保持不变。
- 两个共享笔记状态与重放状态的服务器对象，不能同时执行同一次确认。
- 空声明和显式表单声明都有效，而仅支持 URL 时，会返回准确的 `-32021` 表单能力要求。
- 不受支持的版本使用准确的 `-32022` 数据形态。
- 不带 ID 的通知不会产生 JSON-RPC 响应。

## 交付成果

`outputs/skill-elicitation-form-designer.md` 用于设计显式作用域、授权检查、MRTR 表单、响应分支和状态绑定。它拒绝把已弃用的 Roots 当作沙箱，也拒绝通过表单模式收集秘密。

## 练习

1. 用 SQLite 替换内存重放存储。在同一事务中认领 nonce 并删除笔记，然后证明两个进程不可能同时提交。
2. 添加 `url` 能力协商与带外设置流程。不要把第三方凭证放入 `inputResponses`。
3. 用临时 SQLite 数据库替换内存笔记映射。在变更事务内再次检查授权与包含关系。
4. 为真实文件系统实现添加符号链接策略。解释为什么仅检查 URI 的词法包含关系无法阻止符号链接逃逸。
5. 设计一个 2025-11-25 适配器，把现代 MRTR 处理器输出映射为旧版服务器发起的信息征询，并与当前处理器隔离。

## 关键术语

| 术语 | 在 2026-07-28 中的含义 |
|------|------------------------|
| Roots | 已弃用的信息性工作区提示，不是授权或沙箱 |
| 显式作用域 | 请求参数中可见的工作区、目录或资源句柄 |
| 包含性 | 经过规范化的路径组件检查，用于确保目标位于边界内 |
| 信息征询 | 在 MCP 操作期间获取用户输入的客户端功能 |
| 表单模式 | 使用受限扁平模式的带内结构化用户输入 |
| URL 模式 | 用于敏感或外部工作流的带外交互 |
| MRTR | 返回无状态的输入必需结果，随后发起全新重试 |
| `requestState` | 由客户端原样回显、服务器执行完整性检查的不透明状态 |
| 拒绝 | 用户明确拒绝 |
| 取消 | 在未批准的情况下关闭或未完成交互 |

## 旧版兼容性

对于固定使用 2025-11-25 的对等端，`roots/list`、`notifications/roots/list_changed` 和由服务器实时发起的 `elicitation/create` 可能仍然存在。应明确把该适配器标记为旧版。不要允许旧版 Roots 列表绕过服务器授权，也不要把协议会话假设带入现代处理器。

## 延伸阅读

- [MCP 2026-07-28 信息征询](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots 弃用说明](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
