# MCP 模型输入：Sampling 迁移与无状态 MRTR

> MCP 2026-07-28 不再建议新设计采用 Sampling，并移除了服务器到客户端的请求通道。如果现有工作流仍需要使用客户端的模型，服务器会返回 `input_required` 结果，客户端再携带模型输出重试原始请求。这样一来，推理循环在协议层变得显式、有界且无状态。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 07 课（MCP 服务器）、第 13 阶段 · 第 10 课（资源与提示词）
**Time:** 约 75 分钟

## 学习目标

- 解释 MCP 2026-07-28 为何弃用 Sampling，并为新服务器选择直接集成模型的默认方案。
- 实现一个通过多轮往返请求（MRTR）承载 `sampling/createMessage` 的兼容工作流。
- 在每个请求的 `_meta` 对象中放入协议修订版与客户端能力。
- 返回 `resultType: "input_required"`，并使用全新的 JSON-RPC ID 重试原始方法。
- 对 `requestState` 提供完整性保护，并将它与主体、方法、参数及过期时间绑定。
- 通过能力检查、审批、响应验证与轮次上限，约束模型辅助循环。

## 先做架构决策，再谈协议

以 `summarize_repo` 这样的工具为例，它需要完成两类工作：

1. 确定性工作：列出文件、读取允许访问的文件、验证路径并汇集内容。
2. 模型工作：选择具有代表性的文件，并综合生成摘要。

现在有两种合理的架构可供选择。

### 新服务器：直接集成模型提供方

这是当前的默认方案。服务器负责模型选择、凭证、预算、重试与可观测性，并向 MCP 客户端返回一个普通的 `tools/call` 结果。

如果服务器本身已经是托管服务，或者可预测的模型行为比使用宿主模型更重要，就应选择这种方案。

### 现有 Sampling 工作流：迁移到 MRTR

Sampling 在弃用过渡期内仍然存在。面向 2026-07-28 的服务器不能再向客户端反向发送实时 `sampling/createMessage` 请求，而是把该请求嵌入 `InputRequiredResult`。

只有当使用客户端的模型与凭证确实属于产品需求时，才应选择这条兼容路径。还要记录移除计划，因为新的实现不应继续采用已弃用的 Sampling。

## 无状态契约

2026 年 7 月版协议没有 `initialize` 交换，没有 `notifications/initialized`，也没有 `Mcp-Session-Id`。每个请求都携带过去放在握手中的信息：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

服务器会在每个请求上验证修订版。版本缺失或不是字符串时，属于无效参数，返回 `-32602`。版本字符串不受支持时，返回 `-32022` 以及精确数据 `{"supported":["2026-07-28"],"requested":"<client version>"}`。缺少 Sampling 能力时，返回 `-32021`，并把 `data.requiredCapabilities` 设为 `{"sampling":{}}`。

没有 JSON-RPC `id` 的信封属于通知。接收方可以处理它，但不会发出成功响应或错误响应。对于已接受的通知，可流式 HTTP 适配器返回不带响应体的 `202 Accepted`。

服务器还要实现 `server/discover`，并使用准确的 `supportedVersions` 键、能力、`ttlMs` 和 `cacheScope`，让客户端能在调用工具前了解并缓存服务器契约。因为发现结果公布了 `tools` 能力，服务器还必须实现 `tools/list`。其中顺序确定的 `summarize_repo` 描述符应包含有效的对象型 `inputSchema`、`resultType: "complete"`、服务器身份元数据以及公共缓存提示。

每个成功的现代结果都有一个判别字段：

- `resultType: "complete"` 表示操作已经完成。
- `resultType: "input_required"` 表示客户端必须满足嵌入的请求，然后重试。
- 扩展可以定义其他结果类型。第 13 课中的 Tasks 扩展增加了 `"task"`。

## 一轮 MRTR

服务器不能在处理请求时调用客户端，因此会改为返回以下结果：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

客户端确认自己支持 Sampling，应用审批策略与模型策略，然后获取模型响应。接着，它使用不同的 JSON-RPC ID 发送一个新请求：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

这次重试并不是协议会话的延续，而是一个新请求：它重复原始方法与参数，只添加当前轮次的 `inputResponses`，并逐字节回显 `requestState`。

只有 `tools/call`、`prompts/get` 和 `resources/read` 允许使用 MRTR。服务器不得从无关方法返回 `input_required`。

## 多轮状态

本课需要调用模型两次：

1. `pick_files` 返回一个 JSON 数组。
2. `summary` 返回最终文本。

每次重试只携带当前轮次的响应，因此服务器会把阶段和已验证的中间数据放入下一份 `requestState`。

必须把该值视为由攻击者控制。只签名一个原始阶段名称远远不够，应将状态绑定到：

- 经过身份认证的主体，而不是自行声明的 `clientInfo`；
- 发起请求的方法；
- 原始参数的摘要；
- 较短的过期时间；
- 当前阶段与已验证的中间值。

不需要保密性时使用 HMAC；客户端不应读取状态时，则使用认证加密。签名错误、值已过期、主体改变或参数改变时，都应拒绝请求并返回 `-32602`。

客户端不得解析或修改 `requestState`。它唯一的职责是在重试时回显完全相同的字符串。

## 模型偏好只是提示

`costPriority`、`speedPriority` 和 `intelligencePriority` 是彼此独立的偏好。它们不是概率分布，无需相加等于一。客户端拥有模型策略，因此可以忽略这些偏好。

如果你仍在维护旧版 Sampling 流程，请将 `includeContext` 保持为 `"none"`。其他上下文模式会增加泄漏风险，而且它们本身也已弃用。请在请求中显式传递最少量的上下文。

## 安全不变量

对于嵌入式 Sampling 请求，客户端就是信任边界。

- 当策略要求审批时，向用户展示服务器要求模型执行的任务。
- 限制 MRTR 轮数，否则恶意服务器可以制造模型支出循环。
- 在把任何 Sampling 响应用作文件名、URL 或工具输入之前，先对其进行验证。
- 限制每轮的字节数与 token 数。
- 拒绝当前客户端能力中未声明的输入请求。
- 不要让模型输出参与授权决策。
- 记录原始方法和输入请求键，但不要记录敏感的提示内容。

`clientInfo` 与 `serverInfo` 是显示及诊断元数据，绝不能把其中任何一个用作经过身份认证的标识。

```figure
t3-sampling-flip
```

## 构建它

`code/main.py` 不依赖第三方包，实现了完整的两轮流程：

- `server/discover` 返回 `supportedVersions`、公布工具支持并返回缓存提示。
- `tools/list` 返回一个顺序确定且可缓存的 `summarize_repo` 描述符，其中包含对象型输入模式。
- `tools/call` 验证逐请求元数据。
- 第一个结果嵌入用于选择文件的 `sampling/createMessage`。
- 第一次重试验证模型结果，并嵌入第二个请求。
- 受 HMAC 保护的 `requestState` 在各独立请求之间携带阶段。
- 最终结果使用 `resultType: "complete"`。

模拟宿主模型让示例保持确定性。连接真实宿主时，只需替换 `fake_host_model`。服务器端状态机应继续保持确定性和可测试性。

## 使用它

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

预期检查点：

- 发现操作返回带 `ttlMs` 与 `cacheScope` 的完整结果。
- 工具发现返回相同的已排序描述符，其中带有 `resultType`、服务器身份和缓存提示。
- 缺失能力与版本不受支持时，分别使用准确的 `-32021` 和 `-32022` 错误数据。
- 不带 ID 的通知不会产生 JSON-RPC 响应。
- 请求 ID 为 `[1, 2, 3]`，证明每一轮 MRTR 都彼此独立。
- 前两个结果均为 `input_required`。
- 最终结果为 `complete`，并包含所选文件和摘要。
- 在重试时更改原始参数，会导致请求状态检查失败。

## 交付成果

`outputs/skill-sampling-loop-designer.md` 现在是一份迁移规划器。它会先判断是否应当移除 Sampling，改为直接集成模型。如果确实需要兼容，它会生成 MRTR 轮次、状态绑定、能力门、预算、验证措施与移除计划。

## 练习

1. 把文件选择响应改成无效 JSON。确认服务器返回 `-32602`，而不是直接信任模型输出。
2. 在第一次调用与重试之间更改 `audience`。解释密封状态为何能够阻止跨请求复用。
3. 添加第三轮，让宿主评析摘要。把先前摘要放入已签名状态，并把整个流程限制为三轮。
4. 用服务器自有的模型适配器替换模拟宿主回调，从而移除 Sampling。列出哪些审批、计费与可观测性职责会转移到服务器。
5. 使用比截止时间晚一秒的状态值，添加一项过期测试。

## 关键术语

| 术语 | 在 2026-07-28 中的含义 |
|------|------------------------|
| Sampling | 已弃用的功能，用于向客户端模型请求补全结果 |
| MRTR | 请求处理中需要客户端输入时使用的无状态重试模式 |
| `InputRequiredResult` | 带有 `resultType: "input_required"` 的结果 |
| `inputRequests` | 由服务器分配键名的映射，其中嵌入信息征询、Sampling 或根目录请求 |
| `inputResponses` | 当前轮次的客户端结果，键名与 `inputRequests` 对应 |
| `requestState` | 客户端原样回显、服务器负责验证的不透明服务器状态 |
| `resultType` | 现代 MCP 结果必需的判别字段 |
| 直接模型集成 | 对需要模型推理的新服务器所推荐的替代方案 |
| 能力门 | 阻止发送客户端未声明支持的嵌入请求的规则 |
| 循环预算 | 操作所允许的最大轮数、token 数、字节数、时间和支出 |

## 旧版兼容性

固定使用 2025-11-25 的客户端仍可通过实时连接，采用较早的、由服务器发起的 `sampling/createMessage` 流程。只应在特定版本的适配器中保留该行为，不能让有会话状态的路径成为 2026-07-28 服务器的架构。

官方 SDK 可以为旧版对等端转换现代 `input_required` 处理器。这个垫片是一条兼容性边界，并不意味着可以添加新的会话依赖逻辑。

## 延伸阅读

- [MCP 2026-07-28 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling 弃用说明](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
