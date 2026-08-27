# 构建 MCP 服务器：无状态 Python 与 TypeScript

> 现代 MCP 服务器不会记住握手。它会在每个请求上验证元数据、运行一个处理器，再返回一个类型化结果。

**Type:** 构建
**Languages:** Python, TypeScript
**Prerequisites:** 第 13 阶段 · 第 06 课（MCP 基础）
**Time:** 约 85 分钟

## 学习目标

- 实现必需的 `server/discover`，用于 MCP `2026-07-28`。
- 在每个请求上验证协议版本与客户端能力。
- 使用确定性列表顺序公开工具、资源与提示词。
- 在正确的结果上返回 `resultType`、服务器身份与缓存提示。
- 通过以换行符分隔的 stdio，在 Python 与 TypeScript 中提供同一份无状态契约。

## 问题

在第一条消息之后保存客户端能力的服务器，编写起来很容易，运维起来却很困难。同一个进程可能连续服务多个客户端，远程请求也可能落到另一个工作器。过期的能力声明会让行为跨授权边界泄漏。

MCP `2026-07-28` 通过让每个请求能够自我描述，解决了这个问题的协议部分。应用仍然可以保存持久笔记、任务或显式状态句柄；不能保存的是会改变后续请求解码方式的隐藏协议状态。

本课会分别用 Python 和 TypeScript 构建笔记服务器。两个版本的协议核心都只使用各自标准库，公开相同方法，并强制执行相同线路契约。

## 概念

### 现代分派循环

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

stdio 仍有三条重要规则：

- 只向 stdout 写入 JSON-RPC 消息，诊断信息写入 stderr。
- 使用换行符分隔消息，并在每次响应后立即刷新。
- stdin 到达 EOF 时立即退出。

进程生命周期属于传输层，不是现代 MCP 会话。

### 请求验证

每个请求都必须包含：

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

前两个字段是必需的，建议提供 `clientInfo`。若身份字段存在，应验证其形态，但不要把它当作身份认证。

版本不受支持时，返回代码 `-32022`，并包含 `requested` 和 `supported`。缺少请求元数据属于无效参数，代码为 `-32602`。绝不能从先前调用中补齐缺失字段。

### 必需的发现方法

现代服务器必须实现 `server/discover`。完整的发现结果包含受支持的现代版本、能力、可选说明、缓存提示，以及结果 `_meta` 中的服务器身份：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

发现不会解锁服务器。客户端可以在未调用发现的情况下直接调用 `tools/list`，因为 `tools/list` 已经携带同样的请求元数据。

### 工具

`tools/list` 返回按确定顺序排列的工具描述符。稳定顺序可以改善响应缓存，并让模型上下文保持稳定。结果还必须包含 `ttlMs` 与 `cacheScope`。

`tools/call` 返回内容块和 `isError`。如果协议信封或方法参数无效，使用 JSON-RPC 错误；如果有效工具调用已经运行，但工具自身失败，则返回 `isError: true`。

工具注解仍然只是提示，不是强制机制：

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

宿主应使用它们进行确认与展示，但服务器仍必须强制执行真实授权。

### 资源

`resources/list` 返回稳定的 URI 描述符，`resources/read` 返回类型化内容。二者在 `2026-07-28` 中都可缓存，因此都包含 `ttlMs` 和 `cacheScope`。

用户专属笔记数据应使用 `cacheScope: "private"`。共享缓存不得跨授权上下文复用私有响应。

现代变更传递不使用 `resources/subscribe`。客户端会打开 `subscriptions/listen`，请求 `resourceSubscriptions` 或列表变更类别。第 10 课会构建该流程。

### 提示词

`prompts/list` 可缓存且顺序确定。`prompts/get` 使用参数渲染具名提示词。渲染后的提示词结果是完整结果，但它并不属于必须提供缓存提示的可缓存列表或读取结果。

### 每个成功结果都有类型

示例使用同一个包装函数处理所有成功结果：

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

列表、读取与发现处理器会额外添加 `ttlMs` 和 `cacheScope`。集中使用这层包装，可以防止某个处理器悄然遗漏现代结果字段。

### 不允许服务器主动发起请求

现代服务器可以发送与某个客户端请求有关的通知，也可以在客户端打开的 `subscriptions/listen` 数据流上发送通知，但不得自行发起 JSON-RPC 请求。

如果处理器需要采样、信息征询或根目录输入，就返回 `input_required` 结果。客户端满足其中嵌入的输入请求，再使用新的请求 ID 重试原方法。第 11 课会介绍这种多轮往返请求模式。

### 显式旧版兼容

兼容新旧两个时代的服务器，也可以在明确隔离的旧版分支上实现 `2025-11-25` 握手。必需的现代 `_meta` 字段存在时选择现代行为，收到 `initialize` 时选择旧版行为。

不要把 `2026-07-28` 请求送入旧版握手路径，也不要在旧版初始化结果上添加现代 `resultType` 字段。本课代码刻意只实现现代协议，以便清晰呈现其不变量。

```figure
t3-dispatch-loop
```

## 投入使用

运行 Python 服务器的有限演示与测试：

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

使用 TypeScript 运行器执行 TypeScript 移植版：

```bash
npx tsx main.ts --demo
```

演示会发送 `server/discover`，列出各类原语，调用工具，并展示不受支持的版本错误。每个现代请求都会重复元数据，每个成功结果都包含服务器身份。

## 交付成果

本课交付 `outputs/skill-mcp-server-scaffolder.md`。它会生成一份现代服务器方案，其中包含发现契约、逐请求验证、确定且可缓存的列表，以及可选的隔离旧版适配器。

## 练习

1. 从一个请求中移除能力字段，证明服务器不会复用前一个请求的声明。
2. 反转 `TOOLS`、`PROMPTS` 与笔记的插入顺序，确认所有列表结果仍然稳定。
3. 添加一个破坏性的 `notes_delete` 工具，并要求执行器内部进行授权检查。`destructiveHint` 只保留为用户体验提示。
4. 添加 `resources/templates/list`，包含 `ttlMs`、`cacheScope` 与确定性排序。
5. 为 `2025-11-25` 构建独立旧版适配器，并添加测试证明现代请求绝不会进入它。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 无状态服务器 | 根据每个请求自身元数据处理请求，不保存协议会话记忆 |
| `server/discover` | 公布版本与能力的必需现代方法 |
| 完整结果 | 带 `resultType: "complete"` 的现代成功结果 |
| 可缓存结果 | 包含 `ttlMs` 与 `cacheScope` 的发现、列表或资源读取结果 |
| 确定性列表 | 相同逻辑注册表始终产生相同项目顺序 |
| 服务器身份 | 建议提供的 `io.modelcontextprotocol/serverInfo`，位于结果 `_meta` 中 |
| 工具错误 | 有效工具调用返回带 `isError: true` 的内容 |
| 协议错误 | 通过 `error` 返回的无效 JSON-RPC 或 MCP 请求 |

## 延伸阅读

- [MCP 2026-07-28 规范](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 工具](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP 资源](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 提示词](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
