# 无状态协议上的 MCP Apps

> 交互式结果本质上仍是一次 MCP 工具与资源交换。2026-07-28 核心协议让这次交换自包含，Apps 扩展则在其上增加了沙箱化的浏览器界面。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 07 课（MCP 服务器）、第 13 阶段 · 第 10 课（资源）
**Time:** 约 75 分钟

## 学习目标

- 通过 `server/discover` 和逐请求扩展能力声明 MCP Apps。
- 在工具被调用前，就在工具上声明一个 `ui://` 资源。
- 在 2026-07-28 无状态线协议上返回完整的工具结果和资源结果。
- 将 Apps 的 `ui/initialize` 桥接消息与已移除的 MCP 核心握手区分开。
- 应用来源验证、沙箱、CSP 与最小权限原则。

## 问题

文本结果可以描述时间线，却无法让用户直接筛选、检查或操作这条时间线。

MCP Apps 通过一个可选扩展解决呈现问题。工具定义指向一个 `ui://` 资源。主机可以在工具运行前获取并审查该资源，在沙箱 iframe 中渲染它，并通过 JSON-RPC 桥接来中介所有应用操作。

核心协议已于 2026-07-28 发生变化。不要再把 App 包装进旧的连接生命周期：

- 核心协议中不再有 `initialize` 请求或 `notifications/initialized` 通知。
- 不再有 `Mcp-Session-Id` 请求头。
- 每个请求都在 `params._meta` 中携带协议版本和客户端能力。
- 服务器实现 `server/discover`，让客户端检查版本、核心能力和扩展。
- 每个成功结果都带有 `resultType` 判别字段。
- Streamable HTTP 每次请求使用一个 POST；现代 GET 和 DELETE 入口返回 405。

Apps 桥接仍有一个名为 `ui/initialize` 的方法。它属于 iframe 的 postMessage 方言，并不会重新创建核心 MCP 会话。

## 核心概念

### 两种协议，一项功能

应明确区分各层：

1. MCP 核心协议承载 `server/discover`、`tools/list`、`tools/call`、`resources/list` 和 `resources/read`。
2. MCP Apps 扩展声明 UI，并定义 iframe 到主机的桥接。
3. 浏览器沙箱规则限制 UI 能访问的对象。

扩展标识符为 `io.modelcontextprotocol/ui`。通信双方都要选择启用。客户端在每个请求的能力对象中发送扩展支持：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

建议提供 `clientInfo` 以便诊断。它是客户端自行声明的数据，而不是授权身份。

### 渲染前先发现

服务器的发现结果声明该扩展：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

服务器必须支持发现。客户端不必在每项操作前都调用发现，因为每项操作都会携带自己的能力信息。

### 在工具定义上声明 UI

现代 Apps 合约通过 `tools/list` 把 UI 绑定到工具：

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

这是有意放在调用前的元数据。主机可以在某个结果请求显示 HTML 之前预加载、缓存并进行安全审查。兼容代码可能接受旧的扁平元数据键，但新服务器应该输出嵌套的 `_meta.ui.resourceUri` 形式。

当前核心协议中的 `tools/list` 可以缓存。应提供确定性排序、`ttlMs` 和 `cacheScope`。当用户或令牌会改变可见工具时，应使用 `private`。

### 返回数据，再由主机绑定视图

工具调用返回普通内容和结构化数据：

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

主机已经知道哪个视图属于该工具。不要为了重复 URI 而发明新的内容块。

### 将 App 作为资源提供

服务器在发现结果中声明 `resources`，因此也必须实现强制的 `resources/list` 操作。其确定性列表项包含规范 URI、稳定名称、描述和 MIME 类型。列表结果和确定性的工具列表一样，也包含 `resultType`、服务器身份元数据、`ttlMs` 与 `cacheScope`。

主机发送 `resources/read`。使用 Streamable HTTP 时，请求包含：

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

请求头的值必须与 JSON-RPC 正文一致。不一致属于协议错误 `-32020`。

结果包含 HTML 资源和缓存提示：

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### 把 UI 资源当作可执行内容缓存

App 资源不能与普通文本混为一谈。其缓存项能够执行桥接代码、渲染工具数据，并请求由主机中介的操作。缓存键应包含规范的 `ui://` URI、已准入的服务器身份与版本、资源内容摘要；当 `cacheScope` 为 private 时，还要包含授权上下文。绝不能跨主体复用私有 App 资源，因为即使 URI 相同，HTML 或其策略元数据也可能不同。

当 `ttlMs` 到期、工具的 `_meta.ui.resourceUri` 绑定变化、服务器版本或已准入的描述符固定值变化，或已确认的资源变更订阅点名该 URI 时，都应使缓存项失效。重新挂载前，应重新获取并重新审查 CSP 与权限。不能仅仅因为新资源版本尚未加载，就让陈旧 iframe 保留更宽泛的权限。

### 先拒绝线协议歧义，再判断功能策略

验证有明确的顺序。首先验证 JSON-RPC 结构，并要求协议元数据是字符串、客户端能力映射是对象；然后比较路由请求头与正文；最后才判断匹配后的协议版本是否受支持。这个顺序可防止代理与服务器把同一数据解释成不同请求。

| 条件 | HTTP | JSON-RPC 错误 |
|-----------|------|----------------|
| 请求头与正文中的版本、方法或名称不一致 | 400 | `-32020` |
| 请求头与正文一致，但版本不受支持 | 400 | `-32022`，且 `data` 必须精确为 `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` 缺少 Apps 扩展能力 | 400 | `-32021`，且带有 `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| 方法未知 | 404 | `-32601` |

JSON-RPC 通知没有 `id`，因此服务器绝不会为它发出 JSON-RPC 响应。已接受的 HTTP 通知返回 202 和空响应体。错误可以改变 HTTP 状态码，但仍不能凭空为通知创建 JSON-RPC 错误正文。

### 沙箱是边界，不是信任结论

主机控制 iframe。App 无法直接读取主机 cookie、本地存储或页面 DOM；所有特权工作都必须经过桥接。

使用以下默认策略：

- 先将所有 CSP 域列表留空，再只添加 App 必需的来源。fetch、XHR 和 WebSocket 使用 `connectDomains`；脚本、样式、图片和字体使用 `resourceDomains`。
- 条件允许时，将代码和数据打包在资源内部。
- 除非可见功能确实需要，否则不要请求相机、麦克风或定位权限。
- 将 `postMessage` 固定到对端的精确来源，并拒绝其他所有来源的事件。
- 把工具参数、工具结果、资源文本和桥接消息都视为不可信输入。
- 用户同意必须由主机管理。iframe 不能自行批准有后果的操作。

不要把教程里的固定 `sandbox` 属性复制给所有主机。主机必须依据 App 的来源模型及自身隔离设计来选择标志。

获准域名依旧可能成为数据外泄路径。`connectDomains: ["https://api.example.com"]` 意味着 App 内执行的任何脚本都能把被允许的数据发送到那里。精确匹配来源可以防止目标混淆，却无法判断负载是否恰当。默认保持连接访问为空，不要把 bearer token 放进 iframe；条件允许时，由主机代理范围狭窄的操作，限制请求与响应大小，并审计每个出站请求由哪个用户操作触发。应将 `resourceDomains` 与 `connectDomains` 分开处理；允许加载字体或脚本，不应同时授予任意数据上传能力。

### Apps 桥接有自己的生命周期

Apps 桥接是运行在 `postMessage` 上的一种 JSON-RPC 方言。它可以交换 `ui/initialize` 与 `ui/*` 通知，也可以代理看似核心协议的方法，例如 `tools/call`。

View 发送 `ui/initialize`，其中带有 `appInfo` 和一个 `appCapabilities` 对象。主机返回自己的能力和主机上下文。只有收到该响应后，View 才发送 `ui/notifications/initialized`。主机必须等到这个 Apps 通知后，才能向 View 发送消息。

这个本地握手只会在一个 iframe 和一个主机 frame 之间建立桥接，不会协商 MCP 协议版本、创建服务器状态或签发传输会话。注意精确的前缀差异：核心 `notifications/initialized` 已被移除，而 Apps 的 `ui/notifications/initialized` 仍然存在。由桥接工具调用生成的核心请求，是一个全新的自包含请求，拥有新的 JSON-RPC ID 和完整请求元数据。

### 主机上下文、操作与撤销

桥接初始化后，主机仍然是权限主体。View 只有通过主机已声明的能力，才能请求工具操作、导航、剪贴板使用或其他特权效果。主机会验证带类型的请求、当前用户、目标和参数，应用审批策略，并且可以拒绝。按钮点击与有效的桥接消息只表达意图，都不会授予权限。

应把主题、尺寸和无障碍能力视为可变的主机上下文，而非一次性的渲染输入：

- 应用主机提供的颜色和排版 token，并在主题或对比度偏好改变时响应。
- 允许 View 报告期望尺寸，但由主机限制并应用 iframe 尺寸，防止内容逃逸布局或制造欺骗性覆盖层。
- 在 iframe 内维持键盘顺序、可见焦点、无障碍名称、屏幕阅读器状态、足够对比度、缩放与减少动效行为。
- 在调整尺寸和重新渲染后，重新测试主机控件与 View 控件之间的焦点转移。

App 打开期间，能力可能因用户切换账户、策略变化、服务器被隔离或主机收紧同意范围而撤销。应在操作发生时检查能力和授权，而不只是在 `ui/initialize` 时检查。发生撤销后，拒绝待处理的特权调用，停止不再符合策略的网络活动，清除已渲染的敏感状态；如果 UI 资源本身不再获准，则重新挂载或回退到文本。View 必须把拒绝当成正常结果处理，不能持续重试直到主机让步。

### 回退也是合约的一部分

支持 Apps 的服务器仍可服务未声明 UI 扩展的主机：

- 返回同一个不带 `_meta.ui` 的工具，并将它放在 `tools/list` 的结果中。
- 为 `tools/call` 保留有用的文本结果。
- 对 UI 的 `resources/read` 返回缺少能力错误。
- 判断工具是否完成时，绝不能假设 iframe 一定存在。

```figure
t3-ui-sandbox
```

## 动手构建

`code/main.py` 在不使用 SDK 的情况下构建一个小型进程内协议模型。它验证当前请求信封与 Streamable HTTP 路由值，通过 `server/discover` 声明 Apps，列出工具和资源，执行工具，并提供一个自包含 HTML 资源。

模型接收已解析的正文和路由请求头。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或 `Accept`。完整的 Streamable HTTP 适配器请参见第 09 课；该适配器要求 `Content-Type: application/json`，并要求 `Accept` 值同时包含 `application/json` 与 `text/event-stream`。

运行：

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

检查输出中的以下五点：

1. 每次调用都彼此独立。
2. 每个请求都带有 `_meta` 能力。
3. `resources/list` 会在读取任何资源之前返回稳定描述符。
4. 每个结果都带有 `resultType` 和服务器身份元数据。
5. 不会出现任何核心会话标识符。

## 实际使用

从 `server/discover` 开始。确认 `io.modelcontextprotocol/ui` 出现在服务器扩展映射中。然后调用两次 `tools/list`：一次带 Apps 能力，一次不带。第一个响应会声明资源；第二个响应仍是可用的纯文本工具。

读取 `ui://notes/timeline.html`。在 HTML 中搜索 `hostOrigin` 和 `event.origin` 防护。这两行是桥接没有使用通配符目标的最小可见证据。

## 交付成果

本课交付 `outputs/skill-mcp-apps-spec.md`。在编写框架代码前，使用它审查 App 合约。它会迫使作者明确当前核心信封、扩展协商、回退方案、UI 资源、缓存策略、CSP、权限、桥接方法和同意边界。

## 练习

1. 将客户端能力改为空扩展映射。确认 `tools/list` 保留工具，但移除 UI 绑定。
2. 发送 `Mcp-Name: ui://notes/other.html`，而正文读取时间线。确认得到错误 `-32020`。
3. 将资源改为 `cacheScope: private`。说明何种用户特定条件使这一设置合理。
4. 将脚本移到 `https://static.example.com/app.js`。把该来源加入 `resourceDomains`，并解释新增的供应链风险。
5. 添加一个 `notes_open` 工具，并通过主机路由按钮点击。用户审批仍须留在主机中。

## 关键术语

| 术语 | 含义 |
|------|---------|
| MCP Apps | 可选扩展，用于在 MCP 主机中渲染交互式 HTML |
| `io.modelcontextprotocol/ui` | 通信双方都要声明的扩展标识符 |
| `ui://` | App UI 模板使用的资源方案 |
| `text/html;profile=mcp-app` | MCP App HTML 的 MIME 类型 |
| `server/discover` | 当前用于发现协议与能力的 RPC |
| `resources/list` | 服务器声明资源后必须实现的资源列表方法 |
| `resultType` | 现代成功结果必需的判别字段 |
| `ui/initialize` | Apps 桥接的第一个请求，与已移除的核心初始化相互独立 |
| `ui/notifications/initialized` | 主机响应后由 Apps View 发送的就绪通知 |
| CSP | 限制脚本、样式、图片和网络来源的浏览器策略 |
| 文本回退 | 对不支持 Apps 的主机仍保留的工具行为 |

## 延伸阅读

- [MCP 2026-07-28 基础协议](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps 概览](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps 构建指南](https://modelcontextprotocol.io/extensions/apps/build)
- [官方扩展支持矩阵](https://modelcontextprotocol.io/extensions/client-matrix)
