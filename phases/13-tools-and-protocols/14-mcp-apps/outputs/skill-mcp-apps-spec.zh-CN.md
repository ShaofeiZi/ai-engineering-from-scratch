---
name: mcp-apps-spec
description: 在无状态的 2026-07-28 协议上设计并审查 MCP App 契约。
version: 2.0.0
phase: 13
lesson: 14
tags: [mcp, apps, stateless, ui-resources, csp, sandbox]
---

给定一个可能需要交互式视图的 MCP 工具，产出一份与框架无关的契约。

## 必需输入

- 工具名称、参数、普通文本结果和结构化结果。
- 视图必须支持的用户交互。
- 数据敏感性，以及响应是否会因授权上下文而异。
- 视图所需的浏览器权限和外部源。
- 在不支持 Apps 的宿主中的纯文本行为。

## 产出

1. 当前核心信封。展示 `2026-07-28`、按请求的 `protocolVersion`、`clientCapabilities`、推荐的 `clientInfo`、匹配的 `Mcp-Method` 和 `Mcp-Name` 头，以及 `resultType` 响应。
2. 发现条目。在 `server/discover` 中通告 `io.modelcontextprotocol/ui`，并设置保守的 `ttlMs` 和 `cacheScope`。
3. 工具声明。在 `tools/list` 返回的工具上放置嵌套的 `_meta.ui.resourceUri`。不要等到 `tools/call` 才暴露 UI。
4. 资源契约。在 `resources/read` 之前包含确定性的 `resources/list` 元数据。提供一个规范的 `ui://` URI、稳定的名称和描述、`text/html;profile=mcp-app`、缓存提示、CSP 域名列表（`connectDomains`、`resourceDomains`、`frameDomains`、`baseUriDomains`）以及最小权限对象。
5. 结果契约。无论宿主是否渲染 App，都返回有用的文本和结构化数据。
6. 桥接契约。列出每个 Apps `ui/*` 或代理方法、确切的消息来源、参数 schema、结果 schema 以及宿主侧的同意检查。
7. 回退。描述当客户端省略 Apps 扩展能力时的工具和结果。
8. 验证表。覆盖以下情况：路由前的 HTTP 400 `-32020` 头不匹配、包含确切支持版本和请求版本数据的 HTTP 400 `-32022`、包含 `data.requiredCapabilities` 的 HTTP 400 `-32021`、HTTP 404 `-32601`、202 空体通知、CSP 违规、不可信内容、未授权的桥接调用以及文本回退。
9. 传输边界。如果实现接收的是已解析的请求和头，将其标记为进程内协议模型，并将其与第 09 课的完整 Streamable HTTP 适配器关联。真正的适配器必须要求 JSON Content-Type 以及包含 JSON 加 SSE 的 Accept 值。

## 严格拒绝

- 将核心 `initialize`、`notifications/initialized` 或 `Mcp-Session-Id` 路径呈现为当前 MCP。
- 通配符 `postMessage` 目标源，或跳过 `event.origin` 验证的接收器。
- 仅在工具运行后才暴露的 UI 绑定。
- 通配符 CSP 域名列表、无限制的网络源，或没有可见功能的权限。
- 用户控制的 HTML 在没有定义净化边界的情况下被插入。
- 将 iframe 点击视为宿主授权的有后果的 UI 操作。
- 通告资源但省略 `resources/list` 的服务器。
- 任何针对无 `id` 通知的 JSON-RPC 响应体。

## 兼容性边界

旧版扁平 UI 元数据可作为回退被读取，但新输出使用嵌套的 `_meta.ui.resourceUri`。`ui/initialize` 仅在被标识为 Apps postMessage 握手时才被允许。它绝不替代已移除的 MCP 核心初始化。

## 输出格式

返回一个紧凑设计，包含以下标题：Core Wire、Discovery、Tool、Resource、Result、Bridge、Security、Fallback、Verification。以风险最高的来源、权限或同意假设作为结尾。
