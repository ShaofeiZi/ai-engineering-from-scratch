---
name: mcp-transport-migrator
description: 将基于会话的旧版 MCP HTTP 传输迁移至无状态、仅 POST 的 2026-07-28 契约。
version: 2.0.0
phase: 13
lesson: 09
tags: [mcp, streamable-http, stateless, migration, headers]
---

给定一个基于会话的 Streamable HTTP 或 HTTP+SSE 服务器，为 MCP `2026-07-28` 生成一份迁移操作手册。

需产出：

1. 端点映射。定义一个接受 POST 的现代 MCP 端点。每个 JSON-RPC 请求或通知都通过一个新的 POST 发送。
2. 响应映射。对单个响应使用 `application/json`，或对相关通知后跟最终响应的情况使用请求作用域的 `text/event-stream`。
3. 已移除行为。对现代 GET 和 DELETE 返回 `405`。忽略 `Mcp-Session-Id` 和 `Last-Event-ID`；绝不生成、回显、撤销或恢复它们。
4. 请求元数据。要求每个请求体 `_meta` 中包含协议版本和客户端能力，并建议附带客户端身份信息。
5. 头部校验。要求 `MCP-Protocol-Version`、`Mcp-Method` 以及条件性的 `Mcp-Name`。解码 Base64 哨兵值并将头部与请求体进行比对。不匹配时返回 `-32020`。当匹配的版本不受支持时返回 `-32022`，并附带精确的数据键 `supported` 和 `requested`。
6. 订阅迁移。用 POST `subscriptions/listen` 替换独立的 GET、`resources/subscribe` 和 `resources/unsubscribe`。在确认回执、每条通知以及最终结果上标注 `io.modelcontextprotocol/subscriptionId`，其值等于 listen 请求的 id。
7. 状态迁移。用绑定到已认证主体的显式、不透明的应用句柄替换连接亲和性。
8. 兼容窗口。将旧端点保持独立并清晰标注。在任何旧版回退之前，必须先检查现代 POST 错误。不要使用 `301` 或 `302` 重定向 POST，因为方法和请求体的保留是不安全的。
9. 验证。测试 Origin 拒绝、POST 媒体协商、请求体元数据、镜像头部、JSON 响应、无请求体的已接受通知 `202`、作用域内 SSE 订阅元数据、GET 和 DELETE 的 `405`、被忽略的已移除头部，以及使用新 id 的中断流重试。

严格拒绝：

- 将会话 id、独立 GET、DELETE 或重放呈现为现代行为。
- 通过进程或连接记忆共享每请求能力。
- 发送服务端发起的 JSON-RPC 请求。
- 使用 `Last-Event-ID` 恢复现代 SSE 流。
- 在识别出现代错误后回退到旧版。
- 在迁移期间使用重定向来移动 JSON-RPC POST。

拒绝规则：

- 拒绝在无认证、授权和精确 Origin 策略的情况下公开暴露。
- 拒绝将隐藏的粘性路由作为显式工作流状态的替代方案。
- 拒绝在没有应用级幂等控制的情况下自动重试非幂等操作。

输出一份迁移前后端点对照表、分阶段上线方案、回滚边界以及可执行的合规检查清单。最后给出旧版路由将被移除的确切日期。
