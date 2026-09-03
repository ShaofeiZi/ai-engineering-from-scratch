---
name: task-store-designer
description: 使用当前 Tasks 扩展、无状态请求、显式所有权、轮询、输入更新与取消机制，设计持久化的 MCP 工作。
version: 2.0.0
phase: 13
lesson: 13
tags: [mcp, tasks, extension, durable-state, stateless]
---

基于 `io.modelcontextprotocol/tasks` 扩展设计长时间运行的 MCP 工作。

产出：

1. 资格判定。解释为什么该操作需要任务，而非同步的 `tools/call`。
2. 能力契约。在 `server/discover` 中展示精确的 `supportedVersions`、capabilities、`ttlMs` 和 `cacheScope`，以及每请求客户端能力中的 Tasks 扩展。如果通告了工具，须包含强制确定性的 `tools/list` 描述符及有效的对象类型 `inputSchema`、服务器身份元数据和缓存提示。当扩展缺失时使用 `-32021` 并附带 `requiredCapabilities` 对象；当版本不受支持时使用 `-32022` 并附带精确的 `supported` 和 `requested` 数据。
3. 创建事务。持久化任务直到 `tasks/get` 能够解析它，然后返回服务器定向的 `resultType: "task"`。
4. 状态结构。包含 `taskId`、`status`、`statusMessage`、ISO 时间戳、`ttlMs`、`pollIntervalMs`、权威所有者、原始操作引用、结果或错误、未完成的输入请求以及所有已签发的 input key。已完成任务中嵌套的 `CallToolResult` 必须包含 `resultType: "complete"`，并且 SHOULD 包含其自身的 `io.modelcontextprotocol/serverInfo` 元数据。
5. 当前方法。定义 `tasks/get`、`tasks/update` 和 `tasks/cancel`。对于 Streamable HTTP，每个请求将 `Mcp-Name` 设置为 `params.taskId`。不得引入 `tasks/status`、`tasks/result` 或 `tasks/list`。
6. 输入续传。将创建前的 MRTR 与创建后的 `tasks/get` 加 `tasks/update` 分开。要求 input key 在生命周期内唯一，并处理部分响应。
7. 持久化方案。选择原子文件系统存储、事务型数据库或共享队列与存储。包含 worker 租约与重启行为。
8. 所有权策略。按租户和主体对每个任务方法和订阅进行授权。绝不可将 task-id 的知晓视为权限。
9. 取消契约。声明确认是协作式的，且未必会导致 `cancelled`。
10. 通知选项。在 POST 响应 SSE 流上使用 `subscriptions/listen` 和 `notifications/tasks`，以轮询为基线。在确认和每个任务通知中放入 `io.modelcontextprotocol/subscriptionId`，其值等于 listen 请求 id。无 id 的通知不接收 JSON-RPC 响应；被接受的 HTTP 通知接收 `202` 且无 body。
11. 过期策略。从创建时起解读 `ttlMs`，定义清理行为，并避免泄露其他租户任务是否存在的信息。
12. 迁移映射。用当前扩展流程替换客户端请求的任务标志和已移除的实验性方法。

硬性拒绝：

- 在持久化读可见性之前返回任务句柄。
- 向未通告该扩展的请求返回 `resultType: "task"`。
- 将 `params._meta.task.required`、`tasks/status`、`tasks/result` 或 `tasks/list` 用作当前 API。
- 将 `initialize`、`Mcp-Session-Id`、粘性路由或隐藏的传输会话状态作为任务存储。
- 将 `tasks/cancel` 确认视为 worker 已停止的证据。
- 在一个任务生命周期内重用 `inputRequests` 的 key。
- 将任务返回给非其权威所有者的调用方。
- 通过独立 GET、会话 SSE 或 `Last-Event-ID` 重放实现通知投递。

拒绝规则：

- 除非调用方给出具体的持久化需求，否则拒绝为快速确定性查找创建任务。
- 当工作必须能在进程重启后存活时，拒绝仅使用内存的生产存储。
- 拒绝无界的结果载荷；将大型产物外部存储并返回经授权的资源句柄。
- 拒绝缺少显式租户所有权、过滤、分页和留存策略的历史端点。

输出一页式设计，包含生命周期表、线路方法、持久化事务、所有权规则、输入流程、轮询节奏、取消语义、订阅选项、过期清理、故障模型和遗留迁移映射。
