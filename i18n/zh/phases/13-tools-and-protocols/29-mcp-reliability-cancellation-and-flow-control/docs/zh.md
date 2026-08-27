# MCP 可靠性、取消与流量控制

> 请求 ID 只能关联消息。它不能保证副作用安全，不能停止 worker，也不能保护数据流免受慢速消费者影响。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 09、13 课
**Time:** 约 120 分钟

## 学习目标

- 为 stdio 和 Streamable HTTP 实现正确的取消信号。
- 解决完成与取消之间的竞态，并避免在取消后继续发送消息。
- 区分请求取消与持久化 `tasks/cancel` 语义。
- 根据副作用和显式幂等键做出重试决策。
- 在保留最终响应的同时限制 progress 队列。
- 通过重新连接、重新获取和带抖动的退避恢复数据流。

## 问题

顺利路径会掩盖代价最为高昂的分布式系统 bug。

客户端调用工具。服务器开始工作。progress 不断到达。代理缓冲数据流。客户端达到超时并断开。服务器在一毫秒后完成。客户端使用新的 JSON-RPC ID 重试。于是 mutation 执行了两次。

每个组件在本地都表现正确，整个系统却在全局层面失败了。

MCP 定义消息与传输行为，但你的应用仍要负责：

- 时间预算；
- 业务幂等性；
- 有界队列；
- 重试分类；
- 持久化 Task 状态；
- 重新连接与重新获取策略。

本课把这些决策构建进一个确定性模拟器。代码中没有 sleep、socket 或
随机故障。你可以直接控制取消事件的顺序。一个同步线程测试会迫使两个
ledger 客户端争夺同一个幂等键。

## 请求取消取决于传输方式

每种传输的意图都相同：客户端不再需要某个进行中请求的结果。线路上的信号则各不相同。

### stdio

stdio 使用一条共享的双向通道。客户端发送 notification：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

该 notification 只发不等。服务器不会为它发送 JSON-RPC response。

服务器应停止工作、释放资源，并避免为已取消请求发送响应。如果请求未知、已经完成，或无法安全停止，服务器可以忽略取消。

格式错误、目标未知和已经完成的取消 notification 都应忽略。把这些竞态转换为新错误，只会制造更多竞态。

### Streamable HTTP

现代 Streamable HTTP 为每个请求提供独立的 HTTP response 或 SSE response stream。客户端通过关闭该请求的 response stream 来取消。

不要为普通 HTTP 请求 POST `notifications/cancelled`。stream 关闭本身就是取消信号。

服务器观察到断开连接后，应停止工作，并且不得再为该请求发送消息。

### 服务器发起的取消范围很窄

服务器不能用 `notifications/cancelled` 随意取消客户端调用。在 stdio 上，服务器发起的取消仅用于终止 `subscriptions/listen` 请求。应将这条路径与普通客户端请求取消分开。

## 取消是一种竞态

以下两种事件顺序都有效。

### 取消获胜

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### 完成获胜

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

客户端也必须忽略它已经放弃的请求所收到的迟到响应。由于网络延迟，两端都无法证明对方先观察到了哪个事件。

```figure
mcp-reliability-race
```

本课的 `RequestCoordinator` 只存储一个终态。取消后，`complete()` 不会返回任何响应。迟到的取消无法更改已经完成的记录。

## 超时需要两只时钟

仅使用一个 inactivity timer 并不够。

应设置两种限制：

1. **空闲超时。** 请求最长可以多久不产生活跃的有效活动。
2. **最大超时。** 从请求开始计算的绝对 wall-clock 预算。

progress 可以重置空闲时钟，却绝不能移除最大截止时间。

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

在 1500 ms 时，请求仍然活跃，因为距最近一次 progress 只有 300 ms。在 2000 ms 时，即使 1999 ms 刚到达另一个 progress event，最大截止时间仍会取消请求。

progress 是可选的。服务器可以接受 progress token，却不发出任何更新。绝不能因为 token 存在，就把它变成无限超时。

MCP progress 值必须递增。完成或取消后停止 notification。还要限制 progress 速率，防止快速 worker 淹没传输层。

## 请求取消不等于 `tasks/cancel`

这些机制解决的是不同的生命周期。

| 机制 | 目标 | 信号 | 成功的含义 |
|-----------|--------|--------|--------------------|
| stdio 上的请求取消 | 一个进行中的 RPC | `notifications/cancelled` | 客户端已放弃请求；服务器应在可行时停止 |
| HTTP 上的请求取消 | 一条进行中的 response stream | 关闭 stream | 客户端已放弃请求；服务器应在可行时停止 |
| `tasks/cancel` | 一个持久化 Task | 普通 MCP 请求 | 服务器已确认取消意图 |

`tasks/cancel` 成功并不能证明 worker 已经停止。在某个 worker checkpoint 观察到标志前，Task 可能仍保持 `working`。而且工作可能在到达该 checkpoint 前就已完成。

HTTP 连接关闭时，不要清除持久化 Task 状态。创建 Task 的原因，正是让它的生命周期超越单次请求和单条连接。

## 新 JSON-RPC ID 不代表幂等

JSON-RPC ID 用来关联请求与响应，并不标识业务操作。

假设客户端用 ID `41` 提交一笔扣款，响应丢失后改用 ID `42` 重试。服务器看到的是两条不同消息。如果没有应用级 key，它无法知道二者代表同一次结账。

幂等键标识的是业务意图：

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

服务器存储：

- key；
- 操作参数的 fingerprint；
- 已提交的结果。

同一个 key 加相同参数会返回已存储结果。同一个 key 配上不同参数则会被拒绝。这样可以防止意外复用 key 导致另一项业务操作发生 mutation。

### Ledger 边界必须原子且持久

以下顺序并不安全：

```text
check key
run mutation
store result
```

两个 worker 可能同时观察到 key 不存在，并都运行 mutation。如果 effect
发生后、存储前进程崩溃，重试时也会产生同样的歧义。

本课使用文件支持的 SQLite ledger。`BEGIN IMMEDIATE` 把 key 检查、
模拟业务 effect、执行计数器和结果存储序列化到同一个事务中。因此，
使用相同 key 竞争的两个独立 ledger 连接，只会观察到一次已提交结果和
一次执行。关闭再重新打开 ledger 后，该记录仍然存在。

每个返回值都由已保存的 JSON 重新构造。调用方永远拿不到 ledger 内部
持有的可变对象，因此修改返回的 dictionary 不会损坏后续 replay 结果。

模拟器的业务 effect 是同一个 SQLite 事务中的 receipt 和执行计数器。
真实付款、部署或外部 API 调用，并不会因为写入本地表就自动变成原子操作。
生产系统需要持久化共享数据库事务、transactional outbox，或由上游
提供商强制执行同一个幂等键。仅靠进程锁无法保护多个 replica，也无法
经受重启。

### 重试矩阵

实现重试前先完成分类。

| 类别 | 示例 | 重试规则 |
|------|---------|------------|
| 安全 | 没有副作用的确定性读取 | 理解故障边界后，使用新的 JSON-RPC ID 重试 |
| 有条件 | 带持久化幂等键的 mutation | 使用同一个 key 和完全相同的参数重试 |
| 不安全 | 没有业务去重机制的 mutation | 不要自动重试；先核对状态 |

`readOnlyHint` 和 `idempotentHint` 等工具注解仍然只是不可信提示。真正决定重试是否安全的，是应用契约和服务器实现。

## 背压是正确性的一部分

SSE 生产者生成 progress 的速度可能高于客户端、代理或网络的消费速度。无界队列会把处理缓慢转变为内存耗尽。

使用有界队列，并明确哪些内容可以丢失。

progress 可以替换。同一个 token 的较新 progress 值取代旧值。最终 JSON-RPC 响应则不可替换。

本课缓冲区应用以下策略：

1. 合并同一 token 的相邻 progress。
2. 达到容量时丢弃最旧的 progress。
3. 把 stream 标记为需要从权威来源重新获取。
4. 保留最终响应。
5. 如果保留某个最终响应意味着必须丢弃另一个最终响应，则拒绝进入这种状态。

这是带显式恢复机制的有界丢失。静默丢失不是一种策略。

### 代理缓冲

服务器可能正确地进行流式传输，但反向代理仍会把事件留在缓冲区中。

对于 SSE response，发送：

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

2026 Streamable HTTP 规范建议使用 `X-Accel-Buffering: no`，使兼容代理立即传递事件。

对于长期保持但较安静的 stream，定期发出 SSE comment：

```text
:
```

客户端会忽略 comment 行。中间设施能观察到流量，因此不太可能关闭空闲连接。

Keepalive 不等于 progress。不能仅仅因为传输注释到达，就重置操作的语义空闲超时。

## 重新连接意味着重新获取

现代 Streamable HTTP 不支持通过 `Last-Event-ID` 恢复 SSE。

`subscriptions/listen` stream 中断后：

1. 使用新的 JSON-RPC ID 创建新的 listen 请求。
2. 恢复所需的 subscription filter。
3. 通过权威方法重新获取受影响的工具、资源、提示词或 Task。
4. 根据稳定标识符去重应用状态。
5. 不要仅仅因为响应丢失，就重放不安全的 mutation。

示例恢复方案会明确把 `sendLastEventId` 设为 false，并列出需要重新获取的资源。

### 防止重新连接惊群

如果 10,000 个客户端都在恰好一秒后重连，正在恢复的服务器会再次崩溃。

使用带抖动且有上限的指数退避。本课根据客户端 ID 和尝试次数计算确定性抖动，使测试保持可复现：

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

生产环境可以使用密码学安全随机数或运行时随机数。不变量是让重试时间分散，而不是某一个具体公式。

## 构建它

`code/main.py` 构建了五个小型可靠性组件。

### `RequestCoordinator`

- 使用空闲截止时间和最大截止时间启动一个进行中请求；
- 发出单调递增的 progress notification；
- 生成正确的 stdio 或 HTTP 取消信号；
- 忽略无效取消 notification；
- 明确表达取消与完成之间的终态竞态；
- 将服务器发起的取消保留给 stdio subscription。

### `MutationLedger`

- 证明没有业务 key 时，两个 JSON-RPC ID 会导致执行两次；
- 使用文件支持的 SQLite 事务完成 key 检查、模拟副作用、执行计数器和
  结果提交；
- 在独立 ledger 连接之间，按一个幂等键去重相同参数；
- 拒绝一个 key 搭配不同参数复用；
- 返回防御性副本，并在重新打开后保留已提交记录。

### `DurableTaskService`

- 确认取消请求；
- 在 worker checkpoint 之前让 Task 保持 `working`；
- 演示为什么确认不等于最终状态。

### `BoundedSseBuffer`

- 在压力下合并或丢弃 progress；
- 记录需要从权威来源重新获取；
- 永远不丢弃最终响应。

### 恢复辅助函数

- 返回适合经过代理传输的 SSE 请求头和 keepalive 注释；
- 创建重新连接和重新获取计划；
- 使用确定性的指数退避与 jitter 分散重试。

## 使用它

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

演示会运行核心竞态的两种结果，在临时文件支持的 ledger 中执行一次
事务性去重 mutation，使有界 progress buffer 过载，并展示持久化 Task
如何从“已确认取消”转为“worker 已观察到取消”。

## 交互实验

不添加 sleep，运行四种事件顺序。

1. 启动请求 `A`，取消它，然后调用 `complete()`。
2. 启动请求 `B`，完成它，然后送达取消。
3. 启动请求 `C`，在每个空闲截止时间前发出 progress，然后越过最大截止时间。
4. 通过 Streamable HTTP 启动请求 `D`，并关闭它的响应流。

为每个场景记录：

- 请求的终态；
- 是否存在最终响应；
- 放在线路上的取消信号；
- 客户端应忽略哪个事件。

然后把 `D` 改为 stdio。操作相同，但取消信号必须改变。

## 实践实验

添加一个 `reserve_inventory` mutation 到 `MutationLedger`。

要求：

1. key 绑定 SKU、数量、tenant 和操作名称。
2. 使用同一个 key 和相同参数重试时，返回第一次 reservation。
3. 使用相同 key 但更改数量重试时失败，且不能再次 reservation。
4. 已提交但响应丢失的执行，可以按 key 核对。
5. 结果不记录 secret 或支付数据。
6. 客户端没有提供 key 时禁用自动重试。
7. 模拟 subscription 中断，并在决定下一步操作前重新获取 inventory 记录。
8. 在屏障点启动两个 ledger 连接，并发提交同一个 key。断言只提交了一次 reservation。
9. 修改第一次返回的 reservation 对象。重放该 key，并证明已存储结果没有变化。
10. 关闭并重新打开 ledger 文件，然后按 key 核对 reservation。

实验必须忠实反映真实边界：如果 inventory 位于另一个服务中，应说明
该服务是否接受同一个幂等键，或事务型 outbox 是否负责连接
本地提交与远程副作用。

## 交付产物

`outputs/skill-mcp-reliability-reviewer.md` 是一项扁平的可靠性审查 Skill。向它提供 MCP 操作、传输方式、超时策略、重试行为、队列策略与恢复方案，它会返回竞态表、重试分类、幂等边界、流量控制检查和故障夹具。

## 验证

满足以下陈述时，本课即告完成：

- stdio 取消发送 `notifications/cancelled`，并且不接收响应。
- Streamable HTTP 取消会关闭请求流，不发送取消 POST。
- 先取消后完成会抑制最终响应。
- 先完成后取消会保留响应，并忽略迟到的取消。
- progress 可以重置空闲超时，但绝不能重置最大超时。
- 仅换一个新的 JSON-RPC ID 会再次执行 mutation。
- 在两个连接并发竞争时，一个幂等键加完全相同的参数只会执行一次。
- 已提交记录在重新打开后仍然存在，replay 返回防御性副本。
- 修改某一次返回结果无法改变已存储结果。
- 有界缓冲区始终不超过容量，并保留最终响应。
- 重新连接会创建新请求，不发送 `Last-Event-ID`，并重新获取受影响状态。
- `tasks/cancel` 确认后，Task 在 worker 观察到它之前仍保持非终态。

## 生产故障模式

| 故障 | 可观察症状 | 正确响应 |
|---------|--------------------|------------------|
| HTTP 客户端 POST 取消 notification | 服务器与客户端对请求生命周期理解不一致 | 关闭请求的 SSE 响应流 |
| 服务器在接受取消后仍响应 | 客户端收到无法使用的迟到结果 | 取消获胜时停止工作并抑制后续消息 |
| progress 重置所有截止时间 | 卡死的工作永远不会终止 | 保留单独的绝对最大超时 |
| 把新 RPC ID 当作去重手段 | 扣款、部署或删除执行两次 | 添加持久化应用幂等键 |
| Key 检查与副作用相互分离 | 并发 worker 都观察到 key 缺失 | 原子提交 key 占用、副作用记录和结果 |
| 跨副本使用内存 ledger | 重启或另一 worker 忘记既有提交 | 使用共享持久化存储或上游幂等机制 |
| 直接返回已存储可变结果 | 调用方的修改破坏后续 replay | 序列化已提交结果并返回防御性副本 |
| Key 搭配变化后的参数复用 | 一个 key 指代两项业务意图 | 保存并比较参数 fingerprint |
| 无界 progress 队列 | 慢速消费者使内存不断增长 | 在容量限制内合并并丢弃可替换的 progress |
| 压力下丢弃最终响应 | 客户端无法得知请求结果 | 预留容量或淘汰 progress，绝不丢最终响应 |
| 代理缓冲 SSE | Progress 成批到达或在超时后到达 | 禁用缓冲，并配置兼容的代理超时 |
| 假定存在 `Last-Event-ID` | 客户端从服务器不支持的状态恢复 | 使用新请求重连并重新获取 |
| 所有客户端立即重连 | 恢复过程制造另一次中断 | 使用带抖动且有上限的指数退避 |
| 把 Task 确认当成最终取消 | UI 显示已停止后 worker 仍在运行 | 轮询 Task，直到进入终态 |

## 与综合项目的联系

工具生态综合项目应把可靠性视为可执行证据，而不是架构图中的一段文字。

要求提供以下产物：

- 每种传输各一份取消竞态记录；
- 每个公开 mutation 的重试表；
- 一条幂等键记录与一个不匹配夹具；
- 一份并发同 key 记录、重新打开检查及 mutation alias 检查；
- 一次有界缓冲区过载结果；
- 反向代理 SSE 请求头与空闲策略；
- 一份列出权威重新获取方法的重连方案；
- 综合项目使用 Task 时的一份持久化 Task 取消追踪。

本地进程中的一次绿色请求，只能证明顺利路径。只有当响应丢失、迟到取消、慢速消费者和重连惊群都有确定结果时，综合项目才具备生产就绪性。

## 关键术语

| 术语 | 含义 |
|------|---------|
| Request cancellation | 放弃一个进行中的 MCP 请求 |
| Cancellation race | 终态完成事件与取消事件之间的竞争 |
| Idle timeout | 从最近一次有效请求活动开始计算的限制 |
| Maximum timeout | 从请求开始计算、不受 progress 影响的绝对限制 |
| Idempotency key | 对一项业务意图去重的应用标识符 |
| Atomic ledger | 将 key 占用、effect 记录和结果作为一个单元提交的持久化边界 |
| Backpressure | 生产者速度超过消费者时施加的控制 |
| Progress coalescing | 用更新且具有权威性的 progress 替换旧 progress |
| Refetch | stream 出现缺口后重新读取当前状态 |
| Jitter | 将重试分散到不同时间点的有意变化 |

## 延伸阅读

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
