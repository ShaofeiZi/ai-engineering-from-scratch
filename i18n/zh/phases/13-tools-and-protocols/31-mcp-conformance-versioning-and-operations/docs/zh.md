# MCP 一致性工程：版本、证据与运维

> 仅仅因为服务器通过某个 SDK 走通了顺利路径，并不能说明它符合规范。一致性存在于线路数据、版本边界、中间设施以及回滚过程中。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 09 课（传输）、第 13 阶段 · 第 17 课（网关）、第 13 阶段 · 第 30 课（注册表准入）
**Time:** 约 100 分钟

## 学习目标

- 将 MCP 规范性规则转化为 golden 与 negative 线路记录。
- 将严格的 `2026-07-28` 行为与有边界的旧版回退分开。
- 区分新增的未知字段与无效的未知 `resultType`。
- 比较原始 JSON-RPC 证据与 SDK 规范化后的视图。
- 通过真实代理边界证明 header 与 body 的完整性。
- 使用已脱敏的记录、健康状态与回滚证据作为发布门禁。

## 问题

客户端通过某个 SDK 调用 `tools/list` 并获得工具，集成测试通过了。

这个结果仍留下许多重要问题：

- 请求是否携带了现代的逐请求协议元数据？
- `MCP-Protocol-Version`、`Mcp-Method` 和 `Mcp-Name` 是否与 JSON-RPC body 一致？
- 线路上的响应是否包含有效 `resultType`，还是由 SDK 合成了一个？
- 客户端是否会保留未来新增的字段？
- 一个已识别的现代错误会不会意外触发旧版握手？
- 代理是否保留了源站状态与 JSON-RPC error？
- notification serializer 是否发出了禁止出现的响应？
- 运维人员能否在不存储 secret 的前提下，证明某个版本为何获准发布或被回滚？

一致性是一组可观察的不变量。应在生产流量被迫发现问题之前，先构建能够捕获这些不变量的测试框架。

```figure
mcp-conformance-operations
```

## 从版本时代开始

MCP `2026-07-28` 使用自包含的逐请求元数据。现代请求携带 `params._meta.io.modelcontextprotocol/protocolVersion` 和 `params._meta.io.modelcontextprotocol/clientCapabilities`。带 namespace 的精确 key 很重要；裸 `protocolVersion` 或 `clientCapabilities` 别名属于格式错误。如果 HTTP 边界存在镜像路由 header，其值必须与 JSON-RPC body 一致。现代成功结果携带 `resultType`。

截至 `2025-11-25` 的版本使用较早的 initialization 时代。只有客户端已经选择该早期时代后，缺少 `resultType` 的旧版结果才会解释为 complete。

不要构建一个同时接受两种结构的宽松验证器。使用两个分支：

| 分支 | 进入依据 | 缺少 `resultType` | Initialization |
|---|---|---|---|
| 现代 | 成功的 `server/discover` 或已识别的现代响应 | 无效 | 不是默认路径 |
| 旧版 | 配置的 allowlist，加上无结论现代探测之后有效的旧版 `initialize` 结果 | 解释为 complete | 该时代要求执行 |

分离这两个分支，可以防止格式错误的现代 peer 因此获得更宽松的验证待遇。

### 严格模式

严格模式要求现代行为的明确证据。成功的 `server/discover` 可证明现代分支。已识别的现代 JSON-RPC error 也能证明。此时应修正请求或停止。绝不能因为服务器返回 `-32020`、`-32021` 或 `-32022` 而降级。

### 回退模式

回退模式执行一次有边界的现代探测。超时、空回复、连接关闭或无法识别的响应都只能说明结果不确定，不能证明对端是旧版。只有明确配置或列入兼容白名单的端点，随后才可以收到一次有边界的旧版探测；而且客户端只能在验证该探测的 `initialize` 结果和协商得到的旧版修订号后，选择旧版分支。

回退不是“任何错误后都尝试旧版”。已识别的现代错误包含有用的纠正信息。出错后降级可能掩盖 header 不匹配、缺少 capability 声明或不支持的版本。

这样可以防止攻击者、服务中断或过滤代理通过丢弃现代响应强迫降级。把端点策略、无结论的现代观察、精确的旧版正向证据和最终选择的时代一起记录。

在每份记录旁保存所选时代。如果缺少这项事实，同一个缺失字段可能在一次测试中显得可以接受，在另一次测试中却是无效的。

## 构建记录语料库

记录夹具应记录真正穿过边界的内容，而不仅是 SDK 调用：

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

保留两类夹具。

### Golden 记录

Golden 记录用于证明应被接受的行为：

- 元数据与 header 一致的现代发现或方法请求
- 包含必需字段的 complete 结果
- 方法可以请求更多输入时的 `input_required` 结果
- 仅在对应 capability 已公布后才接受 extension 结果
- 缺少 `resultType` 的旧版结果，但只能出现在已选择的旧版时代
- 处理 notification 时不产生 JSON-RPC 响应

Golden 记录应精确，而不必庞大。保持易变 ID 与时间戳确定，或者在比较前将其规范化。

### Negative 记录

Negative 记录用于证明拒绝行为：

- header 与 body 不匹配
- 缺少逐请求 capability
- 匹配但不受支持的协议版本
- 缺少现代 `resultType`
- 未知或未公布的 `resultType`
- 响应的 `jsonrpc` 不是 `2.0`，或者 ID 的值或 JSON 类型与请求不同
- 响应同时包含 `result` 和 `error`，或二者均不包含
- error 没有 integer `code` 和 string `message`
- 已知协议错误映射到错误的 HTTP 状态
- 为 notification 发出响应
- 格式错误的 JSON-RPC envelope
- 代理压扁协议错误

每个负例都要断言拒绝边界与稳定错误码。“调用失败”太过宽泛。代理生成的 500 与源站 `-32020` 都可能表现为失败，却向运维人员传达完全不同的信息。

header 不匹配夹具必须包含服务器实际返回的 HTTP 400 JSON-RPC 响应，并带有匹配的请求 ID 和错误码 `-32020`。只要本地验证器观察到 `HeaderMismatch`，就应自动强制验证这一点，不要把响应验证做成可选夹具标志。即使本地拒绝码正确，如果该用例得到 HTTP 500 且没有 body，它仍应失败。一个在自身请求验证器抛出异常后就停止的测试框架，只测试了自己，没有测试服务器的线路行为。

官方 MCP conformance 项目适合作为外部测试套件与带版本的参考。同时也要保留本地记录，因为它们能捕获通用套件不可能了解的代理、SDK、身份验证、扩展和发布路径。

## Header 值必须与 RPC Body 一致

在现代 Streamable HTTP 中，中间设施可以使用镜像 header 路由或执行策略。JSON-RPC body 仍是协议真相源。不匹配是完整性故障，不是一条让系统二选一的提示。

按以下顺序验证：

1. 解析并验证 JSON-RPC envelope 与元数据类型。
2. 比较 `MCP-Protocol-Version` 与 `params._meta.io.modelcontextprotocol/protocolVersion`。
3. 比较 `Mcp-Method` 与 `method`。
4. 如果该方法有路由名称，比较 `Mcp-Name` 与对应 body 值。
5. 确认相等后，再判断匹配的版本与 capability 集合是否受支持。

这一顺序能区分 mismatch `-32020` 与 unsupported version `-32022`。它也能阻止网关按 header 中的名称授权、源站却执行另一个 body 名称。

HTTP field 名称不区分大小写，而值仍区分大小写。查找前先规范化 header 名称，并拒绝相互冲突的重复值。对于不安全、非 ASCII 或带有前导/尾随空白的 `Mcp-Name`，在与 body 比较前，应解码精确的 `=?base64?{Base64EncodedValue}?=` UTF-8 哨兵。对于不完整哨兵、无效 Base64、无效 UTF-8 或未经编码的不安全值，使用 `-32020` 拒绝。即使 body 包含相同字符，原始的两端空白也无效，因为该值在传输前必须使用哨兵编码。

中间设施可能在请求到达 MCP 服务器前就拒绝格式错误的 HTTP，因此其失败可能是一个不含 JSON-RPC 的 HTTP error。应记录拒绝来自中间设施还是源站。如果源站 MCP 服务器处理的是有效 JSON-RPC 请求，就应使用协议错误契约。

## 未知字段不等于未知结果

前向兼容需要两条不同规则。

### 新增的未知字段

结果对象与 `_meta` 映射可以新增字段。验证器应根据自身角色保留或忽略新增字段，除非该字段违反保留契约。示例会把完整原始结果保存在证据中，并接受已知结果旁边的 `futureHint`。

如果你是透明代理，保留未知字段通常比删除更安全。如果你是应用客户端，忽略它也可能合理。差异测试仍应暴露 SDK 删除了该字段，让这种行为成为有意决定。

### 未知 `resultType`

`resultType` 是 discriminator。核心现代结果使用 `complete` 或 `input_required`。只有在相应 capability 已公布时，extension 才能增加其他值。例如，Tasks extension 可以在协商该 capability 的上下文中增加 `task`。

未知或未公布的 discriminator 不能安全地视为 complete。客户端不知道这样会丢弃哪种生命周期。应直接拒绝。

因此，同一个原始响应可以同时包含可接受的未知字段与不可接受的未知 result type。应分别测试两种情况。

discriminator 只是第一层。随后还要验证方法专属载荷。complete `tools/list` 结果需要 `tools` 数组，其中描述符具有唯一的非空名称、有用的描述以及以 object 为根的 `inputSchema`。`task` 结果只有在具备 Tasks capability 的合格 `tools/call` 中才有效，并要求 `taskId`、已知状态、创建与更新时间戳、`ttlMs`，以及有效的可选 polling interval。complete `completion/complete` 结果需要一个 `completion` 对象，其中最多包含 100 个 string 值；可选 `total` 必须是非负 integer，且不能小于已返回值数量；可选 `hasMore` 必须是 Boolean。即使 `resultType` 拼写正确，也无法让格式错误的载荷符合规范。

## Notification 不变量

JSON-RPC notification 没有 `id`。接收方不得发送 JSON-RPC success 或 error response。

对于通过验证的 HTTP notification 结构，测试框架期望 HTTP `202` 和空 body。MCP `2026-07-28` 没有定义通过 Streamable HTTP 从客户端发往服务器的核心 notification。示例只使用带 namespace 的课程 extension notification 来测试单向 serializer 不变量。不要把它描述成新的核心方法。

测试 serializer，而不仅是 handler。handler 可能返回 `None`，middleware 却把它包装成 JSON success 对象。应捕获最终 egress 字节。

## 添加 SDK 差异测试

SDK 经常把线路对象转换成更方便的语言类型。这很实用，但规范化对象无法证明实际收到的内容。

对于每个高风险 fixture，应捕获：

1. SDK 解码前的原始状态、header 与 response body。
2. SDK 规范化后的返回值或 exception。
3. 所选时代对应的预期语义投影。
4. SDK 提升、合成、删除或更改的字段。

示例允许 SDK 只删除 `resultType`、`_meta`、`ttlMs` 和 `cacheScope` 等已知线路记账字段，再比较应用载荷。它会报告被删除的 `futureHint`，因为这个未知语义字段消失了。

不要假定每项差异都是 SDK bug。目标是让转换可见。你需要判断组件是可以忽略新增字段的应用端点，还是应该保留该字段的透明中间设施。

对发布的每个 SDK 及其版本运行 differential。若两个 SDK 对同一份记录做出不同规范化，发布策略应提前说明哪些行为可以接受，而不是事后挑选最方便的输出。

## 捕获代理证据

大多数生产 MCP 故障会跨越多个进程。记录三种视图：

| 视图 | 最少证据 |
|---|---|
| Ingress | 请求 header、JSON-RPC body、content type、经过身份验证的 route、接收时间 |
| Origin | 转发的 header 与 body digest、源站状态、响应 header 与 body |
| Egress | 客户端可见的状态、header、body 与发送时间 |

示例检测两种常见转换：

- 源站 HTTP 400 或 404 JSON-RPC error 被转换成通用代理 500
- egress JSON-RPC body 与源站 body 不同

还应针对具体部署加入 content type、`Accept`、压缩、request-scoped SSE、cache header 和 trace 关联断言。在策略允许时捕获 TLS 终止点两侧。绝不能为了证明路径而记录凭据。

## 在证据离开内存前脱敏

脱敏是一致性运维的一部分，不是事后清理工作。应在序列化、哈希、日志记录、测试产物写入或故障上传之前完成。

示例先对 key 名称进行 case-fold 并移除 separator，再递归替换 `Authorization`、`Cookie`、`Set-Cookie`、`X-Api-Key`、`accessToken`、`clientSecret`、`registrationAccessToken`、`token`、`password`、`secret` 和 `api_key` 等 key 下的值。规范化过程与 denylist 必须使用同一种形式，避免 camelCase、连字符、下划线和点号变体绕过彼此的策略。生产 collector 还应添加方法专属的参数策略，因为 `query` 这类看似无害的 key 也可能包含个人数据或受监管数据。

对脱敏后的证据包计算 hash。原始 capture 只应在事件访问控制下短期保存。

## 把健康状态与回滚纳入门禁

协议一致性是发布的必要条件，但并不充分。符合协议的候选版本仍可能超时、泄漏内存或压垮依赖服务。

发布前定义健康窗口：

- 最小样本数
- 最大错误率
- 最大延迟 percentile
- saturation 或资源限制
- 观察时长
- 与已获准 baseline 的比较

发布前也要定义回滚证据：

- 精确的上一版本
- 准入证据 digest
- SHA-256 artifact 与描述符 pin
- 当前 Registry 状态
- 当前健康结果
- 路由恢复过程
- 由可信 release-controller 身份对这些精确字段作出的 attestation

要求在 promotion 前就验证回滚目标并确认其健康，而不是候选版本失败后才检查。没有可用恢复路径的成功发布，不算生产就绪。

如果候选版本失败，而回滚目标又缺少这些证据，应暂停流量，不能猜测。“回滚到之前的东西”不是运维控制。

不要把就绪性简化为 truthiness 检查，例如非空版本、`healthy: "yes"` 或任意 evidence 字符串。示例要求精确类型、active 状态、三个 SHA-256 digest、可信 signer，以及一份覆盖完整回滚载荷的有效 HMAC-SHA-256 attestation。其确定性 demo key 只是非 secret fixture。生产环境应在发布边界注入受保护 key、KMS 验证结果或公钥 attestation verifier。

发布门禁还会拒绝空的 transcript、SDK differential 或 proxy evidence。每个来源都必须携带有效 evidence digest。绿色健康窗口不能填补从未被观察过的边界。

## 构建它

运行标准库测试框架：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

演示会运行恰好十五份 golden 与 negative 记录，其中包括有效和格式错误的 completion 结果；比较原始结果与 SDK 视图；检查一个压扁源站错误的代理；评估健康状态；认证回滚证据，并选择该目标。

预期结构：

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

按以下顺序阅读 `code/main.py`：

1. `validate_request()` 执行特定时代的请求与 header 规则。
2. `validate_result()` 区分缺失的旧版 discriminator、有效现代值、extension 与未知值。
3. `select_era()` 实现严格策略与有边界的回退策略。
4. `run_transcript()` 评估 golden 与 negative fixture。
5. `compare_sdk_view()` 暴露规范化差异。
6. `inspect_proxy()` 比较 ingress、origin 与 egress 证据。
7. `redact()` 在对证据计算 hash 前移除明显 secret。
8. `rollback_evidence_ready()` 验证精确 pin 字段和可信发布 attestation。
9. `ReleaseGate.evaluate()` 将非空的一致性、SDK、代理、健康与回滚证据结合起来。

## 使用它

在四个节点运行测试框架：

1. 每次实现变化时，通过进程内 test adapter 运行。
2. 针对构建后的客户端和服务器二进制文件，通过真实传输运行。
3. 在 staging 环境中，通过已部署的代理或网关运行。
4. canary rollout 期间，结合实时健康与回滚证据运行。

在各层保持相同的稳定 case 名称。`negative-header-body-mismatch` 在 unit、end-to-end、proxy 和 canary 报告中应表示同一个不变量。因为边界变化，evidence digest 会不同；要求本身不应改变。

把 fixture schema 存入版本控制。把脱敏后的运行证据存入发布系统。原始 capture 只应在事件访问控制下短期保存。

## 交互实验

### 实验 A：证明时代边界

从 `code` 目录打开 Python：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

运行：

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

旧版调用会推断 `complete`。现代调用会抛出 `ProtocolViolation`。现在测试回退：

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

第一个超时会失败关闭，因为静默不是旧版证据。第二个调用选择旧版，仅仅因为配置允许且观察到了有效的旧版 initialization 结果。已识别的 missing-capability error 则证明应走现代分支。

### 实验 B：新增字段与 discriminator

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

第一个结果保留 `futureHint`。第二个结果因为 lifecycle discriminator 未知而被拒绝。

### 实验 C：检查 SDK 转换

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

判断你的组件可以忽略 `futureHint`，还是必须转发它。把这个选择写进发布策略，不能静默抹去 differential。

### 实验 D：修复代理

修改 demo exchange，使 egress 保留源站状态与 body。再次运行 `python3 main.py`。proxy issue 应消失，但 SDK differential 仍会阻止 promotion。然后在 SDK 视图中加入 `futureHint`；当每个证据来源都通过时，观察 action 变为 `promote`。

## 实践实验

向测试框架添加 request-scoped SSE 记录。

要求：

- 捕获响应状态、content type、有序 SSE event 与 stream 终止状态。
- 证明每个 JSON-RPC event 都包含符合所选时代的有效 result 或 error。
- 为代理在转发前缓冲完整 stream 添加一个 negative case。
- 为 JSON-RPC ID 与请求不同的 SSE event 添加一个 negative case。
- 写入证据前对 event data 脱敏。
- 在健康窗口中包含 stream 时长、首 event 延迟与 event 数量。
- stream 失败时，让发布门禁只选择有证据支持的回滚目标。

成功标准是：同一个 case 能直接运行，也能通过代理运行，并由报告指出究竟是哪一处边界改变了行为。

## 交付产物

本课交付 `outputs/skill-mcp-conformance-release-gate.md`。用它可以把服务器、客户端、网关或 SDK 变更转化为带版本的一致性矩阵与发布决策。该产物要求原始线路证据、negative case、显式时代选择、SDK differential、代理证明、脱敏、健康阈值和回滚证据。

## 验证

运行 demo 和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- 每个已包含的 golden 与 negative 记录都得到预期结果
- 现代请求要求精确的 namespaced 元数据 key
- HTTP header 名称按大小写不敏感方式匹配，编码后的 `Mcp-Name` 值得到精确解码
- header 与 body 不匹配时返回现代 mismatch 错误码
- 响应版本、ID、result/error 互斥性、error 结构与 HTTP 映射得到验证
- 方法专属的工具列表、task 与 completion 载荷要求得到执行
- 每个观察到的 `HeaderMismatch` 都必须有一份真实 HTTP 400 JSON-RPC `-32020` 响应
- 原始 `Mcp-Name` 空白被拒绝，精确哨兵编码的空白可以往返还原
- 缺失 `resultType` 只在已选择的旧版时代中有效
- 新增字段能通过原始验证，而未知 result type 会失败
- extension result type 要求对应 capability 已公布
- 已识别的现代错误绝不会导致旧版回退
- notification 不产生 JSON-RPC 响应
- SDK bookkeeping 字段删除与语义字段丢失能够区分
- 代理错误压扁能够检测，凭据会在 camelCase 与各种 separator 变体中递归脱敏
- promotion 要求非空 transcript、SDK、proxy 与健康运维证据
- promotion 和 rollback 都要求已认证、已固定、active 且健康的回滚目标

## 生产故障模式

| 故障 | 薄弱测试报告的内容 | 测试框架必须证明的内容 |
|---|---|---|
| SDK 合成缺失的 discriminator | “tools/list 通过” | 原始现代结果缺少 `resultType`，因此无效 |
| 客户端在 `-32021` 后降级 | “旧版重试成功” | 已识别现代错误禁止回退 |
| 未知 result type 被视为 complete | “响应已解析” | 未公布的 lifecycle discriminator 被拒绝 |
| 代理授权一个工具，源站执行另一个 | “请求到达服务器” | 每一跳的 `Mcp-Name` 都等于 body 路由名称 |
| 测试框架在读取服务器响应前抛出异常 | “header mismatch 测试通过” | 捕获并验证 HTTP 400 与 JSON-RPC `-32020` 响应 |
| 代理把源站 400 转成通用 500 | “上游错误” | 源站与 egress 状态及 JSON-RPC body 均被保留 |
| Notification middleware 发出 `{result: null}` | “handler 返回 none” | 最终 egress body 为空，不存在 JSON-RPC 响应 |
| SDK 删除新增字段 | “typed object 相同” | 原始与规范化视图显示具体丢失字段 |
| 故障 artifact 泄漏 bearer token | “debug bundle 已上传” | 在 hash、日志记录或上传之前完成脱敏 |
| 凭据 key 风格绕过脱敏 | “denylist 包含 api_key” | camelCase 与 separator 变体使用同一种规范 denylist 形式 |
| Canary 没有样本却看似健康 | “零错误” | 强制执行最小样本数 |
| 回滚选择未知 build | “上一部署已恢复” | 目标版本、准入 digest、pin、状态与健康信息齐全 |

## 运维规则

测试你发送的字节、每个中间设施转发的字节、每个 SDK 暴露的语义，以及运维人员在压力下使用的证据。兼容性是显式分支。回滚是有证据支持的发布动作。二者都不应成为宽松 parser 的意外副作用。

## 延伸阅读

- [MCP 2026-07-28 基础协议](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP 版本协商](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [官方 MCP conformance 项目](https://github.com/modelcontextprotocol/conformance)
