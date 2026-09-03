# MCP 工具契约与内容

> 只有当发现、参数、结果、分页和传输元数据遵循同一份契约时，工具才适合安全地自动化调用。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 07、09、10 课
**Time:** 约 120 分钟

## 学习目标

- 使用 JSON Schema 2020-12 定义工具输入和输出。
- 验证结构化结果，而不假定它们一定是 JSON 对象。
- 在文本、图像、音频、资源链接和嵌入资源之间做出选择。
- 在工具暴露给模型之前，拒绝不安全的 `x-mcp-header` 定义。
- 对参数 header 值进行编码，并验证 header 与 body 精确一致。
- 遍历 cursor 分页，而不解释 cursor 的值。
- 限制 `completion/complete` 建议的范围并执行授权。

## 问题

调用一个 Python 函数很容易。让 AI 宿主调用远程能力，则是一个契约问题。

服务器发布描述符。客户端把描述符转成模型上下文和用户界面。模型创建参数。网关可能根据镜像到 header 中的信息路由请求。服务器执行工具。最后，客户端判断结果是否足够安全、有效，能够返回给模型。

任何一个薄弱边界都会破坏整条链路。

考虑以下五种故障：

- 描述符声明结果是对象，但服务器返回数组。
- `nextCursor` 是空字符串时，客户端停止分页。
- token 参数被镜像到 HTTP header，因而暴露给中间设施。
- Unicode 路由值未经处理便作为原始 header 发送，网关和源站随后对同一组字节作出不同解释。
- completion 端点向没有访问权限的调用方建议生产环境。

这些问题无法靠更好的 prompt 修复。它们需要明确的协议契约与应用契约。

## 契约流水线

把每次工具调用视为五道关卡：

1. **发现。** 读取确定性的分页工具列表。
2. **准入。** 验证每个描述符，并应用本地安全策略。
3. **调用。** 验证参数并构造传输元数据。
4. **执行。** 运行 handler，并正确分类故障。
5. **消费。** 在模型使用前验证 content block 和结构化输出。

```figure
mcp-contract-pipeline
```

准入关与消费关由宿主掌控。服务器无法强迫客户端信任自己的 annotation、schema 或输出。

## JSON Schema 是运行时边界

在 MCP `2026-07-28` 中，`inputSchema` 和 `outputSchema` 使用 JSON Schema。如果未提供 `$schema`，默认 dialect 为 2020-12。

输入 schema 必须是一个 schema 对象。即使工具不接收参数，也应该准确声明它接受什么：

```json
{
  "type": "object",
  "additionalProperties": false
}
```

这比 `{ "type": "object" }` 更严格，后者允许任意属性。

输出 schema 是可选的。一旦服务器发布了输出 schema，每个完整工具
结果就承诺返回符合要求的 `structuredContent`，包括带有
`isError: true` 的结果。错误标志只对执行结果分类，并不会免除已经
发布的输出契约。客户端应亲自验证结果，而不是盲目信任描述符。

### 结构化内容可以是任何 JSON 值

不要把 `structuredContent` 硬编码为字典。它可以是：

- 对象；
- 数组；
- 字符串；
- 数字；
- 布尔值；
- `null`。

以下工具返回数组：

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

它的成功结果是有效的：

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

为了兼容，结构化结果还应在一个文本块中包含序列化后的 JSON。该文本不是验证依据，真正的验证来源是 `structuredContent`。

### 小型验证器仍能揭示边界

本课有意只实现 JSON Schema 的一个子集，以便完全使用 Python 标准库。它检查示例工具所用到的机制：

- object、array、string、integer、number、boolean 和 null 类型；
- 必需属性；
- `additionalProperties: false`；
- 数组 item；
- enum 值；
- 最小字符串长度。

这不能替代完整的生产级验证器。真正可复用的知识在于验证发生的位置：发现之后验证描述符，执行之前验证参数，消费之前验证结构化结果。

## Content Block 承担不同成本

`content` 数组可以组合多种内容类型。

| 类型 | 适用场景 | 主要边界 |
|------|------------|---------------|
| `text` | 供人和模型阅读的摘要 | 把文本视为不可信输出 |
| `image` | 以 base64 编码的视觉证据 | 验证媒体类型和大小 |
| `audio` | 以 base64 编码的语音或录音输出 | 验证媒体类型和时长限制 |
| `resource_link` | 客户端稍后可获取的 URI | 后续读取资源时重新授权 |
| `resource` | 直接嵌入结果的数据 | 立即执行载荷与内容限制 |

资源链接不能证明该资源会出现在 `resources/list` 中。它只是本次工具调用返回的引用。客户端跟随 URI 时仍要应用自己的资源策略。

嵌入资源可以避免额外一次往返，但会增大当前响应。对于大型或独立变化的产物，使用链接；对于必须与结果原子传输的小型证据，使用嵌入资源。

本课的 `evidence_bundle` 结果包含全部五种类型。客户端接受结果前会验证每个 block。

## `x-mcp-header` 是路由元数据

`inputSchema` 中的属性可以声明 `x-mcp-header`。使用 Streamable HTTP 时，客户端会把该参数镜像到 `Mcp-Param-{name}`。

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

对于 `region: "eu-west"`，传输层可以发出：

```http
Mcp-Param-Region: eu-west
```

该 annotation 的目的，是让负载均衡器、网关或策略引擎无需解析 JSON body 就能路由。它不是存放凭据的位置。

协议对 annotation 有以下约束：

- header 名称不能为空，且须符合 HTTP field-name token 语法；
- header 名称不区分大小写时仍须唯一；
- 属性类型只能是 string、integer 或 boolean；
- 不允许 `number`；
- annotation 只能出现在 `inputSchema.properties` 的直接成员上；
- integer 值必须位于 `-9007199254740991` 到 `9007199254740991` 之间。

位置规则是语法性的，并采用失败关闭策略。应遍历整个 schema 树，
而不只是当前验证器恰好认识的属性。对于嵌套对象 `properties` 下、
`oneOf` 分支中、`items` 中、通过 `$ref` 到达的定义中，或任意输出
schema 中的 annotation，都要拒绝。解析引用不会让被引用节点变成
顶层直接属性。

本课还加入一项部署策略：拒绝镜像名称类似 `password`、`secret`、`token`、`api_key` 或 `authorization` 的描述符。官方规范建议服务器作者不要镜像敏感参数。客户端可以把这项建议升级为强制准入规则。

审计 header 名称，而不是它的值。示例代码记录 `Mcp-Param-Region`，但不会把 `eu-west` 写入审计事件。

### 构造 HTTP Header 前先编码值

只有当参数值是由 `!` 到 `~` 范围内可见 ASCII 字符组成的非空字符串，
并且看起来不像编码哨兵时，才能以纯文本传输。其余所有值都使用以下
精确形式：

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` 是对原始 UTF-8 字节执行的标准 base64。不要先 trim、
normalize 或替换值。Unicode、空字符串、空格、tab、控制字符、CR 或 LF、
前导或尾随空白，以及任何以 `=?base64?` 开头的值都必须编码。对于看似
哨兵的值再次编码，接收方才能还原字面上的原始文本，而不会误把它当成
传输语法解码。

布尔值渲染为小写 `true` 或 `false`。整数以十进制渲染，且必须位于
JavaScript 安全整数范围内。超出范围的值应直接拒绝，不能任由中间设施
将其舍入。

### 服务器检查镜像副本

生成 header 只是客户端一半的工作。在 Streamable HTTP 边界，服务器必须：

1. 查找已识别的 `Mcp-Param-*` 名称，不区分 header 名称大小写；
2. 如果存在精确的 base64 哨兵形式，则对其解码；
3. 将解码后的文本与 JSON body 中对应参数精确比较；
4. 在分发前拒绝缺失、重复、意外、格式错误或不匹配的已识别 header。

拒绝响应为 HTTP `400`，并使用 JSON-RPC 错误码 `-32020`。body 值及其
编码后的 header 形式都不应进入审计记录。只记录已识别的 header 名称
和拒绝类别。

`code/main.py` 直接建模了该边界。[第 09 课](../../09-mcp-transports/)
讲解更完整的 Streamable HTTP 验证顺序，包括方法与协议版本一致性。

## 分页 Cursor 是不透明的

MCP 列表操作使用 cursor 分页。服务器选择页面大小与 cursor 格式。客户端只需做一个判断：

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

不要这样写：

```python
if not result.get("nextCursor"):
    break
```

空字符串是有效 cursor。用 truthiness 判断会过早停止。

客户端不得解码 cursor、递增 cursor、比较它与前一个 cursor 的顺序，也不能从中推断页码。服务器可能对 cursor 签名、将其绑定到特定目录版本，或映射到私有状态。这些都是服务器的实现细节。

示例服务器刻意在第一页后返回 `""`。客户端必须在第二次请求中原样发送该值。其 trace 为：

```text
<first request with no cursor>
<second request with cursor "">
```

无效 cursor 会产生 JSON-RPC invalid params，错误码为 `-32602`。

## Completion 是授权接口

`completion/complete` 为 prompt 参数和资源模板参数提供建议。它适合交互式表单，但也可能泄露原本由普通 list 方法保护的名称。

completion 请求会指定引用以及正在补全的参数：

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

结果最多返回 100 个值，并且可以报告 `total` 与 `hasMore`。

应用与所引用 prompt 或资源相同的授权边界。示例中的 analyst 会收到 `development` 和 `staging`。只有 operator 才能收到 `production`。

生产级 completion 还需要：

- 输入验证；
- 根据调用方过滤；
- 客户端请求防抖；
- 服务器限流；
- 有界的结果数量；
- 不暴露敏感建议值的日志。

Completion 是辅助功能，不是绕过发现机制的通道。

## 两层错误

协议错误必须与工具执行错误分开。

当 MCP 请求无法被正确分发时，使用 JSON-RPC error：

- 未知工具名称；
- 请求结构格式错误；
- 缺少请求元数据；
- cursor 无效。

当调用已经到达工具，而工具报告可处理的失败时，返回带 `isError: true` 的完整工具结果：

- 报告数据源不可用；
- 日期超出支持范围；
- 业务规则拒绝所请求的操作。

模型通常可以修复工具执行错误，却无法修复违反自身输出 schema 的服务器。

如果工具声明了输出 schema，就要在该 schema 内建模可处理的失败。
示例 `route_report` 的失败结果会返回请求的区域及
`accepted: false`，同时提供供人阅读的错误文本和 `isError: true`。

## 构建它

`code/main.py` 使用 Python 标准库构建边界的两侧。

服务器实现：

- 逐请求验证 MCP 元数据；
- 带 tools 和 completions capability 的 `server/discover`；
- 确定性的 `tools/list` 分页；
- 四个工具描述符，其中一个必须被拒绝；
- 数组结构化输出；
- 当前所有工具 content block 类型；
- Streamable HTTP 一致性关卡：解码已识别的参数 header，并在不匹配时
  返回 HTTP `400` 和 JSON-RPC `-32020`；
- 经过授权且有限流的 completion。

客户端实现：

- 描述符准入；
- 对 `x-mcp-header` 位置做全树验证，并应用敏感字段策略；
- 精确的“纯可见 ASCII 或 base64 UTF-8”值编码；
- 跟随空字符串的 opaque cursor 循环；
- 参数和结果验证；
- content block 验证；
- 只含 header 名称、不含值的审计事件。

刻意设置的不安全描述符属于教学数据。它证明拒绝一个工具并不会阻止其他有效工具加载。

## 使用它

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

演示会打印获准工具、被拒绝的描述符、两次分页请求、结构化数组内容、
content block 类型、镜像 header 名称、值是否需要编码、HTTP 一致性状态，
以及按调用方过滤的 completion 值。

## 交互实验

打开 `code/main.py` 并找到 `TOOLS`。

1. 将 `tag_catalog.outputSchema.type` 从 `array` 改为 `object`。
2. 运行演示。客户端应拒绝返回的数组。
3. 恢复 schema。
4. 保持第一页的 `nextCursor` 为 `""`，然后让最后一页返回
   `nextCursor: None`，而不是省略该字段。
5. 运行测试并比较 cursor trace。
6. 向一个 string 属性添加 `x-mcp-header: "Authorization"`。
7. 确认描述符在调用前的准入阶段被拒绝。
8. 尝试让 `region` 包含 Unicode、换行符、两端空格，以及字面文本
   `=?base64?SGVsbG8=?=`。解码每个发出的 header，并证明原始值被精确保留。
9. 把 annotation 移到 `oneOf`、`items` 或 `$ref` 定义下。确认每个
   描述符都被拒绝，即使演示从未使用该分支。
10. 删除已识别的 header，或更改其解码值。确认 HTTP 边界返回状态
    `400` 和 JSON-RPC 错误码 `-32020`。

重点不在于背下某种 JSON 结构，而在于观察每道关卡如何在自己负责的边界上失败。

## 实践实验

使用 `search_evidence` 工具扩展契约实验。

要求：

1. 输入 schema 接受 `query`、`limit` 和一个安全的 `region` 路由字段。
2. 输出 schema 是对象数组，每个对象包含 `uri`、`title` 和 `score`。
3. 每个结果项都包含兼容文本和一个资源链接。
4. 参数拒绝未知属性。
5. 通过应用验证限制 `limit`。
6. 无权访问某个 URI 的调用方，在 completion 或工具输出中都不能看到它。
7. 测试覆盖不合规的 score、无效 header annotation 和两页列表。
8. header 值测试覆盖可见 ASCII、Unicode、控制字符、空白、形似哨兵的文本，
   以及 JavaScript 安全整数的两个边界值。
9. HTTP fixture 接受大小写不敏感的 header 名称，但对缺失或不匹配的
   已识别值返回状态 `400` 和错误码 `-32020`。

## 交付产物

`outputs/skill-mcp-contract-reviewer.md` 是一项扁平、可复用的审查 Skill。向它提供工具描述符、结果样例、分页行为与 completion 策略，它会返回准入决策、结果验证方案、header 策略和具体的失败测试。

## 验证

满足以下陈述时，本课即告完成：

- `tools/list` 在重复调用时返回相同的逻辑顺序。
- 当 `nextCursor` 为 `""` 时，客户端会发起第二次请求。
- 不安全的敏感 header 描述符被排除，其他工具仍然可用。
- 数组能通过其 array 输出 schema。
- 对象无法通过同一份 array schema。
- 错误结果不能省略或违反已发布的输出 schema。
- text、image、audio、resource link 与 embedded resource block 均能通过验证。
- header 审计事件只包含名称，不包含值。
- 纯可见 ASCII 保持明文；Unicode、控制字符、带边缘空白、空值以及
  形似哨兵的值通过精确 base64 UTF-8 编码后仍能往返还原。
- 超出 JavaScript 安全范围的镜像整数会被拒绝。
- 位于 `oneOf`、`items`、嵌套对象、`$ref` 定义或输出 schema 中的
  annotation 会在准入阶段被拒绝。
- 只有当解码值与 body 完全一致时，大小写不敏感的已识别 header 名称
  才能通过；缺失或不匹配的副本会产生 HTTP `400` 和 JSON-RPC `-32020`。
- Analyst 的 completion 永远不会返回 `production`。
- 工具失败使用 `isError: true`；格式错误的协议调用使用 JSON-RPC `error`。

## 生产故障模式

| 故障 | 学习者看到的现象 | 正确响应 |
|---------|-----------------------|------------------|
| 客户端假定输出是对象 | 有效数组失败或被静默包装 | 根据已发布 schema 验证，不要限制为 object 类型 |
| 把空 cursor 当成 false | 最后几页消失 | 只要 `nextCursor` 存在且不为 null 就继续 |
| 镜像敏感值 | secret 出现在代理、WAF 或 trace 数据中 | 拒绝描述符，把 secret 留在受保护请求数据中 |
| 镜像原始 Unicode 或空白 | 网关与源站理解不一致，或值被规范化 | 使用精确的 base64 UTF-8 哨兵编码，解码后再比较 |
| annotation 藏在 schema 分支中 | 客户端在准入时漏掉路由元数据 | 遍历整个 schema 树，并且只允许顶层直接属性 |
| 镜像大整数 | JavaScript 中间设施舍入路由值 | 拒绝 JavaScript 安全整数范围之外的值 |
| Header 与 body 不一致 | 网关路由到一个目标，源站却执行另一个 | 分发前拒绝，返回 HTTP `400` 和 JSON-RPC `-32020` |
| 忽略输出 schema | 下游代码消费损坏的结构 | 在模型或应用使用前验证 |
| 自动信任资源链接 | 调用方跟随未授权 URI | 每次读取资源时重新授权 |
| Completion 共享全局建议 | 隐藏的租户名称泄漏 | 按调用方、引用和授权过滤 |
| 把工具 annotation 当作策略 | 破坏性操作绕过确认 | 在 annotation 之外执行授权与审批 |
| 一个格式错误的工具破坏发现 | 整台服务器不可用 | 拒绝错误描述符，并独立准入有效工具 |

## 与综合项目的联系

阶段 13 综合项目需要一个可以合并多台服务器工具的网关。本课提供其准入核心。

使用本课产物评判以下四类综合项目证据：

- 确定且完整的分页发现；
- 暴露给模型前的描述符验证；
- 已验证的结构化输出与有界 content block；
- 保持授权边界的 completion 与路由元数据。

不能只因为一次 `tools/call` 成功，就声称网关兼容。应捕获描述符、页面 trace、获准工具集、被拒工具集和一份已验证结果。

## 关键术语

| 术语 | 含义 |
|------|---------|
| `inputSchema` | 定义工具可接受参数的 JSON Schema 对象 |
| `outputSchema` | 定义 `structuredContent` 的可选 JSON Schema |
| `structuredContent` | 工具结果产生的任意 JSON 值 |
| Content block | 有类型的文本、图像、音频、资源链接或嵌入资源 |
| `x-mcp-header` | 将 primitive 参数镜像到 Streamable HTTP 元数据的 schema annotation |
| Opaque cursor | 由服务器签发、客户端不解释其值的分页 token |
| Completion reference | 正在补全其参数的 prompt 名称或资源 URI/template |
| Admission | 客户端决定暴露还是拒绝已发现描述符 |

## 延伸阅读

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
