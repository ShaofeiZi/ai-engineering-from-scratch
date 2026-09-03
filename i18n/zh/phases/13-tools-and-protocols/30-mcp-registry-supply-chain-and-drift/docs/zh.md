# MCP Registry 供应链：准入、漂移与回滚

> Registry 条目只能告诉你发布方声明了什么。生产准入需要证明你获取了什么、观察到了什么、批准了什么，以及能够安全恢复什么。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 13 阶段 · 第 17 课（网关与注册表）、第 13 阶段 · 第 18 课（生产环境身份验证）
**Time:** 约 90 分钟

## 学习目标

- 区分 Registry 发布、软件包 provenance、运行时发现与本地审批。
- 验证 MCP 服务器 namespace，而不信任它自己记录中的名称。
- 固定不可变的发布证据、执行来源证据、provenance 证据和实时描述符证据。
- 在准入后检测 Registry 状态变化与运行时漂移。
- 将路由回滚到先前已获准的版本，而不改写历史。
- 维护一份可检测篡改的准入 ledger，解释每次决策。

## 问题

你在 Registry 中找到 `com.example/inventory`。它的描述看起来正确，软件包确实存在，服务器也会响应 `server/discover`。

这不是一个事实，而是一条来自不同权威方的事实链：

1. 一名通过 namespace 身份验证的发布方提交了一条记录。
2. 软件包 Registry 提供了具有特定身份和 digest 的 artifact。
3. 一个正在运行的端点报告了协议版本、capability、工具与诊断性服务器信息。
4. 你的组织决定允许这个精确组合。

如果把这些事实压缩成“它在 Registry 里，所以可以信任”，就会产生供应链盲区。有效发布仍可能被弃用。如果不固定 digest，软件包 tag 可能指向意外 artifact。服务器可能在审查后新增破坏性工具。回滚还可能悄悄选择一个从未获准的版本。

解决方法是在每个边界上保存证据，并由准入控制器作出决定。

## Registry 是索引，不是你的审批系统

官方 MCP Registry 存储服务器元数据。其 `server.json` 记录指明服务器版本，并声明一个或多个软件包或远程端点。发布规则还会执行 namespace 身份验证、软件包所有权检查、受限 Registry 规则，并限制发布方元数据的位置。

这些控制回答的是发布问题。生产策略仍要回答部署问题：

| 边界 | 问题 | 证据所有者 |
|---|---|---|
| Namespace | 发布方是否有权使用该名称？ | Registry 身份验证，加上你提供的已验证 namespace 输入 |
| 记录 | 发布方为该版本声明了什么？ | 不可变的 `server.json` digest |
| 执行来源 | 哪个软件包或远程端点会真正执行？ | 声明的来源字段、已验证所有权结果、传输方式与可信 digest |
| 运行时 | 端点此刻暴露什么？ | `server/discover` 与工具描述符 |
| 准入 | 策略是否批准了这个精确集合？ | 本地 pin 与 ledger 条目 |
| 运维 | 它是否仍然安全，什么可以替代它？ | 漂移检查、状态同步、健康状态与回滚路由 |

Registry schema 版本与 MCP 协议版本相互独立。一条记录可以使用已发布的 `2025-12-11` 服务器 schema，而在线服务器支持 MCP `2026-07-28`。绝不能根据其中一个推断另一个。

```figure
mcp-registry-admission
```

## 一次准入决策中的七项控制

### 1. Namespace 验证

官方 Registry 名称使用经过身份验证的 namespace。已验证的域名可以映射为反向域名前缀。例如，对 `example.com` 的控制权可以确立 `com.example/*`。

不要接受简单的字符串前缀检查：

```python
server_name.startswith("com.example")
```

因为它也会接受 `com.exampleevil/tool`。应在 `/` 处分割名称，要求 slug 非空，并精确比较 namespace 段。更重要的是，要把经过验证的 namespace 从身份验证结果传入准入流程。不要从不可信记录中推导信任。

GitHub 支持的 namespace 与域名支持的 namespace 使用不同的身份验证路径。无论哪一种，都应规范化为同一项准入输入：精确的已验证 namespace 字符串。

### 2. Provenance 关联

对于软件包记录，声明与实际获取的 artifact 必须通过显式字段关联：

- 软件包 Registry 类型
- 软件包标识符
- 软件包版本
- 已验证的所有权结果
- 下载 artifact 的 digest

还要验证声明的软件包传输方式。只有远程端点、没有软件包的记录同样有效，不能因为缺少软件包而拒绝。对于远程来源，应将声明的 URL 与传输类型，同独立验证的端点所有权以及可信连接或部署证据的 digest 关联起来。

本课代码同时支持两种来源，并将选定来源与 Registry 来源、服务器名称、Registry 版本、记录 digest 和证据 digest 一起进行哈希。得到的 provenance digest 是指向完整证据集的紧凑指针，不能替代证据本身的留存。

绝不能只接受待验证 artifact 自己提供的 digest。应在可信获取边界计算 digest，或者由软件包服务提供，并验证该服务的验证结果。

### 3. 固定决策，而不只是版本

Registry 版本是唯一的发布标识符。已发布元数据不可变；记录有变化就必须发布新版本。Registry 建议使用 semantic versioning，但不强制要求，也不接受版本范围。

因此，`^1.4` 不是准入 pin，“latest”也不是。一份有用的 pin 包含：

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

固定多层证据，可以判断究竟是哪一处边界发生变化。同一 Registry 版本下 record digest 变化，属于 Registry 完整性故障。同一软件包坐标或远程部署下 source digest 变化，属于执行来源完整性故障。toolset digest 变化，则是运行时漂移。

### 4. 实时漂移检测

准入流程应观察真正会接收流量的服务器。调用 `server/discover`，通过可信路径列出或以其他方式取得已暴露的工具描述符，并验证：

- `2026-07-28` 位于 `supportedVersions` 中
- 本地要求的所有 capability 均存在
- 每个工具描述符都具备必需的身份与 schema 表面
- 在后续检查中，规范化描述符 digest 与已获准 pin 一致

可选结果 `_meta["io.modelcontextprotocol/serverInfo"]` 的值属于服务器自报的展示、日志与调试上下文。可以把它记录为诊断证据，但绝不能用来确立 namespace、软件包所有权、端点所有权、准入或任何其他安全决策。直接的 `serverInfo` 别名如果位于 `_meta` 之外，就并非契约字段，不应提升为诊断证据。

只规范化顺序没有语义的字段。示例在哈希前按稳定名称对工具列表排序，因此无害的列表顺序变化不会触发漂移。它不会丢弃任何描述符字段。新增工具、修改 schema、修改描述或新增 annotation，都会改变 pin。

示例会把格式错误的描述符以及任何描述符 digest 变化都视为漂移：隔离 pin、删除其 active route，并禁止把该版本用作回滚目标。生产策略即使要允许编辑性改动，也应要求重新审查，因为描述会影响模型选择工具。“外观性”元数据也可能改变智能体行为。

### 5. Registry 状态是实时状态

Registry API 会在每条服务器记录旁附带 response-level `_meta` 对象。由 Registry 管理的字段位于 `_meta["io.modelcontextprotocol.registry/official"]` 下。应把响应的 `_meta` 对象传入准入流程，并读取 `_meta["io.modelcontextprotocol.registry/official"].status`。直接的 `_meta.status` 值不符合官方线路结构。不要把响应元数据与发布记录自身的 `_meta` 混淆。状态可以是：

- `active`：默认返回，并且可按本地策略准入
- `deprecated`：仍可发现并附带警告，但不再适合作为安全的自动选择
- `deleted`：默认隐藏，但历史记录仍可通过 deleted 或增量视图获取

准入后仍要同步状态。如果 active 版本变为 deprecated 或 deleted，就隔离其 pin，并停止把新工作路由给它。保留证据。记录从默认列表中删除，不代表你有权清除自己的审计轨迹。

发布方提供的自定义元数据，在发布记录中只能放在 `_meta.io.modelcontextprotocol.registry/publisher-provided` 下。Registry 管理的响应元数据与之独立。绝不能允许发布方自行设置官方状态。

### 6. 回滚意味着恢复路由

回滚不会编辑不可变发布。回滚应选择一个先前已经获准、当前仍然合格的 pin，并修改 active route。

安全目标必须满足：

1. 拥有一条完整的准入记录。
2. 按你的策略，其 Registry 状态仍为 active。
3. 没有因为运行时或安全证据而被隔离。
4. 仍然解析到固定的软件包和实时描述符集合。
5. 通过当前健康检查。

示例重点实现前三项。真实 reconciler 应在激活前重新获取软件包并重新检查在线端点。

### 7. 追加准入 Ledger

准入数据库说明什么处于 active。ledger 则解释为什么。

每条示例记录都包含序号、时间、事件、服务器、版本、结果、原因、证据、上一条记录的 hash 和自身 hash。修改早期记录的结果，会破坏该条记录及其后每一条链接的验证。

这能检测篡改，却不会神奇地阻止篡改。应定期把 ledger head 锚定到另一个信任域，例如签名发布元数据或 write-once 存储。限制谁可以追加。不要把授权 token、软件包凭据、工具参数或私有端点数据写入证据。

## 构建它

可运行的控制器位于 `code/main.py`，仅使用 Python 标准库。

先运行有限演示：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

演示执行五项操作：

1. 以匹配的 namespace、软件包 provenance、协议、capability 和工具准入 `1.0.0`。
2. 准入 `1.1.0`，并将其设为 active。
3. 观察到运行时出现意外的删除工具。
4. 观察到 `1.1.0` 的 Registry 状态变为 `deprecated`。
5. 将路由恢复到仍处于已准入状态的 `1.0.0` pin。

预期结构：

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

按以下顺序阅读实现：

1. `namespace_for_domain()` 与 `namespace_matches()` 建立精确的命名权限。
2. `digest()` 与 `normalized_tools()` 生成确定性证据。
3. `RegistryAdmissionController.admit()` 将发布、provenance、运行时和策略关联起来。
4. `check_live()` 将新的观察结果与 pin 比较。
5. `observe_registry_status()` 隔离 Registry 状态发生变化的版本。
6. `rollback()` 只激活先前已获准且当前合格的目标。
7. `AdmissionLedger.verify()` 检测历史记录是否被修改。

## 使用它

把控制器放在发现与路由之间：

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

这些任务应使用不同身份。Registry sync worker 只需要读取元数据。artifact verifier 需要软件包获取权限。route reconciler 需要激活已批准 pin 的权限。没有任何一个角色需要全部凭据。

应明确表示 rollout 状态。“Approved”表示证据通过策略。“Active”表示路由当前选择它。“Quarantined”表示它不能接收新工作。“Superseded”表示另一个已获准版本处于 active。不要用一个 Boolean 编码全部四种含义。

在服务器暴露到 `tools/list` 之前执行准入。否则，客户端可能在发布与策略评估之间的空档发现工具。

## 交互实验

你将逐个观察边界失败。

### 实验 A：namespace 冲突

从代码目录打开 Python shell：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

然后运行：

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

第一个结果是 `True`；第二个是 `False`。在本地把精确比较替换为 `startswith`，观察第二个名称为什么会越过边界。继续前请恢复精确比较。

### 实验 B：描述符漂移

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

检查 reasons 和 route 状态。软件包与 Registry 记录都没有变化，但运行时工具表面发生了变化，因此控制器隔离并停用了该 pin。这正说明供应链控制必须延续到安装之后。

### 实验 C：状态与回滚

准入 `1.1.0`，将它标记为 deprecated，然后分别尝试两个回滚目标：

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

被隔离的目标会遭到拒绝。较早的 active pin 会被接受。ledger 仍然有效。

## 实践实验

为控制器增加双人审批关卡。

要求：

- 把审批保存为签名证据引用，而不是 pin 中可变的姓名。
- 如果 toolset 中含有带 `destructiveHint: true` 的工具，要求两个不同 reviewer 身份批准。
- 拒绝重复 reviewer 身份。
- 审批不完整时，在 ledger 中保留原始准入尝试。
- 为零个、一个、重复以及两个不同审批添加测试。
- 不记录签名、凭据或完整私有工具参数。

成功标准是：在两个身份都批准完全相同的 record、package 和 toolset digest 之前，破坏性工具不能变为 active。

## 交付产物

本课交付 `outputs/skill-mcp-registry-admission.md`。审查新 Registry 版本或调查漂移时，可把它作为扁平、可复用的 runbook。它定义输入、拒绝规则、证据包、状态核对和回滚证明，不依赖示例中的类名。

## 验证

运行演示和确定性测试套件：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

验证应证明：

- 精确 namespace 边界会拒绝相似前缀
- 只有官方 namespaced Registry 状态才能让版本合格
- 未验证或不匹配的软件包及远程证据会被拒绝
- 发布方元数据无法冒充 Registry 管理的元数据
- 工具顺序得到规范化，同时不隐藏描述符变化
- 格式错误的软件包与工具结构会安全拒绝
- `serverInfo` 始终只是诊断信息，绝不提供准入权限
- 描述符漂移会隔离、停用 pin，并禁止回滚到该 pin
- 状态变化会隔离 active pin
- 回滚不能选择被隔离或未知的版本
- ledger 篡改能够被检测

## 生产故障模式

| 故障 | 发生原因 | 必需响应 |
|---|---|---|
| 名称看似有效，但 namespace 从未验证 | 策略信任了记录文本 | 拒绝，直到可信 namespace 验证器提供精确前缀 |
| 同一软件包坐标返回新字节 | 可变上游或分发链遭到入侵 | 停止激活、保留两个 digest，并调查获取边界 |
| “Latest”未经审查发生变化 | 浮动选择逃逸 pin | 只解析已获准的精确版本和 digest |
| 审批后出现新工具 | 运行时漂移或部署发生变化 | 隔离路由，并捕获新的描述符观察结果 |
| 已弃用版本仍处于 active | 状态同步缺失或延迟 | 定时核对状态，并在激活前核对 |
| 已删除记录从默认同步中消失 | 客户端只请求了 active 记录 | 使用增量或感知 deleted 的核对方式，并保留本地历史 |
| 回滚目标从未准入 | 路由控制与审批状态脱节 | 拒绝回滚，并为目标执行新的准入流程 |
| 攻击者重写全部条目后，ledger 在本地仍能验证 | hash chain 没有外部锚点 | 把签名 ledger head 发布到独立信任域 |
| 证据包含 bearer token 或工具参数 | 日志复制了完整请求 | 采集时脱敏，只保存最小证明 |

## 运维规则

发布回答“这个身份能否发布这个名称？”准入回答“我们是否会执行这个精确 artifact，并暴露这些精确行为？”必须分离两项决策，固定每一处关联，并让回滚依据证据而不是记忆做选择。

## 延伸阅读

- [官方 Registry server.json 要求](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [官方 Registry OpenAPI 契约](https://registry.modelcontextprotocol.io/openapi.yaml)
- [MCP 2026-07-28 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
