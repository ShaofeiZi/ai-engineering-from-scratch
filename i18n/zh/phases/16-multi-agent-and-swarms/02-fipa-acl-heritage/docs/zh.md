# FIPA-ACL 与言语行为的传承

> 在 MCP 和 A2A 之前，已有 FIPA-ACL。2000 年，IEEE 智能物理代理基金会批准了一种智能体通信语言，其中包含二十种施为词、两种内容语言，以及合同网、订阅／通知、条件请求等一组交互协议。它逐渐淡出产业界，是因为本体带来的开销对 Web 而言过于沉重；但随着 LLM 推动多智能体系统复兴，人们正在悄然重新实现相同思想，只是舍弃了形式语义：JSON 合约替代施为词，自然语言替代本体。本课将认真研读 FIPA-ACL，帮助你分辨 2026 年的协议决策中哪些是旧概念的再造、哪些确属创新，以及当前浪潮将再次遭遇哪些早在 21 世纪初就已解决的问题。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 阶段 16 · 01（为何需要多智能体）
**Time:** 约 60 分钟

## 问题

2026 年的智能体协议版图十分拥挤：MCP 面向工具，A2A 面向智能体，ACP 面向企业审计，ANP 面向去中心化信任，NLIP 面向自然语言内容，此外还有 CA-MCP 和二十多项研究提案。每份规范都宣称自己具有奠基意义。

坦率地看，其中大多数都在重新发现一棵非常具体、已有二十年历史的决策树。Austin（1962）和 Searle（1969）的言语行为理论提出“话语就是行动”。KQML（1993）将这一思想转化为线上通信协议。FIPA-ACL（2000 年批准）完成了参考性的标准化工作：二十种施为词、SL0 / SL1 内容语言，以及合同网和订阅—通知等交互协议。JADE 与 JACK 是 Java 参考平台。到 2010 年前后，这项工作逐渐式微，因为本体开销过于沉重，而 Web 技术栈赢得了竞争。

当你看到 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，看到的其实是以 JSON 为原生格式、语义更宽松的 FIPA 决策翻版。了解这段传承能让你看清两件事：哪些新“创新”实际上是重新发明，以及新规范将重新遭遇哪些旧有失败模式。

## 核心概念

### 一段话理解言语行为

Austin 注意到，有些句子不是在描述世界，而是在改变世界。“我承诺。”“我请求。”“我宣布。”他将其称为施为话语。Searle 将它形式化为五类：断言类、指令类、承诺类、表达类、宣告类。KQML（Finin 等，1993）把这一理论落到软件智能体中：一条消息由施为词（动作）和内容（动作所针对的对象）组成。FIPA-ACL 弥补了 KQML 的缺口，并围绕二十种施为词完成标准化。

### 二十种 FIPA 施为词（部分列表）

| 施为词 | 意图 |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你执行 X” |
| `query-if` | “P 是否为真？” |
| `query-ref` | “X 的值是什么？” |
| `propose` | “我提议我们执行 X” |
| `accept-proposal` | “我接受该提议” |
| `reject-proposal` | “我拒绝该提议” |
| `agree` | “我同意执行 X” |
| `refuse` | “我拒绝执行 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “无法解析你的消息” |
| `cfp` | “针对 X 征集提案” |
| `subscribe` | “X 变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试了 X，但失败了” |

完整列表见 `fipa00037.pdf`（FIPA ACL Message Structure）。重点不在于背诵，而在于其中每一种施为词，都对应着 LLM 协议最终会重新加入的一项原语。

### 规范的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封，一个字段（`content`）承载负载。其余字段恰恰就是你每次在 JSON 协议上添加重试、会话串联和本体时都会重新发明的东西。

### 两个旧平台

**JADE**（Java Agent DEvelopment framework，1999–2020 年代）是使用最广泛的 FIPA 兼容运行时。智能体扩展一个基类、交换 ACL 消息、运行在容器中，并使用“行为”进行协调。其交互协议库内置合同网、订阅—通知、条件请求和提议—接受。

**JACK**（Agent Oriented Software，商业软件）强调建立在 FIPA 消息之上的 BDI（Belief–Desire–Intention，信念—愿望—意图）推理。它更形式化，但采用率较低。

随着 Web 技术栈接管多智能体用例，两者都逐渐衰落。MCP 和 A2A 则是 2026 年的运行时“容器”。

### FIPA 为何式微

- **本体开销。** FIPA 要求使用共享本体来解析 `content`。达成本体共识往往需要数年的标准制定；Web 则直接采用 HTTP + JSON。
- **无人采用的形式语义。** SL（Semantic Language）提供严格的真值条件，但大多数生产系统使用自由格式内容，忽略了这些形式化规则。
- **工具链锁定。** JADE 只支持 Java，JACK 则是商业软件。多语言团队绕开了两者。
- **互联网赢得了技术栈之争。** REST、随后出现的 JSON-RPC 与 gRPC 取代了 ACL 的传输方式。

### LLM 复兴的是轻量版 FIPA

比较 FIPA `request` 与 MCP `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

两者信封相同，语法不同。它们都承载发送者、接收者、意图、负载与关联 ID。谁也不是相对于另一方的革命；它们只是针对同一设计作出了不同取舍。

Liu 等人在 2025 年的综述《A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP》（arXiv:2505.02279）明确说明了这条传承关系：MCP 对应工具使用言语行为，A2A 对应智能体间言语行为，ACP 对应审计追踪言语行为，ANP 对应去中心化身份扩展。新规范是使用 JSON 语法、语义更宽松的 ACL 后裔。

### 直白地说明取舍

**FIPA 提供、而现代规范舍弃的能力：**

- 形式语义——可以证明 `inform` 蕴含发送者相信其内容。
- 规范的施为词目录——无须再次争论“我们是否需要 `cancel`？”
- 数十年的交互协议模式——合同网、订阅—通知、提议—接受——以及它们已知的正确性属性。

**现代规范提供、而 FIPA 未能提供的能力：**

- 与所有现代工具兼容的 JSON 原生负载。
- LLM 无须手写本体即可解释的自然语言内容。
- Web 技术栈传输（HTTP、SSE、WebSocket）。
- 通过实时 MCP `server/discover` 和 A2A Agent Cards 发现能力。

用更宽松的意图语义换取更容易的实现，这就是确切的取舍。

### 值得移植的交互协议

FIPA 提供了约 15 种交互协议，其中三种值得带入 LLM 多智能体系统：

1. **合同网协议（CNP）。** 管理方发出 `cfp`（call for proposals，征集提案）；竞标方以 `propose` 响应；管理方接受或拒绝。这是经典的任务市场模式（阶段 16 · 16 协商）。
2. **订阅／通知。** 订阅方发送 `subscribe`；每当主题变化，发布方就发送 `inform`。2026 年的每个事件总线都在使用这种模式。
3. **条件请求。** “当条件 Y 成立时执行 X。”它是带前置条件的延迟操作。2026 年的对应机制是持久化工作流引擎中的延迟任务（阶段 16 · 22 生产扩展）。

每一种都可以干净地映射到现代消息队列、HTTP + 轮询或 SSE 流式传输。

### 舍弃本体后会出什么问题

没有共享本体，智能体只能从自然语言内容中推断含义。已有记录的 2026 年失败模式是**语义漂移**：两个智能体用同一个词（`"customer"`）表示略有差异的概念，接收方智能体按照错误解释采取行动，而模式验证器无法发现问题。FIPA 的本体要求会在解析时拒绝这条消息。

不采用完整本体时，可以使用以下缓解措施：

- 在 `content` 上应用 JSON Schema——在传输层拒绝结构错误。
- 使用类型化工件（A2A）——拒绝模态不匹配。
- 在信封中显式加入施为词——即使内容是自然语言，意图也不会产生歧义。

### 2026 年规范与言语行为传承的映射

| 现代规范 | FIPA 对应概念 | 保留的内容 | 舍弃的内容 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式语义、本体 |
| MCP `resources/read` | `query-ref` | 显式意图、关联 ID | 形式语义 |
| A2A Task 生命周期 | 合同网 + 条件请求 | 异步生命周期、状态转换 | 形式完备性保证 |
| A2A 流式事件 | 订阅／通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板系统（Hayes-Roth，1985） | 多写入者共享记忆 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | schema |

从上到下阅读这张表，会发现一个共同模式：保留结构原语，舍弃形式化规则，再让 LLM 掩盖其中的歧义。

```figure
sw-contract-net
```

## 动手构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 转换器。它对规范 ACL 信封进行编码与解码，并展示每种 MCP / A2A 消息结构如何归约为相同的七个字段。演示将：

- 把五种 MCP 风格与 A2A 风格消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 使用 `cfp`、`propose`、`accept-proposal` 与 `reject-proposal`，在一名管理方和三名竞标方之间运行玩具版合同网协商。

运行：

```
python3 code/main.py
```

输出以并排追踪的形式展示每条现代消息的 2026 JSON 形式和 FIPA-ACL 形式，再演示一次合同网投标的往返转换。同样的协议原语在往返后得以保留，只有语法不同。

## 实际使用

`outputs/skill-fipa-mapper.md` 是一项读取任意智能体协议规范并生成 FIPA-ACL 映射的技能。在采用新协议前使用它，回答：“这真的属于新概念，还是仅仅使用 JSON 语法的 `inform`？”

## 交付成果

不要让 FIPA-ACL 复活，而要重新采用它的检查清单：

- 每条消息的意图原语（施为词）是什么？
- 请求—响应与取消是否带有关联 ID？
- 是否存在显式内容语言（JSON-RPC、纯文本、结构化类型工件）？
- 交互协议是否是一等概念，还是你正在从头重新实现合同网？
- 两个智能体对内容含义存在分歧（语义漂移）时会怎样？

在将任何新协议投入生产前，记录这五个问题的答案。

## 练习

1. 运行 `code/main.py`，观察往返编码。识别 `tools/call`、`resources/read` 和 A2A 任务创建分别对应哪个 FIPA 施为词。
2. 扩展合同网演示，添加一个允许 Manager 在投标过程中撤销任务的 `cancel` 施为词。`cancel` 解决了哪个仅靠重试无法解决的失败场景？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。选择一个本课未涉及的施为词，并描述其现代 JSON-RPC 对应形式。
4. 阅读 Liu 等，arXiv:2505.02279。分别列出 MCP、A2A、ACP、ANP 保留和舍弃了哪些 FIPA 施为词族。
5. 为你自己的系统中 `content` 字段设计一个最小 JSON Schema，该字段属于 `request` 施为词。相比纯自然语言，该 schema 带来了什么，又付出了什么代价？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 言语行为 | “会产生行动的话语” | Austin / Searle：将话语视为行动，是 ACL 的理论源头。 |
| FIPA | “那个旧 XML 东西” | IEEE Foundation for Intelligent Physical Agents，2000 年完成 ACL 标准化。 |
| ACL | “Agent Communication Language” | FIPA 的信封格式：施为词 + 内容 + 元数据。 |
| 施为词 | “动词” | 消息的意图类别，例如 `inform`、`request`、`propose`、`cfp`。 |
| KQML | “FIPA 的前身” | Knowledge Query and Manipulation Language（1993），更简单、范围更窄。 |
| 本体 | “共享词汇表” | 对内容语言所讨论概念的形式化定义。 |
| SL0 / SL1 | “FIPA 内容语言” | Semantic Language 0 级与 1 级——形式内容语言家族。 |
| 合同网 | “任务市场” | Manager 发出 cfp，Bidder 提案，Manager 接受；经典交互协议。 |
| 交互协议 | “消息模式” | 具有已知正确性的施为词序列，例如条件请求、订阅—通知等。 |

## 延伸阅读

- [Liu 等——智能体互操作协议综述：MCP、ACP、A2A、ANP](https://arxiv.org/html/2505.02279v1)——将现代规范与 FIPA 传承联系起来的权威 2025 年综述
- [FIPA ACL Message Structure Specification（fipa00037）](http://www.fipa.org/specs/fipa00037/)——2000 年批准的信封格式
- [FIPA Communicative Act Library Specification（fipa00037）](http://www.fipa.org/specs/fipa00037/)——完整施为词目录
- [MCP 2026-07-28 规范](https://modelcontextprotocol.io/specification/2026-07-28)——与 `request`／`query-ref` 对应的当前无状态工具使用规范
- [A2A 规范](https://a2a-protocol.org/latest/specification/)——合同网与订阅—通知的现代智能体间对应规范
