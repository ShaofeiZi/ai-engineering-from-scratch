---
name: fipa-mapper
description: 将任意 2026 年智能体协议规范（MCP、A2A、ACP、ANP、CA-MCP、NLIP 或新协议）映射到 FIPA-ACL 的言语行为和交互协议，以判断哪些是真正的创新、哪些只是重造轮子。
version: 1.0.0
phase: 16
lesson: 02
tags: [multi-agent, protocols, FIPA, speech-acts, interoperability]
---

给定一个新的智能体协议规范，产出 FIPA-ACL 映射，使读者能够辨别哪些部分是重造轮子、哪些是真正的新结构。

产出内容：

1. **信封映射。** 对于规范定义的每种消息类型，命名最接近的 FIPA 言语行为（`inform`、`request`、`query-if`、`query-ref`、`propose`、`accept-proposal`、`reject-proposal`、`cfp`、`subscribe`、`cancel`、`failure`、`not-understood`，或其他约 20 种之一）。如果没有言语行为匹配，精确描述差距所在。
2. **关联模型。** 该规范如何将请求与回复关联、取消操作与原始请求关联、流式事件与 subscribe 关联？与 FIPA 的 `:conversation-id` 和 `:reply-with` 字段进行比较。
3. **内容语言立场。** 该规范是强制要求内容模式（类型化产物、JSON-Schema）、接受自然语言，还是留空不限定？与 FIPA 的 SL0/SL1 和本体字段进行比较。
4. **交互协议库。** 哪些 FIPA 交互协议可以在该规范之上实现：contract-net、subscribe-notify、request-when、propose-accept？命名实现每种协议所需的消息。
5. **发现模型。** 智能体如何找到对等方及其能力（MCP 的 `listTools`、A2A 的 Agent Card、ANP 的 DID + meta-protocol）？与 FIPA 的目录促进器（directory facilitator）和黄页服务进行比较。
6. **重造轮子 vs 创新。** 产出一个包含三列的简表：[FIPA 概念，现代规范等价物，变化之处]。将每行标记为 [reinvention] 或 [novel-structure]。仅当规范引入了 FIPA 所不具备的原语时，该行才标记为 "novel-structure"——去中心化身份、类型化多模态产物和 LLM 可解释内容是常见的候选。

硬性否决条件：

- 任何声称某规范是 "革命性的" 却未展示 FIPA 所不具备的原语的映射。言语行为理论 + 本体开销是失败模式，而非原语本身。
- 忽略发现层的框架比较。没有发现的规范是不完整的，而非创新的。
- 诸如 "协议 X 取代了 FIPA" 之类的说法，却不解释当两个智能体对内容含义有分歧时会发生什么（语义漂移）。

拒绝规则：

- 如果规范处于标准化前阶段（草案发布不到 6 个月，无公开实现），声明该映射是临时性的，并标记最可能发生变化的三个地方。
- 如果规范是闭源或仅限企业使用（某些 ACP 变体），映射已公开文档的部分并指出缺失之处。
- 如果用户仅提供博客文章（无规范文档），在映射之前索要规范文档。

输出：一页简报。以一句话摘要开头（"协议 X 是 FIPA `request`/`subscribe` 加上 JSON 语法和基于 DID 的发现层。"），然后是上述六个部分，最后以一段总结收尾，回答："该规范会重新发现哪个旧的 FIPA 失败模式？"
