---
name: structured-output-designer
description: 为自由文本抽取目标设计严格模式兼容的 JSON Schema 及 Pydantic 模型，并内嵌类型化的拒绝与重试处理存根。
version: 1.0.0
phase: 13
lesson: 04
tags: [structured-output, json-schema, pydantic, strict-mode, extraction]
---

给定一个自由文本抽取目标（发票、简历、支持工单、研究摘要），产出一份生产级抽取契约：JSON Schema 2020-12、Pydantic 模型、拒绝处理器和重试策略。

产出：

1. JSON Schema 2020-12。每个属性都必须有类型。`required` 列出所有属性。每个对象都设置 `additionalProperties: false`。对封闭值集使用枚举。不得使用 `$ref`。不得使用有歧义的 `oneOf` / `anyOf`。需通过 OpenAI 严格模式要求的校验。
2. Pydantic v2 BaseModel。作为 schema 的镜像，使用 Python 类型。`model_json_schema()` 生成的 schema 必须与 (1) 等价。
3. 拒绝处理器。类型化的 `Refusal(reason: str, category: str)` 结果。列出类别：`safety`、`input_mismatch`、`insufficient_info`。
4. 重试策略。三种重试形式：(a) 注入校验错误并重试一次（严格模式之外）；(b) 接受拒绝作为最终结果（严格模式）；(c) 在反复拒绝时升级到更强的模型。
5. 测试向量。十个输入，覆盖正常路径、对抗性字段、部分输入和触发拒绝的用例。每个都附带预期结果。

硬性拒绝：
- 任何含有未类型化字段的 schema。会同时导致严格模式和校验器失败。
- 任何缺少 `additionalProperties: false` 的 schema。会导致幻觉泄漏。
- 任何在没有判别字段的情况下使用 `oneOf` 的 schema。会导致解码歧义。
- 任何未做 JSON Schema 往返校验的 Pydantic 模型。

拒绝规则：
- 如果目标领域包含个人身份数据且没有文档化的用途，则拒绝并路由到 Phase 18（伦理）进行合法依据论证。
- 如果用户要求的 schema 无法用 JSON Schema 2020-12 表达（例如任意递归图），则拒绝并提议最接近的可表达松弛方案。
- 如果抽取目标是“从任意内容中抽取结构化数据”，则拒绝并要求指定具体领域。

输出：一份单页契约，包含 schema JSON、Pydantic 类、拒绝与重试策略以及十个测试向量。最后以一条关于应优先适配哪个提供商及其原因的说明收尾。
