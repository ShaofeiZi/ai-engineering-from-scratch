---
name: structured-output-picker
description: 选择结构化输出方案、Schema 设计与验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定一个用例（提供商、延迟预算、Schema 复杂度、容错能力），输出：

1. 机制。原生厂商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。给出一句理由。
2. Schema 设计。字段顺序（推理在前、答案在后），为「未知」设置可空字段，枚举与正则的取舍，必填字段。
3. 失败策略。最大重试次数、兜底模型、优雅的 `null` 处理、分布外拒绝。
4. 验证计划。Schema 合规率（目标 100%）、语义有效性（LLM 评判）、字段覆盖率、延迟 p50/p99。

拒绝任何将 `answer` 或 `decision` 置于推理字段之前的设计。拒绝在没有 schema 的情况下使用裸 JSON 模式。对仅支持 FSM 的库背后的递归 schema 予以标注。
