---
name: entity-linker
description: 设计实体链接流水线 — 知识库、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定一个用例（领域知识库、语言、数据量、延迟预算），输出：

1. 知识库。Wikidata / Wikipedia / 自定义知识库。版本日期。刷新频率。
2. 候选生成器。别名索引、嵌入或混合方式。目标 mention recall @ K。
3. 消歧器。先验+上下文、基于嵌入、生成式或 LLM 提示。
4. NIL 策略。基于最高分的阈值、分类器或显式 NIL 候选。
5. 评估。在留出集上的 mention recall @ 30、top-1 准确率、NIL 检测 F1。

拒绝任何没有 mention 召回基线的 EL 流水线（如果不知道候选生成是否浮现了正确实体，就无法评估消歧器）。拒绝任何使用 LLM 提示的 EL 但未对输出约束为有效 KB id 的流水线。标记那些因流行度偏差影响少数实体（例如名称冲突）却未进行领域微调的系统。
