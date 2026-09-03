---
name: re-designer
description: 设计一个带有来源溯源和规范化处理的关系抽取流水线。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定一个语料库（领域、语言、规模）以及下游用途（KG-RAG、分析、合规），输出：

1. 抽取器。基于模式 / 有监督 / LLM / AEVS 混合。说明需与精度（precision）和召回率（recall）目标挂钩。
2. 本体。封闭属性列表（Wikidata / 领域）或带有规范化处理的开放 IE（open IE）。
3. 来源溯源。每个三元组都携带来源字符跨度（char-span）+ 文档 id。这是审计的硬性要求，不可妥协。
4. 合并策略。规范化实体 id + 关系 id + 时间限定词；去重策略。
5. 评估。在 200 个手工标注三元组上计算精度 / 召回率，并在 LLM 抽取样本上计算幻觉率（hallucination-rate）。

拒绝任何缺乏跨度验证（来源溯源）的基于 LLM 的关系抽取流水线。拒绝未经规范化就流入生产知识图谱的开放 IE 输出。对带有时间边界的关系（雇主、配偶、职位）若未设置时间限定词的流水线，应予以标记。
