---
name: ner-picker
description: 针对给定抽取任务选择合适的 NER 方案。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定任务描述（领域、标签集、语言、延迟、数据量），输出：

1. 方法。基于规则 + 词典、CRF、BiLSTM-CRF，或 Transformer 微调。
2. 起始模型。给出名称（spaCy 模型 ID，如 `en_core_web_sm` / `en_core_web_trf`；Hugging Face checkpoint ID，如 `dslim/bert-base-NER`；或“custom, trained from scratch”）。
3. 标注策略。BIO、BILOU 或基于区间（span-based）。用一句话说明理由。
4. 评估。使用 `seqeval`。始终报告实体级 F1，绝不报告 token 级 F1。

当标注样本少于 500 条时，拒绝推荐微调 Transformer，除非用户已拥有预训练的领域模型（例如用于医疗的 BioBERT）。对于嵌套实体，应标记为需要基于区间或多轮（multi-pass）模型。若用户在使用现成的 CoNLL-2003 标签的同时提及“生产规模”，则要求进行词典审计。
