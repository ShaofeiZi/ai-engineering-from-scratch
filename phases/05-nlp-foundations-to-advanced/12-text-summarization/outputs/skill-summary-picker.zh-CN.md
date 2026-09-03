---
name: summary-picker
description: 选择抽取式或生成式，指出所用的库，并加入事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务（文档类型、合规要求、长度、算力预算），输出：

1. 方法。抽取式还是生成式。用一句话解释原因。
2. 起始模型 / 库。指出其名称。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed`，或者某个 LLM 提示。
3. 评估方案。ROUGE-1、ROUGE-2、ROUGE-L（使用 `rouge-score` 并启用词干提取）。若是生成式，则再加上事实性检查。
4. 一个需要排查的失败模式。实体替换是生成式新闻摘要中最常见的问题；标记源文档中的实体未出现在摘要中的样本。

对于医疗、法律、金融等受监管内容，若没有事实性闸门把关，应拒绝生成式摘要。当输入超过模型的上下文窗口时，应标记为需要分块 map-reduce 摘要，而不仅是截断。
