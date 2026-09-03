---
name: grammar-pipeline
description: 为下游 NLP 任务设计一个经典的词性标注 + 依存句法分析流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

针对一个下游任务（信息抽取、改写校验、查询分解、词形还原），你输出：

1. 标签集。纯英文的遗留流水线使用 Penn Treebank，多语言或跨语言场景使用 Universal Dependencies。
2. 库。大多数生产环境使用 spaCy（`en_core_web_sm` / `_lg` / `_trf`），学术级多语言使用 stanza，UD 精度最高时使用 trankit。
3. 集成代码片段。调用该库并消费 `.pos_`、`.dep_`、`.head` 的那 3-5 行代码。
4. 需要测试的失败模式。名词-动词歧义（`saw`、`book`、`can`）与介词短语挂靠歧义是经典陷阱。抽样 20 个输出并人工目测。

拒绝推荐自行从零实现解析器。从零构建解析器属于研究课题，而非应用任务。对于任何消费词性标注却未处理大小写变体的流水线，应标记为脆弱。
