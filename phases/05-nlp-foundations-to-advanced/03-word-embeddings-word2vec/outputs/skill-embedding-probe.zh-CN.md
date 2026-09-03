---
name: embedding-probe
description: 检查 word2vec 模型。运行类比推理、查找近邻、诊断模型质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你通过探测训练好的词向量来验证它们是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和一个词表，你需要运行：

1. 三个经典类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦相似度。
2. 对用户提供领域相关词汇进行五个最近邻测试。打印 top-5 近邻及其余弦相似度。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)`，在浮点精度范围内成立。
4. 一个退化检查。如果任何向量的范数低于 0.01 或高于 100，则模型存在训练缺陷，需予以标记。

不要仅凭类比准确率就认定模型优秀。类比基准容易被针对优化，且难以迁移到下游任务。建议将内在评估与下游评估结合使用。
