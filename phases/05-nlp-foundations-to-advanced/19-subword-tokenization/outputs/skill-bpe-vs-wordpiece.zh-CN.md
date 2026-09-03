---
name: skill-bpe-vs-wordpiece
description: 针对给定语料和部署目标，选择分词器算法、词表大小与库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定一个语料（规模、语言、领域）和部署目标（从零开始训练 / 微调 / 兼容 API 的推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。用一句话说明理由。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。说明理由。
3. 词表大小。四舍五入到最近的 1k。结合模型规模和语言覆盖范围说明理由。
4. 覆盖设置。`character_coverage`、`byte_fallback`、特殊 token 列表。
5. 验证方案。在留出集上的平均每词 token 数、OOV 率、压缩比、解码往返一致性。

对于在包含罕见文字内容的语料上训练 character-coverage <0.995 的分词器，应予拒绝。对于没有在 CI 中进行冻结的 `tokenizer.json` 哈希校验的词表，应拒绝发布。对于任何词表低于 16k 的单语分词器，应标记为可能规格不足。
