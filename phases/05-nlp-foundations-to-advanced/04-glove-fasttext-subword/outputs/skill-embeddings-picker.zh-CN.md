---
name: skill-embeddings-picker
description: 为新的语言模型或文本管线挑选分词方案。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，你应输出：

1. 分词策略（词级、BPE、WordPiece、SentencePiece、字节级 BPE）。附一句理由。
2. 词表大小目标。纯英语语言模型：32k。多语言：64k-100k。代码：50k-100k。
3. 库调用及确切的训练命令。指明库名（Hugging Face `tokenizers`、`sentencepiece`）。引用参数。
4. 一个可复现性陷阱。分词器与模型不匹配是最常见的隐性生产 bug。指明哪个分词器与哪个预训练检查点配对，并警告不要互换。

当用户在对预训练 LLM 进行微调时，拒绝推荐训练自定义分词器（微调必须使用预训练分词器）。拒绝为任何生产推理路径推荐词级分词。将非英语或多脚本语料标记为需要启用字节回退的 SentencePiece。
