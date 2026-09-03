---
name: topic-picker
description: 为语料库选择 LDA 或 BERTopic。指定库、调参项、评估方式。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定一个语料库描述（文档数量、平均长度、领域、语言、计算预算），输出：

1. 算法。LDA / NMF / BERTopic / Top2Vec / FASTopic。给出一句话的理由。
2. 配置。主题数量（从约 sqrt(n_docs) 起步）、`min_df` / `max_df` 过滤参数、神经网络方法所用的嵌入模型。
3. 评估。通过 `gensim.models.CoherenceModel` 计算主题一致性（c_v）、主题多样性，并辅以 20 个样本的人工通读。
4. 需要探查的失败模式。对于 LDA，是"垃圾主题"吸收停用词和高频词；对于 BERTopic，是 -1 离群簇吞并语义模糊的文档。

如果文档长度超过嵌入模型的上下文窗口且未制定分块策略，则拒绝使用 BERTopic。对于过短的文本（推文、不足 10 个 token 的评论），拒绝使用 LDA，因为一致性会崩塌。任何低于 5 或高于 200 的 n_topics 取值，对于真实数据而言大概率是错误的，应予以标记。
