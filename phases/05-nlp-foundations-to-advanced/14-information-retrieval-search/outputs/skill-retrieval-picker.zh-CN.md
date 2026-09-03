---
name: retrieval-picker
description: 针对给定的语料库和查询模式，选择合适的检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（语料库规模、查询模式、延迟预算、质量门槛、基础设施约束），输出以下内容：

1. 技术栈。仅 BM25、仅稠密检索、混合检索（BM25 + 稠密 + RRF）、混合检索 + cross-encoder 重排序，或三路检索（BM25 + 稠密 + 学习型稀疏检索）。
2. 稠密编码器。给出具体模型名称（`all-MiniLM-L6-v2`、`bge-large-en-v1.5`、`e5-large-v2`、`paraphrase-multilingual-MiniLM-L12-v2`）。需与语言、领域、上下文长度相匹配。
3. 重排序器。若使用 cross-encoder，需给出具体模型名称（`cross-encoder/ms-marco-MiniLM-L-6-v2`、`BAAI/bge-reranker-large`）。需提示在 top-30 结果上会增加约 30-100ms 的延迟。
4. 评估方案。Recall@10 是检索器的主要指标。对于多答案场景使用 MRR。先建立基线，再以此为参照衡量增量改进。

对于包含命名实体、错误代码或产品 SKU 的语料库，除非用户能提供证据表明稠密检索可以处理精确匹配，否则拒绝推荐仅使用稠密检索的方案。对于高风险检索场景（法律、医疗），当最终 top-5 结果将直接决定用户答案时，拒绝跳过重排序步骤。
