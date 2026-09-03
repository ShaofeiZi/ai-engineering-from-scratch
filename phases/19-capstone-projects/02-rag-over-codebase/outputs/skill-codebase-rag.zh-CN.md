---
name: codebase-rag
description: 构建跨仓库语义搜索系统，具备 AST 感知分块、混合检索、增量重新索引和带引用的回答。
version: 1.0.0
phase: 19
lesson: 02
tags: [capstone, rag, code-search, tree-sitter, qdrant, bm25, hybrid-retrieval]
---

给定 10 个以上、总计至少 200 万行代码的仓库，构建一套数据摄入流水线、混合索引和强制引用的查询智能体，使其能够以可验证的 file:line 锚点回答跨仓库问题。

构建计划：

1. 用 tree-sitter 解析每个文件。在函数和类节点边界处分块。存储 `{repo, path, start_line, end_line, symbol, body}`。
2. 使用 Claude Haiku 4.5 或 Gemini 2.5 Flash 对每个分块进行摘要，系统提示词启用 prompt cache。将一句话摘要存储在分块旁。
3. 索引到三个结构中：Qdrant（稠密向量，Voyage-code-3 或 nomic-embed-code）、Tantivy（带字段权重的 BM25）和 kuzu（符号图边，覆盖导入、调用、继承关系）。
4. 构建一个 LangGraph 查询智能体，包含三个节点：retrieve（稠密并行 BM25）、rerank（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）、synth（Claude Sonnet 4.7，启用 prompt cache 并要求 file:line 引用）。
5. 后过滤：拒绝任何缺乏可验证 `(repo/path:start-end)` 锚点的声明；重新提问或丢弃。
6. 串联一个 git push webhook，计算符号级 diff 并仅对变更分块重新嵌入。目标：200 万行代码规模下，50 文件提交在 60 秒内可搜索。
7. 用 100 题留出集评估。报告 MRR@10、nDCG@10、引用忠实度和延迟百分位。
8. 运行每周漂移巡检任务，重新执行评估并在 MRR@10 下降超过 5% 时告警。

评估量表：

| 权重 | 标准 | 度量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 题留出集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 回答声明中具有可验证 file:line 锚点的比例 |
| 20 | 延迟与规模 | 在索引语料规模下 10k QPS 的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到可搜索的时间，基于 50 文件提交 |
| 15 | 用户体验与答案格式 | 引用可点击性、代码片段预览、后续追问入口 |

硬性拒绝条件：

- 使用固定大小 token 分块而非 AST 感知分块。这会毒害生成代码比重高的语料库。
- 仅用余弦相似度检索而无 BM25 或重排。已知在精确符号名查询上会失败。
- 缺少强制 file:line 引用的回答。
- 每次 git push 时对全语料库重新嵌入；必须是增量式。

拒绝规则：

- 拒绝在未阅读仓库许可证的情况下对其进行索引。部分许可证禁止嵌入第三方向量存储。
- 拒绝回答声称引用了索引从未见过的文件的查询；在返回前始终验证锚点。
- 拒绝在 p95 超过 4s 时返回完整答案；改为返回部分结果并附带后续追问句柄。

输出：一个包含数据摄入流水线、LangGraph 查询智能体、100 题标注评估集、Langfuse 仪表盘链接的仓库，以及一份分析报告，指出你修复的三个检索失败模式（生成代码毒害、长尾符号召回、跨仓库符号解析）及每种的具体修复变更。
