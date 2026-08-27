# 综合项目 02——代码库 RAG（跨仓库语义搜索）

> 到 2026 年，成熟的工程团队都会配备内部代码搜索系统。它理解的是含义，而不只是字符串。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业知识图、Aider 的 repomap 和 Pinterest 的内部 MCP，架构都很相似：接入多个代码仓库，用 tree-sitter 解析代码，按函数和类分块，执行混合检索与重排，再给出带引用的答案。本综合项目要求你构建一套覆盖 10 个仓库、200 万行代码的系统，并能在每次 git push 后完成增量索引更新。

**Type:** 综合项目
**Languages:** Python（数据摄入）、TypeScript（API + UI）
**Prerequisites:** 第 5 阶段（NLP 基础）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 小时

## 问题

到 2026 年，每个前沿编程智能体都会配备代码库检索层，因为仅靠上下文窗口无法回答跨仓库问题。Claude 的 100 万令牌上下文虽有帮助，却不能取代经过排序的检索。直接对原始代码分块做朴素余弦搜索时，生成代码、单体仓库中的重复内容，以及极少被导入的长尾符号都会干扰结果。生产级方案是在感知抽象语法树（AST）的代码分块上执行混合检索（稠密向量 + BM25），再用重排模型排序，并以符号引用图作为支撑。

你得为一组真实在用的仓库建立索引，而不是只拿一个教程仓库练手；只有这样才能学会这套方法。你需要测量 MRR@10、引用忠实度和增量时效性。真正容易出问题的是基础设施：单体仓库可能有 10 万个文件，一次推送可能改动其中一半，而有些查询必须串联四个仓库才能答对。

## 核心概念

按 AST 结构处理的数据摄入管线会用 tree-sitter 解析每个文件，提取函数节点和类节点，并沿节点边界分块，而不是套用固定的令牌窗口。每个分块有三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏的 BM25 词项，以及一段简短的自然语言摘要。摘要增加了第三种可检索模态：用户可能会问“X 的授权是怎么做的”，摘要中会出现“authz”，即使代码里只有 `check_permission`。

系统采用混合检索。每次查询都会并行发起稠密向量搜索和 BM25 搜索，合并各自排名前 k 的结果，再把并集交给交叉编码器重排模型（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排后的列表会送入长上下文答案合成模型，例如启用提示缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B。系统要求每项论断都标明文件和行号范围；后置过滤器会拒绝没有引用的答案。

保持增量索引及时更新，本质上是个基础设施问题。Git push 会触发差异计算，找出变更的文件和符号。系统只为受影响的分块重新生成嵌入，并重新计算相关的跨文件符号边，例如导入关系和方法调用。这样既能保持索引一致，也不必在每次提交后重新处理 200 万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言的语法（Python、TypeScript、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为回退
- 稀疏索引：Tantivy（Rust）+ BM25F，对符号名称和正文设置不同的字段权重
- 向量数据库：Qdrant 1.12（支持混合搜索），或供向量规模低于 5000 万的团队使用 pgvector + pgvectorscale
- 分块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，启用提示缓存
- 重排器：Cohere rerank-3 或自托管 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows 负责数据摄入，LangGraph 负责查询智能体
- 答案合成模型：Claude Sonnet 4.7（100 万令牌上下文），启用提示缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式），记录导入边与调用边
- 可观测性：为每一步检索与答案合成记录 Langfuse 追踪跨度（span）

```figure
ce-hybrid-retrieval
```

## 动手构建

1. **数据摄入遍历器。** 每次推送钩子触发后，遍历 Git 历史记录并收集变更文件。用 tree-sitter 解析每个文件，提取函数节点、类节点及其完整源码范围。输出分块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **分块摘要器。** 将代码分块成批送入 Haiku 4.5，并对系统提示的前置部分使用提示缓存。提示词是：“用一句话概括这个函数，并说明它的对外契约和副作用。”摘要与代码分块一同存储。

3. **嵌入队列。** 建立两条并行队列：稠密嵌入队列使用 Voyage-code-3，每批 128 条；摘要嵌入队列使用同一模型，但输入摘要字符串。将向量写入 Qdrant，并附上载荷 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 构建按字段加权的 Tantivy 索引：符号名称权重为 4，符号正文权重为 1，摘要权重为 2。这样既能查“名为 X 的函数”，也能查“实现 X 功能的函数”。

5. **符号图。** 为每个分块记录以下边：导入（本文件使用仓库 Z 中的符号 Y）、调用（本函数调用类 C 的方法 M）以及继承。将这些关系存入 kuzu，查询时借此跨仓库扩展检索范围。

6. **查询智能体。** 用三个节点构建 LangGraph。`retrieve` 并行发起稠密向量检索和 BM25 检索，并按（仓库、路径、符号）去重。`rerank` 使用交叉编码器对前 50 项结果重排，只保留前 10 项。`synth` 把重排后的分块作为上下文调用 Claude Sonnet 4.7，缓存系统提示，并强制输出“文件:行号”引用。

7. **强制引用。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的论断都要标记出来，以便重新向模型提问，或直接删除。最终只向用户返回带引用的答案。

8. **增量重建索引。** 每次收到 Webhook 时，计算符号级差异。只为文本发生变化的分块重新生成嵌入；如果分块的导入发生变化，则重新计算其符号边。目标指标是：对于总计 200 万行代码的仓库集合，一次涉及 50 个文件的推送能在 60 秒内完成重新索引。

9. **评测。** 标注 100 个跨仓库问题，并为每题给出“文件:行号”形式的标准答案。测量 MRR@10、nDCG@10、引用忠实度（带有可验证锚点的论断占比）以及 p50/p99 延迟。

## 实际使用

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付成果

交付的技能文件是 `outputs/skill-codebase-rag.md`。输入一组代码仓库后，它会启动数据摄入管线、混合索引和查询智能体，为任何跨仓库问题返回带引用的答案。评分标准如下：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 题留出集上的 MRR@10 与 nDCG@10 |
| 20 | 引用忠实度 | 答案中带有可验证“文件:行号”锚点的论断占比 |
| 20 | 延迟与规模 | 在目标语料规模下的 p95 查询延迟（10k QPS） |
| 20 | 增量索引正确性 | 50 文件提交从 git push 到可检索的时间 |
| 15 | 用户体验与答案格式 | 引用能否点击、代码片段预览和后续追问入口 |
| **100** | | |

## 练习

1. 把 Voyage-code-3 换成自托管的 nomic-embed-code。测量 MRR@10 的变化，并报告启用重排后差距是否缩小。

2. 向语料中加入 20% 的生成代码（由 LLM 生成的样板代码），然后重新评测。观察这些内容如何污染检索结果。为载荷增加“generated”标志，并降低此类命中的权重。

3. 在你的语料规模下，对比 Qdrant 混合搜索与 pgvector + pgvectorscale，并报告批量大小为 1 时的 p99。

4. 增加基于抽样的漂移检查：每周重跑一次 100 题评测。如果 MRR@10 下跌超过 5%，就发出警报。

5. 将系统扩展到跨语言符号解析：例如，一个 Python 函数通过 gRPC 调用 Go 服务。使用符号图把它们关联起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| AST 感知分块（AST-aware chunking） | “函数级拆分” | 沿 tree-sitter 节点边界切分代码，而不是使用固定令牌窗口 |
| 混合搜索（Hybrid search） | “稠密 + 稀疏” | 并行执行 BM25 和向量搜索，合并各自排名前 k 的结果，再进行重排 |
| 交叉编码器重排（Cross-encoder rerank） | “第二阶段排序” | 将每个查询与候选项组成一对后联合评分的模型，比余弦相似度更准确 |
| 提示缓存（Prompt caching） | “缓存的系统提示” | Claude 和 OpenAI 在 2026 年提供的功能，可将重复前缀令牌的成本最多降低 90% |
| 符号图（Symbol graph） | “代码图” | 记录跨文件、跨仓库的导入、调用和继承关系 |
| 引用忠实度（Citation faithfulness） | “有依据的答案比例” | 用户点击锚点并阅读所引用代码范围后，能够验证的论断占比 |
| 增量重建索引（Incremental re-index） | “从推送到可检索的时间” | 从 git push 到变更符号可以被查询所经过的实际时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本综合项目的架构详解参考
- [Aider repo-map](https://aider.chat/docs/repomap.html) — 基于 tree-sitter 排序的仓库视图
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业化符号图 RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 细节
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 内部平台参考
