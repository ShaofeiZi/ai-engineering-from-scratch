# 分块策略对比

> 分块方式决定了检索器究竟有机会召回什么内容。边界一旦切错，后面的 embedding model、reranker 和 LLM 都无法补回这一步造成的损失。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 11 阶段第 04 课（embeddings）、第 06 课（RAG）、第 07 课（advanced RAG）；第 19 阶段 Track B 基础课（第 20–29 课）
**Time:** 约 90 分钟

## 学习目标
- 从零实现五种分块策略：fixed-window、sentence、recursive-split、semantic clustering，以及 structural markdown headers。
- 在带有 gold-labeled answer span 的示例语料上测量 recall@k，并解释为什么某种策略更适合 prose，而另一种策略更适合技术文档。
- 读懂 chunk 长度分布，并识别各策略带来的失败模式：orphan sentences、mid-symbol cuts、header-only chunks、semantic drift。
- 即使不跑 benchmark，也能只根据三个属性为新语料挑选默认策略：document type、average paragraph length，以及格式是否自带显式结构。

## 问题

每条 RAG pipeline 的第一步，都是把源文档切成若干小块。块既要足够小，embedding model 才能装得下；也要足够大，才能保留一个完整、自洽的语义单元。切分边界并不是一个无关紧要的超参数，它直接决定了 retriever 能返回什么内容的上限。

一个查询如果在问 “what does the budget abort threshold look like”，只有当包含这个 threshold 的 chunk 本身可被召回时，检索才可能成功。如果 fixed-window splitter 恰好把 threshold 的值和上下文切开，embedding 就会漂到别的 cluster，BM25 分数下降，reranker 看到的也只剩噪音，最终 LLM 生成的答案就会出错。2024 年论文 “LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs” 表明，仅仅因为 chunking 方案不同，retrieval recall 就可能出现 35 个百分点的绝对波动。2025 年关于 contextual chunk headers 的后续工作缩小了这个差距，但并没有把问题彻底解决。

本课会把五种策略并排实现，让它们在带有 gold-labeled answer span 的 fixture corpus 上跑同一套评测，让你直接用 recall 数字来比较。

## 概念

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### 固定窗口

最直接、也最粗暴的 baseline。每隔 N 个字符硬切一次。也可以加 overlap，这样一个刚好在 N 位置被截断的句子，能在从 N - overlap 开始的下一块里重新完整出现。它很快、确定性强，但边界质量很差。拿它当 control，不要把它当默认值。

### 句子切分

用 regex 或简单状态机按句子边界切分，然后把一个或多个句子打包进目标字符预算内的 chunk。它至少不会把词切成两半，但仍然可能在段落中间、章节中间断开。很多早期 RAG pipeline 都把它当默认值；对于没有其他显式结构的 prose，它依然是个合理选择。

### 递归切分

这是 2023 年那批库带火的层级策略。优先尝试最强的分隔符，例如双换行，也就是 paragraph；如果还不够小，就退到单换行；再不行就退到句子；最后才退到字符级。只要 chunk 已经落进预算，就停止递归。它对结构不一致的文档很强，因为它会按局部结构自适应地选择切法。

### 语义集群

先对每个句子做 embedding，再把主题中心接近的相邻句子聚成一块。每当下一句与当前 centroid 的相似度掉到阈值以下，就在那里切开。它的边界不是按字符，而是按语义变化决定。代价是构建更慢，而且强依赖 embedding model；但对那些在单个段落内部频繁切换主题的文档，它更有韧性。

### 结构性标记标题

如果文档本身携带显式结构，比如 markdown、reStructuredText 或 RFC 风格的编号章节，就直接按 heading boundary 切。每个 chunk 都包含一个 heading，以及它下面直到下一个同级或更高层级标题之前的全部内容。按主题来看，它切出来的块最小也最干净，但前提是语料本身得足够规整。

### 如何测量边界选择

一个 gold-labeled query 会附带答案 span 在源文档中的精确字符 offset。完成 chunking 之后，问题就变成了：retriever 返回的 top-k chunks 里，是否有任意一个与 gold span 发生重叠？如果有，这个 query 的 recall@k 就记为 1；如果没有，就记为 0。对整个 query 集合求平均，再对每种策略重复同样的评估，最后你看到的 spread 就是在告诉你：哪种边界策略能在你的语料里真正站得住。

```figure
ci-chunk-boundaries
```

## 动手实现

`code/main.py` 实现了：

- `fixed_window(text, size, overlap)`，也就是 baseline。
- `sentence_chunks(text, target)`，简单的句子打包器。
- `recursive_split(text, separators, target)`，层级递归切分。
- `semantic_chunks(text, similarity_threshold)`，基于 centroid 的聚类切分，建立在一个确定性的 mock embedding 之上。
- `structural_markdown(text)`，一个能识别标题结构的 splitter。
- `mock_embed(text, dim)`，基于 hash 的 embedding，这样整个循环可以离线跑。
- `DenseIndex`，其数据形状与 Phase 19 Track B 混合检索课里用的是同一类。
- `eval_recall(strategy, corpus, queries, k)`，负责对比各策略的评估循环。
- 一个 `main()`，它会把所有策略都跑在 fixture corpus 上，然后打印一张 recall@k 表。

运行它:

```bash
python3 code/main.py
```

输出是一张小表：每种策略一行，每个 k 一列。sentence 会在结构化 fixture 上落后，structural-markdown 会在 markdown fixture 上胜出。recursive 在 mixed fixture 上表现稳，因为它能自适应；semantic clustering 则会在缺少明显结构线索的 prose fixture 上占优。

## 表格无法掩盖的失败模式

**Orphan sentences.** sentence packing 会产生脱离主题句的碎块，于是 embedding 会漂到错误的 cluster。

**Mid-symbol cuts.** fixed-window 在 code 或 YAML 内部切开时，会把 identifier 直接劈成两半，两半都只剩噪音。

**Header-only chunks.** structural markdown 有时会吐出只包含 `## Title` 的 chunk。要么过滤掉，要么把下一块的首段并进来。

**Semantic drift.** semantic clustering 在整篇语料都围绕同一主题时容易切得不够细。一个 5000 字符的 chunk 会把许多具体答案揉进一个发散的 embedding。要把 semantic 与硬性字符上限一起使用。

**Stale embeddings.** semantic clustering 依赖 embedding model。你一旦换模型，chunk 本身也会变化。要么把 chunk model 与 retrieval model 分开固定，要么每次一起重建 index。

## 不跑基准时如何选默认值

给新语料选默认 chunker，主要看三个属性。

| 属性 | 取值 | 默认策略 |
|----------|-------|---------|
| 文档类型 | 无显式结构的 prose | Recursive split，target 800 |
| 文档类型 | Markdown / RFC / API docs | Structural markdown |
| 文档类型 | 代码 | AST-aware（此处不展开；参见第 19 阶段第 02 课） |
| 段落长度 | 很长且单一主题 | Sentence，target 500 |
| 段落长度 | 较短且主题混杂 | Semantic，threshold 0.6 |

如果拿不准，就先选 recursive split。它是单一策略里最稳的 baseline。

## 用它

生产实践：

- 在发布新 pipeline 前先跑 eval；不要盲信库默认的 chunking 策略。
- 每次换 embedding model 或换 corpus mix，都重新跑一遍；赢家取决于具体语料。
- 在每个 chunk 的 metadata 里记录 strategy name，后面出现回归时才能溯源。

## 放进系统里

第 69 课里的 Track F 端到端 RAG 系统，会把这里选出来的 chunker 作为第一阶段。第 68 课的 eval harness 读取的 recall@k，也和本课 `eval_recall` 返回的是同一种数据形状。选出在你语料上胜出的策略，然后把它继续接入后续系统。

## 练习

1. 加第六种策略：token-window，用 `tiktoken` 而不是字符数。把它和 fixed-window 在同一个 fixture 上比较。
2. 往 prose fixture 里注入 30% 的 code blocks。重新跑表，解释为什么除了 structural markdown 之外，其他策略都会掉 recall。
3. 把确定性的 mock embedding 换成你项目里真实 provider 的 embedding。测 semantic-clustering 的 recall delta，并报告不同策略间的差距是变宽还是变窄。
4. 给每个 chunk 增加一个 `summary` 字段，也就是一句话的 centroid 描述。把 summary 拼接回 chunk body 后重新跑 eval，测量 recall 提升。

## 关键术语

| 术语 | 人们常说的话 | 它真正表示什么 |
|------|-----------------|------------------------|
| Recall@k | "Did we get the right chunk?" | top-k chunk 中是否有任意一个覆盖了 gold answer span |
| Chunk overlap | "Sliding window" | 在下一块中重新包含前一块最后 N 个字符 |
| Structural splitter | "Header-aware chunks" | 按 H1/H2/H3 边界切分，heading 文本本身也属于 chunk |
| Semantic chunker | "Topic-aware chunks" | 对句子做 embedding，按 centroid similarity 聚类，遇到 drift 就切开 |
| Centroid drift | "Topic shift" | 运行均值与下一句之间的 cosine similarity 下降到阈值以下 |

## 进一步阅读

- [LongRAG：用长上下文 LLM 增强 Retrieval-Augmented Generation（arXiv 2406.15319）](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- 第 11 阶段第 06 课 - RAG fundamentals
- 第 11 阶段第 07 课 - advanced RAG
- 第 19 阶段第 65 课 - hybrid retrieval that ranks the chunks produced here
- 第 19 阶段第 68 课 - the eval harness that scores the strategy choice in production
