# 信息检索与搜索

> BM25 精确但脆弱，稠密检索覆盖面广却会漏掉关键词。混合检索是 2026 年的默认方案，其余都是调优。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 04（GloVe、FastText、子词）
**Time:** 约 75 分钟

## 问题

用户输入“what happens if someone lies to get money”，希望找到真正适用的法规：“Section 420 IPC”。关键词搜索会完全漏掉它（没有共享词汇）；如果嵌入没有在法律文本上训练，语义搜索也会漏掉。真正的搜索必须同时处理两种情况。

每个 RAG 系统、每个搜索框、每个文档站点的模糊查找功能，底层都有信息检索。2026 年真正能在生产环境中奏效的架构不是单一方法，而是一条由互补方法组成的链路，每一层都负责捕捉前一层的失败。

本课将构建其中的每个环节，并说明它分别解决哪些失败。

## 概念

![混合检索：BM25 + 稠密检索 + RRF + 交叉编码器重排](../assets/retrieval.svg)

共有四层，按需选用。

1. **稀疏检索（BM25）。** 速度快，精确匹配能力强，语义理解很差。它在倒排索引上运行，对数百万篇文档的单次查询耗时不到 10 毫秒，能够准确找出法规编号、产品代码、错误消息和命名实体。
2. **稠密检索。** 把查询与文档编码成向量，再执行最近邻搜索。它可以捕捉释义和语义相似性，却可能漏掉仅相差一个字符的精确关键词匹配。配合 FAISS 或向量数据库，每次查询耗时 50～200 毫秒。
3. **融合。** 合并稀疏检索与稠密检索的排名列表。倒数排名融合（RRF）是简单的默认方案，因为它忽略处于不同量纲的原始分数，只使用排名位置。如果已知某类信号在当前领域占主导，也可以使用加权融合。
4. **交叉编码器重排。** 从融合结果中取前 30 项，让交叉编码器把查询与文档放在一起，逐对评分，再保留前 5 项。交叉编码器逐对计算的速度比双编码器慢得多，准确率却高得多。只对前 30 项运行，可以摊薄成本。

三路检索（BM25 + 稠密 + SPLADE 等学习式稀疏检索）在 2026 年基准中优于两路检索，但需要支持学习式稀疏索引的基础设施。对大多数团队而言，两路融合再加交叉编码器重排是最佳平衡点。

```figure
gx-hybrid-retrieval
```

## 动手构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有两个参数值得掌握。`k1=1.5` 控制词频饱和程度；数值越高，词语重复获得的权重越大。`b=0.75` 控制长度归一化；0 表示忽略文档长度，1 表示完全归一化。这两个默认值来自 Robertson 在原始论文中的建议，几乎无须调节。

### 第 2 步：使用双编码器进行稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入执行 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 为 384 维，速度快，对大多数英语检索任务也足够强。多语言任务可使用 `paraphrase-multilingual-MiniLM-L12-v2`；追求最高准确率则可使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

常数 `k=60` 来自原始 RRF 论文。较大的 `k` 会拉平排名差异的贡献，较小的 `k` 会让顶部排名占据主导。60 是论文给出的默认值，几乎无须调节。

### 第 4 步：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合在一起。BM25 寻找词法匹配，稠密检索寻找语义匹配，RRF 无须校准分数即可合并两个排名。交叉编码器把查询—文档对放在一起，对前 30 项重新评分，从而捕捉双编码器漏掉的细粒度相关性。最后保留前 5 项。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档确实存在的查询中，它有多大比例出现在前 k 项？ |
| MRR（平均倒数排名） | 第一个相关文档的 1/排名 的平均值。 |
| nDCG@k | 考虑相关程度的分级，而不只是相关/不相关二元判断。 |

对 RAG 而言，检索器的 **Recall@k** 是最重要的指标。如果正确段落不在检索集合中，阅读器就无法回答。

调试建议：对于失败的查询，比较稀疏排名与稠密排名的差异。如果一方找到了正确文档，另一方没有，问题要么是词汇不匹配（修复方法：补上缺失的另一类检索），要么是语义歧义（修复方法：使用更好的嵌入或重排器）。

## 学以致用

2026 年的技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1000～10 万篇文档 | 内存 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF，无须独立数据库。 |
| 10 万～1000 万篇文档 | 稠密部分使用 FAISS 或 pgvector，BM25 使用 Elasticsearch / OpenSearch，并行运行。 |
| 1000 万篇以上文档 | 使用支持混合检索的 Qdrant / Weaviate / Vespa / Milvus，再对前 30 项进行交叉编码器重排。 |
| 追求前沿最佳质量 | 三路检索（BM25 + 稠密 + SPLADE）+ ColBERT 后期交互重排 |

无论选择哪种方案，都要为评估留出预算。先对检索召回率做基准测试，再评估端到端 RAG 准确率。检索器漏掉的内容，阅读器无法补救。

### 2026 年生产 RAG 的实战经验

- **80% 的 RAG 失败源于摄取和分块，而不是模型。** 团队花数周替换大语言模型、调整提示，却没有发现检索每三次就会返回一次错误上下文。先修复分块。
- **分块策略比块大小更重要。** 固定大小切分会破坏表格、代码和嵌套标题。默认应按句子边界切分；对于技术文档和产品手册，语义分块或基于大语言模型的分块值得额外成本。
- **父文档模式。** 检索较小的“子”块以获得精确度。当同一父章节中的多个子块都出现时，替换为父块以保留上下文。这种方法无需重新训练，就能稳定提升答案质量。
- **k_rerank=3 通常最优。** 超过这个数量后，每多加入一个文本块都会增加词元成本和生成延迟，却无法提升答案质量。如果你的系统中 k=8 仍优于 k=3，说明重排器表现不足。
- **HyDE / 查询扩展。** 根据查询生成一个假设答案，对它做嵌入再检索。这可以跨越简短问题与长文档之间的措辞鸿沟，无须训练即可提升精确度。
- **上下文预算保持在 8K 词元以内。** 如果持续达到上限，说明重排器阈值过于宽松。
- **对所有内容做版本控制。** 提示、分块规则、嵌入模型和重排器都要纳入版本管理。任何漂移都会悄然破坏答案质量。CI 应以忠实度、上下文精确率和未回答问题比例作为门禁，在用户看到回归前将其拦截。
- **三路检索（BM25 + 稠密 + SPLADE 等学习式稀疏检索）在 2026 年基准上优于两路检索**，尤其适合混合专有名词与语义的查询。当基础设施支持 SPLADE 索引时即可交付。

根据 2026 年的行业测量，合理的检索设计可以把幻觉减少 70%～90%。RAG 的大部分性能提升来自更好的检索，而不是模型微调。

## 交付成果

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在 500 篇文档的语料库上实现上面的 `hybrid_search`。测试 20 个查询，比较仅 BM25、仅稠密检索和混合检索的 Recall@5。
2. **中等。** 增加 MRR 计算。对每个已知正确文档的测试查询，分别找出正确文档在 BM25、稠密与混合排名中的位置，并报告各自的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调稠密编码器。从 500 个查询—文档对构建训练集，比较微调前后的召回率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25，根据词频、IDF 和长度为文档评分。 |
| 稠密检索 | 向量搜索 | 把查询与文档编码成向量，再寻找最近邻。 |
| 双编码器 | 嵌入模型 | 分别编码查询与文档，查询速度快。 |
| 交叉编码器 | 重排模型 | 联合编码查询与文档，速度慢但准确。 |
| RRF | 排名融合 | 通过累加 `1/(k + rank)` 合并两个排名。 |
| Recall@k | 检索指标 | 相关文档出现在前 k 项中的查询比例。 |

## 延伸阅读

- [Robertson 与 Zaragoza（2009），概率相关性框架：BM25 及其扩展](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)——BM25 的权威论述。
- [Karpukhin 等（2020），用于开放域问答的稠密段落检索](https://arxiv.org/abs/2004.04906)——DPR，经典双编码器。
- [Formal 等（2021），SPLADE：稀疏词法扩展模型](https://arxiv.org/abs/2107.05720)——缩小与稠密检索差距的学习式稀疏检索器。
- [Cormack、Clarke、Büttcher（2009），倒数排名融合优于 Condorcet 与单项排名学习方法](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)——RRF 论文。
- [Khattab 与 Zaharia（2020），ColBERT：高效而有效的段落搜索](https://arxiv.org/abs/2004.12832)——后期交互检索。
