# RAG 分块策略

> 分块配置对检索质量的影响，与嵌入模型的选择同样大（Vectara，NAACL 2025）。分块做错，再多重排也救不回来。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 14（信息检索）、阶段 5 · 22（嵌入模型）
**Time:** 约 60 分钟

## 问题

你把一份 50 页的合同放进 RAG 系统。用户问：“终止条款是什么？”检索器却返回封面。为什么？因为模型使用 512 词元的文本块训练，而终止条款位于第 20 页，被分页符从中间切开，局部又没有能把它与查询联系起来的关键词。

修复方法不是“购买更好的嵌入模型”，而是调整分块。应该多大？是否重叠？在哪里切分？要不要附带周围上下文？

2026 年 2 月的基准给出了令人意外的结果：

- Vectara 2026 年的研究：递归式 512 词元分块以 69% 对 54% 的准确率胜过语义分块。
- 在 Natural Questions 上使用 SPLADE + Mistral-8B 时，重叠没有带来任何可测量收益。
- 上下文悬崖：上下文达到约 2500 个词元时，回答质量会急剧下降。

那个“显而易见”的答案（语义分块、20% 重叠、1000 个词元）通常是错的。本课将帮助你理解六种策略，并说明何时选择哪一种。

## 概念

![在同一段落上展示六种分块策略](../../../../../../phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/assets/chunking.svg)

**固定分块。** 每 N 个字符或词元切一次。最简单的基线，会在句子中间截断；压缩效果好，连贯性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 切分，再依次尝试 `\n`、`.` 和空格，层层回退。它是 2026 年的默认方案。

**语义分块。** 嵌入每个句子，计算相邻句子之间的余弦相似度，在相似度低于阈值处切分。可以保留主题连贯性，但速度较慢；有时会生成只有 40 个词元的碎片，损害检索效果。

**句子分块。** 按句子边界切分，每个块包含一句或由 N 个句子组成的窗口。在最长约 5000 个词元的范围内，其效果可与语义分块相当，成本却低得多。

**父文档。** 同时保存用于检索的小型子块和用于提供上下文的较大父块。按子块检索，返回父块。这种方法可以平稳退化：即使子块划分欠佳，也能返回合理的父块。

**后期分块（2024）。** 先在词元层面嵌入整篇文档，再把词元嵌入池化为文本块嵌入，从而保留跨块上下文。它适用于长上下文嵌入模型（BGE-M3、Jina v3），但计算成本更高。

**上下文化检索（Anthropic，2024）。** 在每个文本块前添加一段由大语言模型生成、说明其在文档中所处位置的摘要（“本块位于终止条款第 3.2 节……”）。Anthropic 自己的基准显示，检索效果可提高 35%～50%，但建立索引的成本很高。

### 胜过所有默认值的规则

让文本块大小与查询类型匹配：

| 查询类型 | 文本块大小 |
|------------|-----------|
| 事实型（“CEO 叫什么名字？”） | 256～512 个词元 |
| 分析型/多跳 | 512～1024 个词元 |
| 理解整个章节 | 1024～2048 个词元 |

数据来自 NVIDIA 2026 年基准。文本块应该足够大，能容纳答案及其局部上下文；又要足够小，让检索器返回的前 K 项聚焦答案，而不是充斥上下文噪声。

```figure
n5-chunk-cuts
```

## 动手构建

### 第 1 步：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第 2 步：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

应在自己的领域上调节 `threshold`。过高会产生碎片，过低则会形成一个巨大的文本块。

### 第 3 步：父文档

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞见是对父块去重。多个子块可能映射到同一个父块，全部返回会浪费上下文。

### 第 4 步：上下文化检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

为添加上下文后的文本块建立索引。查询时，检索过程会受益于额外的周边信号。

### 第 5 步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终进行基准测试。最适合你语料库的策略，未必与任何博客文章中的结论一致。

## 陷阱

- **只在事实型查询上评估分块。** 多跳查询会呈现完全不同的优胜者。应使用按查询类型分层的评估集。
- **语义分块不设最小大小。** 会生成只有 40 个词元的碎片，损害检索。必须强制设置 `min_tokens`。
- **把重叠当作教条。** 2026 年的研究发现，重叠往往没有任何收益，却会让索引成本翻倍。先测量，不要想当然。
- **不强制最小/最大值。** 5 个词元或 5000 个词元的文本块都会破坏检索，必须设置上下限。
- **跨文档分块。** 绝不要让一个文本块跨越两篇文档。始终逐文档切分，再合并结果。

## 学以致用

2026 年的技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建、语料库特征未知 | 递归分块，512 个词元，不重叠 |
| 事实型问答 | 递归分块，256～512 个词元 |
| 分析型/多跳问答 | 递归分块，512～1024 个词元 + 父文档 |
| 交叉引用密集（合同、论文） | 后期分块或上下文化检索 |
| 对话语料库 | 逐轮分块 + 说话人元数据 |
| 短话语（推文、评论） | 一篇文档 = 一个文本块 |

从无重叠的递归 512 词元方案开始，在包含 50 个查询的评估集上测量 Recall@5，再据此调优。

## 交付成果

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 对一篇 20 页的文档分别使用 fixed(512, 0)、recursive(512, 0) 与 recursive(512, 100) 分块，比较文本块数量与边界质量。
2. **中等。** 在 5 篇文档上构建一个包含 30 个查询的评估集。分别测量递归分块、语义分块和父文档方案的 Recall@5。哪一种胜出？结果与博客文章一致吗？
3. **困难。** 实现上下文化检索。测量它相对于递归分块基线的 MRR 提升，并报告索引成本（大语言模型调用）与准确率增益。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 文本块 | 文档的一部分 | 被嵌入、建立索引和检索的子文档单元。 |
| 重叠 | 安全边界 | 相邻文本块共享的 N 个词元；2026 年基准中往往无用。 |
| 语义分块 | 智能分块 | 在相邻句子嵌入相似度下降的位置切分。 |
| 父文档 | 两级检索 | 检索小型子块，返回较大的父块。 |
| 后期分块 | 嵌入后再分块 | 在词元层面嵌入完整文档，再池化成文本块向量。 |
| 上下文化检索 | Anthropic 的技巧 | 建立索引前，为每个文本块添加由大语言模型生成的上下文摘要。 |
| 上下文悬崖 | 2500 词元墙 | RAG 在约 2500 个上下文词元处出现的质量下降（2026 年 1 月）。 |

## 延伸阅读

- [Yepes 等 / LangChain——递归字符切分文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/)——生产环境中的默认方案。
- [Vectara（2024，NAACL 2025），分块配置分析](https://arxiv.org/abs/2410.13070)——分块与嵌入选择同等重要。
- [Jina AI——长上下文嵌入模型中的后期分块（2024）](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)——后期分块论文。
- [Anthropic——上下文化检索](https://www.anthropic.com/news/contextual-retrieval)——通过大语言模型生成上下文前缀，使检索效果提升 35%～50%。
- [NVIDIA 2026 分块大小基准——Premai 摘要](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)——按查询类型选择文本块大小。
