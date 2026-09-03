# 嵌入模型——2026 深入解析

> Word2Vec 为每个词生成一个向量。现代嵌入模型则为每个段落生成一个跨语言向量，提供稀疏、稠密和多向量视图，还能调整大小以适应索引。选择错误，RAG 就会检索到错误内容。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 03（Word2Vec）、阶段 5 · 14（信息检索）
**Time:** 约 60 分钟

## 问题

你的 RAG 系统有 40% 的时间检索到错误段落。罪魁祸首通常不是向量数据库，也不是提示，而是嵌入模型。

2026 年选择嵌入模型，需要在五个维度上作出决定：

1. **稠密、稀疏还是多向量。** 每个段落一个向量、每个词元一个向量，还是一个带权稀疏词袋。
2. **语言覆盖。** 纯英语任务中，单语言英语模型仍然胜出；语料库混合多种语言时，多语言模型更合适。
3. **上下文长度。** 512、8192 还是 32768 个词元——而真正有效的容量通常只有宣传最大值的 60%～70%。
4. **维度预算。** 3072 个全精度浮点数意味着每个向量占 12 KB。达到 1 亿个向量时，每月存储费用为 1300 美元。Matryoshka 截断可把成本降至四分之一。
5. **开放还是托管。** 开放权重意味着你掌控技术栈和数据；托管则意味着用控制权换取始终最新的服务。

本课会说明这些权衡，让你依据证据选择，而不是追随上个季度的流行趋势。

## 概念

![稠密、稀疏与多向量嵌入](../../../../../../phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/assets/embedding-modes.svg)

**稠密嵌入。** 每个段落一个向量（通常为 384～3072 维）。通过余弦相似度按语义接近程度排列段落。OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3 都属于此类，也是默认选择。

**稀疏嵌入。** SPLADE 风格。Transformer 为词表中的每个词元预测权重，再将其中绝大部分置零，最终得到大小为 |vocab| 的稀疏向量。它像 BM25 一样捕捉词法匹配，但词项权重由学习得到，适合关键词密集的查询。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每个词元一个向量，使用 MaxSim 评分：对每个查询词元找出最相似的文档词元，再对分数求和。存储与评分成本更高，但在长查询和领域专用语料库上表现最佳。

**BGE-M3：三者合一。** 一个模型同时输出稠密、稀疏和多向量表示。每一种都可以独立查询，再通过加权求和融合分数。如果希望从一个检查点获得灵活性，它是 2026 年的默认选择。

**Matryoshka 表示学习。** 训练时保证向量的前 N 个维度本身就构成有效嵌入。把 1536 维向量截断到 256 维，只损失约 1% 的准确率，却能节省 6 倍存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2 和 Nomic v1.5+ 都支持它。

### MTEB 排行榜只能说明部分事实

MTEB 是海量文本嵌入基准，发布时包含 8 种任务类型下的 56 项任务，在 MTEB v2 中扩展到 100 多项。2026 年初，Gemini Embedding 2 位居检索榜首（MTEB-R 67.71），Cohere embed-v4 领跑通用榜（MTEB 65.2），BGE-M3 则是开放权重多语言模型第一名（63.0）。排行榜必不可少，却不够充分——始终要在自己的领域上测试。

### 三层模式

| 用例 | 模式 |
|----------|---------|
| 快速首轮检索 | 稠密双编码器（BGE-M3、text-3-small） |
| 提高召回率 | 稀疏检索（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| 对前 50 项提高精确率 | 多向量（ColBERTv2）或交叉编码器重排 |

大多数生产技术栈会同时使用三层。

```figure
gx-matryoshka
```

## 动手构建

### 第 1 步：基线——使用 Sentence-BERT 生成稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 会让点积等于余弦相似度。始终开启它。

### 第 2 步：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后要重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 的训练方式确保最初几个层级近乎无损。非 Matryoshka 模型（原始 Sentence-BERT）经过截断后性能会急剧下降。

### 第 3 步：BGE-M3 的多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次推理调用生成三种索引。分数融合如下：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

应在自己的领域数据上调节权重。

### 第 4 步：在自定义任务上执行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

应在具有*代表性*的任务子集上运行候选模型。不要只相信排行榜名次——你的领域很重要。

### 第 5 步：从零手写余弦相似度

见 `code/main.py`。其中使用仅依赖标准库的平均哈希技巧嵌入。它无法与 Transformer 嵌入竞争，却展示了基本流程：分词 → 向量 → 归一化 → 点积。

## 陷阱

- **查询与文档使用相同模式。** 某些模型（Voyage、Jina-ColBERT）采用非对称编码——查询和文档经过不同路径。务必检查模型卡。
- **缺少前缀。** `bge-*` 模型要求在查询前添加 `"Represent this sentence for searching relevant passages: "`。如果遗漏，召回率会下降 3～5 个百分点。
- **Matryoshka 截断过度。** 从 1536 维截断到 256 维通常安全，截断到 64 维则不安全。必须在评估集上验证。
- **上下文截断。** 大多数模型会悄然截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数不会呈现 p99 延迟。一个 600M 模型也许比 335M 模型高 2 分，却可能让每次查询的成本增加到 3 倍。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 纯英语、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开放权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开放权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集的查询 | 增加 SPLADE 稀疏检索，再与稠密检索进行 RRF 融合 |

2026 年的模式是：从 BGE-M3 或 text-3-large 开始，用 MTEB 在自己的领域上评估；如果领域专用模型领先超过 3 分，再进行替换。

## 交付成果

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 使用 `bge-small-en-v1.5` 以完整维度（384）编码 100 个句子，再使用 Matryoshka 截断至 128 维。在 10 个查询上测量 MRR 降幅。
2. **中等。** 在你所在领域的 500 个段落上，比较 BGE-M3 的稠密、稀疏和 ColBERT 模式。哪种模式的 Recall@10 最高？RRF 融合能否胜过最佳单一模式？
3. **困难。** 在与你的领域最相关的两个任务上，对三个候选模型运行 MTEB。报告 MTEB 分数、100 查询批次的 p99 延迟，以及每百万次查询的成本。选出帕累托最优方案。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 稠密嵌入 | 那个向量 | 每段文本一个定长向量，通过余弦相似度排序。 |
| 稀疏嵌入 | 学习式 BM25 | 每个词表词元一个权重，绝大多数为零，通过端到端训练获得。 |
| 多向量 | ColBERT 风格 | 每个词元一个向量，采用 MaxSim 评分；索引更大，召回率更高。 |
| Matryoshka | 俄罗斯套娃技巧 | 向量的前 N 维本身就构成一个有效的小型嵌入。 |
| MTEB | 那个基准 | 海量文本嵌入基准，发布时有 56 项任务，v2 中超过 100 项。 |
| BEIR | 检索基准 | 18 项零样本检索任务，常用于衡量跨领域稳健性。 |
| 非对称编码 | 查询路径 ≠ 文档路径 | 模型为查询与文档使用不同投影。 |

## 延伸阅读

- [Reimers、Gurevych（2019），Sentence-BERT](https://arxiv.org/abs/1908.10084)——双编码器论文。
- [Muennighoff 等（2022），MTEB：海量文本嵌入基准](https://arxiv.org/abs/2210.07316)——排行榜论文。
- [Chen 等（2024），BGE-M3：多语言、多功能、多粒度](https://arxiv.org/abs/2402.03216)——统一三种模式的模型。
- [Kusupati 等（2022），Matryoshka 表示学习](https://arxiv.org/abs/2205.13147)——维度阶梯训练目标。
- [Santhanam 等（2022），ColBERTv2：通过轻量后期交互实现高效检索](https://arxiv.org/abs/2112.01488)——生产环境中的后期交互。
- [Hugging Face 上的 MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard)——实时排名。
