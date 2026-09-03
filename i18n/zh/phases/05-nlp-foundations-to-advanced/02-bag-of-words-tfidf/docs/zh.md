# 词袋、TF-IDF 与文本表示

> 先计数，再思考。到 2026 年，在边界清晰的任务上，TF-IDF 仍能胜过嵌入。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 01（文本处理）、阶段 2 · 02（从零实现线性回归）
**Time:** 约 75 分钟

## 问题

模型需要数字，而你手中只有字符串。

每条自然语言处理流水线都必须回答同一个问题：如何把长度不定的词元流转换成分类器能够接收的定长向量？这个领域最先采用的答案，是能奏效的最简单方案：数一数每个词，组成一个向量。

这种向量支撑的生产级自然语言处理系统，比任何嵌入模型都多：垃圾邮件过滤器、主题分类器、日志异常检测、搜索排序（BM25 之前）、第一波情感分析系统，以及学术界最初十年的自然语言处理基准。2026 年的从业者面对范围狭窄的分类任务时，仍会优先尝试它。它速度快、可解释；当词语是否出现就是关键时，它的效果往往与拥有 4 亿参数的嵌入模型难分伯仲。

本课先从零构建词袋和 TF-IDF，再用三行代码展示 scikit-learn 如何完成同样的工作，最后指出何种失败模式会促使你改用嵌入。

## 概念

**词袋（BoW）**会丢弃顺序。对于每篇文档，统计词表中的每个词出现了多少次。向量长度等于词表大小，第 `i` 个位置就是第 `i` 个词的计数。

**TF-IDF** 会重新加权词袋。如果一个词出现在每篇文档中，它就没有区分度，因此应降低权重。如果一个词在整个语料库中罕见，却频繁出现在某篇文档中，它就是有效信号，因此应提高权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中，`TF` 是词语在文档中的频率，`df` 是文档频率（包含该词的文档数），`N` 是文档总数。`log` 能限制高频词的权重范围。

二者的关键性质是都会生成坐标轴含义明确的稀疏向量。你可以查看训练后分类器的权重，直接读出哪些词会把文档推向哪个类别。对于 768 维的 BERT 嵌入，你做不到这一点。

```figure
bow-tfidf
```

## 动手构建

### 第 1 步：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：已经分词的文档列表（任何词级分词器都可以；本课的 `code/main.py` 使用简化的小写版本）。输出：`{word: index}` 字典。稳定的插入顺序意味着索引 0 对应第一篇文档中最先出现的词。不同工具的约定不同；scikit-learn 会按字母顺序排序。

### 第 2 步：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

每一行代表一篇文档，每一列代表一个词表索引。`[i][j]` 表示“词 `j` 在文档 `i` 中出现了多少次”。文档 1 中 `cat` 出现两次，所以计数为 2；文档 0 没有 `ran`，所以计数为 0。

### 第 3 步：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

这里有两种值得点明的平滑技巧。`(n+1)/(d+1)` 可以避免 `log(x/0)`。末尾的 `+1` 则保证出现在每篇文档中的词仍有 1（而非 0）的 IDF，与 scikit-learn 的默认行为一致。其他实现会直接使用 `log(N/df)`。两者都可行，平滑版本更温和。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在全部三篇中，所以 IDF 较低；`dog` 只出现一次，所以 IDF 较高。得到的向量是稀疏的（大部分元素都很小），有区分力的词会凸显出来。

### 第 5 步：对每行进行 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不归一化，较长的文档会得到较大的向量，并支配相似度分数。L2 归一化把每篇文档都放到单位超球面上。此时，两行之间的余弦相似度就等于它们的点积。

## 学以致用

scikit-learn 提供了生产级版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用中完成分词、构建词表和词袋。`TfidfVectorizer` 再加入 IDF 加权与 L2 归一化。二者都返回稀疏矩阵。对于 10 万篇文档，稠密版本无法装入内存；在分类器明确要求稠密输入之前，都应保持稀疏形式。

足以改变一切的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 加入二元语法，通常能提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 篇文档中的词，可以精简噪声数据的词表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词，无须硬编码列表即可近似移除停用词。 |
| `stop_words="english"` | scikit-learn 内置的停用词表。是否使用取决于任务——情感分析不应删除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`，适合某个词在单篇文档中反复出现的情况。 |

### TF-IDF 在何时仍能胜出（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。这些任务看重词语是否出现，而非细微语义。
- 小数据场景（数百个带标签样本）。TF-IDF 加逻辑回归不需要预训练成本。
- 任何重视延迟的系统。TF-IDF 加线性模型可以在微秒内给出答案，而让文档通过 Transformer 生成嵌入需要 10～100 毫秒。
- 必须解释预测结果的系统。检查分类器系数即可；权重最高的正向词就是预测理由。

### TF-IDF 在何时失效

首先是语义盲区。考虑下面两篇文档：

- “这部电影一点也不好。（The movie was not good at all.）”
- “这部电影非常精彩。（The movie was excellent.）”

一篇是负面评价，另一篇是正面评价。它们的 TF-IDF 重叠部分恰好是 `{the, movie, was}`。词袋分类器必须记住 `not` 靠近 `good` 时会反转标签。数据足够多时，它可以学到这一点，却永远不会像理解句法的模型那样自然。

另一个问题是推理时出现词表外单词。在 IMDb 影评上训练的词袋模型，面对训练中从未出现过的 `Zoomer-approved` 时无从下手。子词嵌入（第 04 课）可以处理，TF-IDF 不行。

### 混合方案：TF-IDF 加权嵌入

2026 年中等规模数据分类的务实默认方案，是把 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

嵌入提供语义能力，TF-IDF 则强调罕见词。分类器在池化后的向量上训练。当带标签样本少于约 5 万个时，这种方案在情感、主题和意图分类上优于任何一种单独方案。

## 交付成果

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档的得分为 1.0，词表完全不相交的文档得分为 0.0。
2. **中等。** 增加 `n-gram` 对 `bag_of_words` 的支持。参数 `n` 用于生成 `n` 元语法计数。测试 `n=2` 作用于 `["the", "cat", "sat"]` 时，会为 `["the cat", "cat sat"]` 生成二元语法计数。
3. **困难。** 使用 GloVe 100 维向量构建上面的 TF-IDF 加权嵌入混合方案（只下载一次并缓存）。在 20 Newsgroups 数据集上，把它的分类准确率与纯 TF-IDF 和普通均值池化嵌入比较，并报告各种方案分别在哪些情况下胜出。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一篇文档中词表词的计数，会丢弃顺序。 |
| TF | 词频 | 一个词在文档中的计数，可以按文档长度归一化。 |
| DF | 文档频率 | 至少包含一次该词的文档数量。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`，降低随处可见词语的权重。 |
| 稀疏向量 | 大部分为零 | 词表通常有 1 万～10 万个词，其中绝大多数不会出现在某一篇文档中。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn——从文本中提取特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)——权威 API 参考，以及每个参数的说明。
- [Salton, G. 与 Buckley, C.（1988），自动文本检索中的词项加权方法](https://www.sciencedirect.com/science/article/pii/0306457388900210)——让 TF-IDF 在十年间成为默认方案的论文。
- [“为什么 TF-IDF 仍能胜过嵌入”——Ashfaque Thonikkadavan（Medium）](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2)——2026 年对这种老方法何时胜出及其原因的分析。
