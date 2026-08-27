# 词嵌入——从零实现 Word2Vec

> 观其友，知其词。围绕这个想法训练一个浅层网络，几何结构便会自然涌现。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 02（BoW + TF-IDF）、阶段 3 · 03（从零实现反向传播）
**Time:** 约 75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，却不知道它们的含义几乎相同。在 `dog` 上训练的分类器无法泛化到谈论 `puppy` 的评论。你可以列举同义词来勉强弥补，但遇到罕见词、领域术语和任何未曾预料的语言时，这种方法都会失效。

我们需要一种表示，让 `dog` 和 `puppy` 在空间中彼此靠近，让 `king - man + woman` 落在 `queen` 附近，也让在 `dog` 上训练的模型无须额外成本便能向 `puppy` 传递一些信号。

Word2Vec 为我们提供了这样的空间。它是一个双层神经网络，于 2013 年发表，并在万亿词元规模上训练。它的架构简单得几乎令人难为情，结果却重塑了自然语言处理领域长达十年。

## 概念

**分布假说**（Firth，1957）：“观其友，知其词。”如果两个词出现在相似的上下文中，它们的含义很可能也相似。

Word2Vec 有两种形式，二者都利用了这一思想。

- **Skip-gram。** 给定中心词，预测周围的词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练较慢，却更善于处理罕见词，因此成为默认选择。

这个网络只有一个隐藏层，并且没有非线性。输入是覆盖整个词表的独热向量，输出是覆盖整个词表的 softmax。训练完成后，丢弃输出层；隐藏层的权重就是词嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

关键技巧在于：对 10 万个词计算 softmax 成本高得难以承受。Word2Vec 使用**负采样**，把问题变成二元分类任务：预测“这个上下文词是否出现在该中心词附近，是还是不是”。每个训练对只需采样少量负例（未共同出现的词），无须在整个词表上计算 softmax。

```figure
word-vector-arithmetic
```

## 动手构建

### 第 1 步：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每个（中心词，上下文词）对都是一个正训练样本。

### 第 2 步：嵌入表

使用两个矩阵。`W` 是中心词嵌入表（最终保留的矩阵）。`W'` 是上下文词表（通常丢弃，有时会与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

使用较小的随机值初始化。1 万词表、100 维是实际可用的规模；教学时，50 个词、16 维就足以观察几何结构。

### 第 3 步：负采样目标

对于每个正样本对 `(center, context)`，从词表中随机采样 `k` 个词作为负例。训练模型，使正例的点积 `W[center] · W'[context]` 较高，负例的点积较低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇的公式是：正样本对使用逻辑损失（希望 sigmoid 接近 1），负样本对也使用逻辑损失（希望 sigmoid 接近 0）。梯度会流向两张表。完整推导见原始论文；如果想真正记住它，最好拿纸笔亲自推导一次。

### 第 4 步：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上训练足够多轮后，共享上下文的词会得到相似的中心词嵌入。在玩具语料库上，这种效果隐约可见；扩展到数十亿词元后，效果会非常显著。

### 第 5 步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这并不是因为模型知道王权是什么，而是因为向量 `(king - man)` 捕捉到了类似“皇室”的属性，把它加到 `woman` 上，就会落到皇室女性所处的区域附近。

## 学以致用

从零编写 Word2Vec 是为了学习。生产级自然语言处理会使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

在实际工作中，你几乎不会亲自训练 Word2Vec，而是下载预训练向量。

- **GloVe**——斯坦福提出的共现矩阵分解方法。提供 50、100、200 和 300 维检查点，通用覆盖较好。第 04 课会专门介绍 GloVe。
- **fastText**——Facebook 对 Word2Vec 的扩展，会嵌入字符 n 元语法。它能组合子词，从而处理词表外单词。见第 04 课。
- **Google News 预训练 Word2Vec**——300 维、300 万词词表，发布于 2013 年，至今每天仍有人下载。

### Word2Vec 在 2026 年仍能胜出的场景

- 轻量级领域检索。在笔记本电脑上用一小时训练医学摘要，就能得到通用模型无法捕捉的专业向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它，可以得到性别中性轴；公平性研究至今仍在使用这种方法。
- 可解释性。100 维足够小，可以用 PCA 或 t-SNE 绘图，亲眼观察聚类形成。
- 必须在无 GPU 设备端进行推理的场景。Word2Vec 查找只需读取矩阵中的一行。

### Word2Vec 的局限

首先是多义词壁垒。`bank` 只有一个向量，`river bank` 和 `financial bank` 共用它；`table` 的电子表格含义和家具含义也共用一个向量。下游分类器无法从向量中区分这些词义。

上下文嵌入（ELMo、BERT 以及此后的每一种 Transformer）解决了这个问题：它们根据周围上下文，为单词的每次出现生成不同的向量。从 Word2Vec 到 BERT 的跃迁，就是从静态表示到上下文表示。阶段 7 会介绍 Transformer 部分。

另一个问题是词表外单词。如果训练数据中从未出现 `Zoomer-approved`，Word2Vec 就没见过它，也没有后备方案。fastText 通过子词组合修复了这一点（第 04 课）。

## 交付成果

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个微型语料库（20 个关于猫狗的句子）上运行训练循环。训练 200 轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 返回结果的前三名中包含 `dog`。如果没有，就增加训练轮数或词表规模。
2. **中等。** 增加高频词降采样。频率高于 `10^-5` 的词，会按与其频率成比例的概率从训练对中丢弃。测量它对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两条偏见轴：`he - she` 和 `doctor - nurse`。把职业词投影到两条轴上，并报告哪些职业的偏见差距最大。这正是公平性研究者会使用的探测方法。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 词嵌入 | 用向量表示单词 | 从上下文中学习到的稠密低维表示，通常为 100～300 维。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但更适合罕见词。 |
| 负采样 | 训练捷径 | 把整个词表上的 softmax 替换为针对 `k` 个随机词的二元分类。 |
| 静态嵌入 | 每个词一个向量 | 无论上下文如何都使用同一向量，因此无法处理多义词。 |
| 上下文嵌入 | 上下文敏感向量 | 根据周围词语为每次出现生成不同的向量，即 Transformer 的输出。 |
| OOV | 词表外 | 训练中未曾出现的词。Word2Vec 无法为其生成向量。 |

## 延伸阅读

- [Mikolov 等（2013），词与短语的分布式表示及其组合性](https://arxiv.org/abs/1310.4546)——介绍负采样的论文，简短易读。
- [Rong, X.（2014），word2vec 参数学习详解](https://arxiv.org/abs/1411.2738)——如果原始论文的数学看起来太密集，这是最清晰的梯度推导。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html)——真正可用的生产训练配置。
