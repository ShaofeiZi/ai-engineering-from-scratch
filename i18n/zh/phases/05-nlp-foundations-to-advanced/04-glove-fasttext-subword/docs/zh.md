# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练一个嵌入。GloVe 分解共现矩阵，FastText 嵌入词的组成部分，BPE 则搭起了通往 Transformer 的桥梁。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 03（从零实现 Word2Vec）
**Time:** 约 45 分钟

## 问题

Word2Vec 留下了两个悬而未决的问题。

首先，另有一条研究路线直接分解共现矩阵（LSA、HAL），而不是在线执行 Skip-gram 更新。Word2Vec 的迭代方法真的从根本上更优，还是差异仅源自两种方法处理计数的方式？**GloVe** 给出了答案：只要精心选择损失函数，矩阵分解便能达到或超过 Word2Vec 的效果，而且训练成本更低。

其次，两种方法都无法处理从未见过的词。`Zoomer-approved`、`dogecoin`、上周刚创造的任何专有名词，以及罕见词根的每一种屈折形式。**FastText** 通过嵌入字符 n 元语法解决了这个问题：一个词等于其各组成部分（包括词素）之和，因此即使是词表外单词，也能得到合理的向量。

第三，Transformer 出现后，问题再次发生变化。词级词表最多只能容纳约一百万个条目，而真实语言远比这开放。**字节对编码（BPE）**及其同类方法学习由高频子词单元构成的词表，并覆盖所有输入，从而解决了这个问题。每一种现代大语言模型的现代分词器都是子词分词器。

本课将逐一讲解这三种方法，然后说明在不同情况下应选择哪一种。

## 概念

**GloVe（全局向量）。** 构建词—词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现了多少次。训练向量，使 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权，避免高频词对支配结果。完成。

**FastText。** 一个词由其字符 n 元语法与这个词本身相加而成。`where` 会变成 `<wh, whe, her, ere, re>, <where>`。词向量就是这些组成向量之和。训练方式与 Word2Vec 相同。好处是：未见过的词（`whereupon`）也可以由已知的 n 元语法组合出来。

**BPE（字节对编码）。** 从单个字节（或字符）组成的词表开始，统计语料库中的每一对相邻元素，把出现最频繁的一对合并为一个新词元，重复 `k` 轮。最终得到包含 `k + 256` 个词元的词表：高频序列（`ing`、`tion`、`the`）成为单一词元，罕见词则被拆成熟悉的片段。任何句子都能被分词。

```figure
n5-subword-merge
```

## 动手构建

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

这里有两个值得点明的关键环节。加权函数 `f(x) = (x/x_max)^alpha` 会降低极高频词对（如 `(the, and)`）的权重，使它们不会支配损失。最终嵌入是 `W`（中心词）与 `W_tilde`（上下文）两张表之和。把二者相加是论文中公开的一项技巧，效果通常优于只使用其中一张表。

### FastText：感知子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词都用它的一组 n 元语法表示（通常为 3～6 个字符）。词嵌入是各个 n 元语法嵌入之和。在 Skip-gram 训练中，只需把 Word2Vec 原本使用的单一向量替换为它。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要它的一部分 n 元语法已知，你仍能得到向量。`whereupon` 共享 `<wh`、`her`、`ere` 和 `<where` 等已在 `where` 中出现的片段，因此二者会落在彼此附近。

### BPE：学习式子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮会合并最常见的相邻对。经过足够多轮后，高频子串（`low`、`est`、`tion`）会成为单一词元，罕见词则被清晰地拆开。

真正的 GPT / BERT / T5 分词器会学习 3 万～10 万次合并。结果是：任何文本都可以转换成长度受控的已知 ID 序列，不再出现 OOV。

## 学以致用

在实际工作中，你很少亲自训练这些模型，而是加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代，可以这样使用 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀表示单词边界（GPT-2 的约定）。每一种现代分词器都属于 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 预训练通用词向量，不要求容忍 OOV | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误、新词或形态丰富的语言 | FastText |
| 任何送入 Transformer 的内容（训练或推理） | 使用模型自带的分词器，绝不要替换。 |
| 从零训练自己的语言模型 | 先在语料库上训练 BPE 或 SentencePiece 分词器 |
| 使用线性模型的生产文本分类 | 仍然选择 TF-IDF，见第 02 课。 |

## 交付成果

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`，计算两个 n 元语法集合的 Jaccard 重叠度。你应当能看到大量共享片段（`pla`、`lay`、`play`），这就是 FastText 能在形态变体间良好迁移的原因。
2. **中等。** 扩展 `learn_bpe` 以跟踪词表增长。绘制每个语料字符对应的词元数随合并次数变化的曲线。你会看到压缩率起初快速提高，随后渐近于每个词元约 2～3 个字符。
3. **困难。** 在莎士比亚全集上训练一个执行 1000 次合并的 BPE。比较常见词和罕见专有名词的分词结果，测量合并前后每个词的平均词元数，并写下令你意外的发现。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵 | 词—词频率表 | `X[i][j]` = 词 `j` 在词 `i` 周围的窗口中出现了多少次。 |
| 子词 | 单词的一部分 | 字符 n 元语法（FastText）或学习到的词元（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 反复合并最频繁的相邻对，直到词表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 无法处理，FastText 和 BPE 可以。 |
| 字节级 BPE | 对原始字节执行 BPE | GPT-2 的方案。词表从 256 个字节开始，因此任何内容都不会成为 OOV。 |

## 延伸阅读

- [Pennington、Socher、Manning（2014），GloVe：词表示的全局向量](https://nlp.stanford.edu/pubs/glove.pdf)——GloVe 论文，只有七页，至今仍是对损失函数最好的推导。
- [Bojanowski 等（2017），以子词信息丰富词向量](https://arxiv.org/abs/1607.04606)——FastText。
- [Sennrich、Haddow、Birch（2016），使用子词单元进行稀有词神经机器翻译](https://arxiv.org/abs/1508.07909)——把 BPE 引入现代自然语言处理的论文。
- [Hugging Face 分词器概览](https://huggingface.co/docs/transformers/tokenizer_summary)——BPE、WordPiece 与 SentencePiece 在实践中的具体区别。
