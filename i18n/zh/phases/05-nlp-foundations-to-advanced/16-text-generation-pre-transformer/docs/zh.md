# Transformer 之前的文本生成——N 元语言模型

> 如果一个词令人意外，说明模型还不够好。困惑度把意外量化，平滑则让这个数值保持有限。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 01（文本处理）、阶段 2 · 14（朴素贝叶斯）
**Time:** 约 45 分钟

## 问题

在 Transformer、RNN 和词嵌入出现之前，语言模型通过统计一个词跟在前 `n-1` 个词之后的频率来预测下一个词。统计“the cat”→“sat”出现 47 次，“the cat”→“jumped”出现 12 次，“the cat”→“refrigerator”出现 0 次，再归一化成概率分布。

这就是 n 元语言模型。从 1980 年到 2015 年，每个语音识别器、拼写检查器和基于短语的机器翻译系统都依赖它。如今，当你需要廉价的设备端语言建模时，它仍在发挥作用。

真正有趣的问题是如何处理未见过的 n 元语法。原始计数模型会为所有未在训练中出现过的模式分配零概率。这是灾难性的，因为句子很长，几乎每个长句都会包含至少一个未见序列。五十年的平滑研究解决了这个问题，结果便是 Kneser-Ney 平滑；现代深度学习也继承了这种重视实证的传统。

## 概念

![N 元模型：计数、平滑、生成](../assets/ngram.svg)

### 预测游戏

在任何这类机制出现之前，有一项实验定义了什么是语言模型。遮住一个英语句子的下一个字母，让参与者逐个猜测，直到猜中为止，并记录猜测次数；对几百个字母重复这一过程。

猜测次数并非无关紧要的统计。它们是文本的一种无损重新编码：把次数序列交给第二位采用相同策略的猜测者，对方就能还原每个字母，因为他在每个位置都知道猜测的先后顺序。一条消息若能用更少符号重新编码，每个符号所携带的信息就更少，因此猜测次数的统计结果给出了英语熵的上限。

Shannon 在 1951 年进行了这项实验，得出的数字至今仍支配着这个领域。由 27 个符号（26 个字母加空格）组成的字母表，每个字母最多可以携带 `log2(27) ≈ 4.75` 比特。拥有 100 个字母上下文的人类猜测者，只需每个字母 0.6～1.3 比特。英语大约四分之三的字符都近乎是必然选择。早在任何模型有能力学习这种结构之前，人们已经测量出了模型需要学习的结构。

此后的每个语言模型都在机械地玩这场游戏，本课的每个评估数字都在为它计分：

- **交叉熵损失**是模型表示每个符号平均需要的比特数。训练语言模型，本质上就是尽量降低它在猜测游戏中的分数。
- **困惑度**是 `2^bits`（或 `e^nats`）：模型完成猜测后仍面对的分支因子。在 27 个符号上均匀猜测时，困惑度为 27；每个字母只需 1 比特的玩家，困惑度为 2。
- **上下文长度就是玩家的记忆。** 三元模型只带两个词元的记忆，Transformer 则带着 10 万词元玩同一场游戏。规则从未改变，只是玩家变强了。

需要留意一次单位切换：这场游戏以比特（`log2`）衡量每个字母，而下面的 n 元语法公式使用自然对数，以纳特衡量每个词元——由于以纳特计算的困惑度 `e^H` 等于以比特计算的 `2^H`，二者只是用不同单位表示同一个测量结果。

```figure
prediction-game
```

**N 元概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（三元模型通常取 3，四元模型取 4），再根据计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何训练中未出现的 n 元语法都会得到零概率。2007 年针对 Brown 语料库的一项研究发现，即使使用四元模型，留出数据中仍有 30% 的四元语法没有在训练中出现。如果不进行平滑，就无法在任何真实文本上评估。

**平滑方法，按复杂程度排列：**

1. **拉普拉斯（加一）。** 为每个计数加 1。简单，但在罕见事件上表现很差。
2. **Good-Turing。** 根据频数的频数，把概率质量从高频事件重新分配给未见事件。
3. **插值。** 使用可调权重组合 n 元、(n-1) 元等不同阶数的估计。
4. **回退。** 如果 n 元语法的计数为零，就回退到 (n-1) 元语法。Katz 回退对此进行归一化。
5. **绝对折扣。** 从所有计数中减去固定折扣 `D`，把释放的概率质量重新分配给未见事件。
6. **Kneser-Ney。** 使用绝对折扣，并巧妙地选择低阶模型：不使用原始频率，而使用*延续概率*（某个词出现在多少种上下文中）。

Kneser-Ney 的洞见非常深刻。“San Francisco”是常见二元语法，一元词“Francisco”却几乎只出现在“San”后面。朴素绝对折扣会因为“Francisco”计数很高而赋予它较高的一元概率。Kneser-Ney 注意到“Francisco”只出现在一种上下文中，于是相应降低它的延续概率。结果是，以“Francisco”结尾的新二元语法会得到恰当的低概率。

**评估：困惑度。** 在留出测试集上，逐词平均负对数似然的指数。越低越好。困惑度为 100，意味着模型的迷茫程度相当于从 100 个词中均匀选择一个。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 动手构建

### 第 1 步：三元语法计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是一组已经分词的句子，输出是 n 元语法计数与上下文计数。`<s>` 和 `</s>` 是句子边界。

### 第 2 步：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

为每个计数加 1。这样可以平滑概率，却会为未见事件分配过多质量，同时损害已见罕见事件。

### 第 3 步：Kneser-Ney（二元、插值式）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

这里有三个关键部分。`continuation_prob` 表示“这个词出现在多少种不同上下文中？”（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出的概率质量，用于给回退项加权。最终概率等于经过折扣的主项，加上经过加权的延续项。

### 第 4 步：通过采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率成比例采样。使用不同随机种子时，输出始终不同。如果希望得到类似束搜索的输出，就在每一步选择 argmax（贪心），再加入一个小型随机性旋钮（温度）。

### 第 5 步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown 语料库上，调优良好的四元 KN 模型可以达到约 140 的困惑度，Transformer 语言模型在同一测试集上则能达到 15～30，差距约为 10 倍。这就是整个领域转向新架构的原因。

## 学以致用

- **经典自然语言处理教学。** 这是学习平滑、最大似然估计和困惑度最清晰的途径。
- **KenLM。** 生产级 n 元语法库，用于低延迟语音与机器翻译系统的重打分。
- **设备端自动补全。** 键盘里至今仍在使用三元模型。
- **基线。** 在宣布神经语言模型表现良好之前，务必先计算 n 元语言模型的困惑度。如果 Transformer 没有大幅胜过 KN，一定有哪里出了问题。

## 交付成果

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在包含 1000 个莎士比亚句子的语料库上训练三元语言模型，生成 20 个句子。它们会在局部看似合理，整体却缺乏连贯性。这是经典演示。
2. **中等。** 在莎士比亚留出集上实现 KN 模型的困惑度计算，并与拉普拉斯方法比较。你应当看到 KN 的困惑度低 30%～50%。
3. **困难。** 构建三元拼写纠正器：给定拼错的词及其上下文，生成候选修正，并按照语言模型中的上下文概率排序。在公开的 Birkbeck 拼写语料库上评估。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| N 元语法 | 词序列 | 由连续 `n` 个词元组成的序列。 |
| 平滑 | 避免零值 | 重新分配概率质量，使未见事件得到非零概率。 |
| 困惑度 | 语言模型质量指标 | 在留出数据上计算 `exp(-average log-prob)`，越低越好。 |
| 回退 | 使用更短上下文兜底 | 三元语法计数为零时使用二元语法，Katz 回退对此进行了形式化。 |
| Kneser-Ney | 最佳 n 元语法平滑 | 绝对折扣 + 用于低阶模型的延续概率。 |
| 延续概率 | KN 专用概念 | `P(w)` 按 `w` 出现的上下文数量加权，而不是按原始计数加权。 |
| 文本熵 | 每个符号的信息量 | 给定上下文后，编码下一个符号平均所需的比特数。Shannon 在 1951 年估算：对于最多提供 100 个字母上下文的印刷英语，每个字母为 0.6～1.3 比特；这一结果在任何模型出现前就已经测得。 |

## 延伸阅读

- [Shannon（1951），印刷英语的预测与熵](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)——定义了至今所有语言模型都在优化的目标的猜测游戏实验。
- [Jurafsky 与 Martin——《语音与语言处理》第 3 章（2026 草稿）](https://web.stanford.edu/~jurafsky/slp3/3.pdf)——n 元语言模型与平滑的权威讲解。
- [Chen 与 Goodman（1998），语言建模平滑技术的实证研究](https://dash.harvard.edu/handle/1/25104739)——确立 Kneser-Ney 为最佳 n 元语法平滑方法的论文。
- [Kneser 与 Ney（1995），用于 M 元语言建模的改进回退方法](https://ieeexplore.ieee.org/document/479394)——KN 原始论文。
- [KenLM](https://kheafield.com/code/kenlm/)——快速的生产级 n 元语言模型，2026 年仍用于延迟敏感型应用。
