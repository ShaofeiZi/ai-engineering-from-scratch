# 词性标注与句法分析

> 语法曾一度不受追捧。后来每条大语言模型流水线都需要验证结构化抽取，它又回来了。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 01（文本处理）、阶段 2 · 14（朴素贝叶斯）
**Time:** 约 45 分钟

## 问题

第 01 课提到，词形还原需要词性标签。不知道 `running` 是动词，词形还原器就无法把它还原成 `run`；不知道 `better` 是形容词，就无法把它还原成 `good`。

这个前提背后隐藏着一整个子领域。词性标注负责分配语法类别，句法分析则负责还原句子的树形结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典自然语言处理用了二十年完善这两项任务。随后，深度学习把它们压缩成预训练 Transformer 顶部的词元分类任务，研究界便转向了别处。

但应用界没有离开。每条结构化抽取流水线仍在底层使用词性和依存句法树。大语言模型生成的 JSON 会依据语法约束进行验证，问答系统利用依存句法分析拆解查询，机器翻译质量评估器则会检查句法树的对齐情况。

这些知识值得掌握。本课将介绍标签集、基线方法，以及何时应该停止从零实现、直接调用 spaCy。

## 概念

**词性标注**为每个词元标注语法类别。**Penn Treebank（PTB）**标签集是英语的默认标准，包含 36 个标签，细分程度在普通读者看来甚至有些繁琐：`NN` 表示单数名词，`NNS` 表示复数名词，`NNP` 表示单数专有名词，`VBD` 表示过去时动词，`VBZ` 表示第三人称单数现在时动词，等等。**Universal Dependencies（UD）**标签集更粗粒度（17 个标签），且与语言无关；它已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**会生成一棵树，主要有两种形式：

- **成分句法分析。** 名词短语、动词短语和介词短语相互嵌套，输出是一棵以词为叶节点、以非终结类别（NP、VP、PP）为内部节点的树。
- **依存句法分析。** 每个词都有一个它所依赖的中心词，并标有语法关系。输出是一棵树，其中每条边都是一个（中心词、从属词、关系）三元组。

依存句法分析在 2010 年代胜出，因为它可以自然地泛化到各种语言，尤其是语序自由的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 动手构建

### 第 1 步：最高频标签基线

这是能奏效的最简单词性标注器：对于每个词，预测它在训练数据中最常出现的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线的准确率约为 85%。并不好，但任何严肃模型都不应低于这条底线。

### 第 2 步：二元 HMM 标注器

对整个序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

使用两张表：转移概率（给定前一个标签时当前标签的概率）和发射概率（给定标签时出现某个词的概率）。通过带有拉普拉斯平滑的计数估算二者，再用维特比算法解码（在标签网格上进行动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

二元 HMM 在 Brown 语料库上能达到约 93% 的准确率。从 85% 到 93% 的提升主要来自转移概率——模型学会了 `DET NOUN` 很常见，而 `NOUN DET` 很少见。

### 第 3 步：现代标注器为何能胜过它

转移概率与发射概率都是局部的。它们无法理解 `saw` 在“I bought a saw”中是名词，在“I saw the movie.”中却是动词。带有任意特征（后缀、词形、前后单词、单词本身）的 CRF 能达到约 97%；BiLSTM-CRF 或 Transformer 则能达到约 98% 以上。

这项任务的上限由标注者之间的分歧决定。在 Penn Treebank 上，人类标注者的一致率约为 97%。超过 98% 的模型很可能是在过拟合测试集。

### 第 4 步：依存句法分析概要

从零实现完整的依存句法分析超出了本课范围；权威教材请参考 Jurafsky 与 Martin。需要了解两个经典家族：

- **基于转移的**分析器（arc-eager、arc-standard）像移进—归约分析器一样工作：读取词元，把它们移入栈，再执行创建弧的归约动作。贪心解码速度很快。经典实现是 MaltParser，现代神经版本则是 Chen 与 Manning 的基于转移分析器。
- **基于图的**分析器（Eisner 算法、Dozat-Manning 双仿射模型）会为每条可能的中心词—从属词边评分，再选出最大生成树。速度较慢，但准确率更高。

对于大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读取 `dep` 列，句子的语法结构便会显现出来。

## 学以致用

每个生产级自然语言处理库都会把词性标注器和依存句法分析器作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，并与分词、NER、词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。斯坦福对 CoreNLP 的继任者，在 60 多种语言上达到顶尖水平。
- **trankit**。基于 Transformer，UD 准确率很高。
- **NLTK**。`pos_tag`。可用但较慢，也较老旧，适合教学。

### 这些知识在 2026 年仍然重要的场景

- **词形还原。** 第 01 课需要词性信息才能正确还原词形，始终如此。
- **从大语言模型输出中进行结构化抽取。** 验证生成句子是否遵守语法约束（例如主谓一致、必须存在的修饰语）。
- **基于方面的情感分析。** 依存句法分析可以告诉你哪个形容词修饰哪个名词。
- **查询理解。** “movies directed by Wes Anderson starring Bill Murray”可以通过句法树拆解为结构化约束。
- **跨语言迁移。** UD 标签与依存关系不依赖具体语言，可以对新语言执行零样本结构分析。
- **低算力流水线。** 如果无法部署 Transformer，词性 + 依存分析 + 词典仍能带你走很远。

## 交付成果

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 在小型带标注语料库（例如 NLTK 的 Brown 子集）上使用最高频标签基线，测量它在留出句子上的准确率，验证约 85% 的结果。
2. **中等。** 训练上面的二元 HMM，并报告每个标签的精确率与召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存句法分析，从 1000 个句子的样本中提取主语—动词—宾语三元组。在 50 个手工标注的三元组上评估，并记录抽取会在哪些地方失效（通常是被动语态、并列结构和省略的主语）。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 词性标签 | 词的类型 | 语法类别。PTB 有 36 个，UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用，细分动词时态和名词单复数。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗粒度，与语言无关，是跨语言工作的默认选择。 |
| 依存句法树 | 句子树 | 每个词有一个中心词，每条边带有一种语法关系。 |
| 维特比算法 | 动态规划 | 根据发射概率与转移概率，找出概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky 与 Martin——《语音与语言处理》第 8、18 章](https://web.stanford.edu/~jurafsky/slp3/)——讲解词性与句法分析的权威教材。
- [Universal Dependencies 项目](https://universaldependencies.org/)——每种多语言分析器都使用的跨语言标签集与树库集合。
- [spaCy 语言学特征指南](https://spacy.io/usage/linguistic-features)——`Token` 上每个公开属性的实用参考。
- [Chen 与 Manning（2014），使用神经网络实现快速准确的依存句法分析器](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf)——推动神经句法分析器进入主流的论文。
