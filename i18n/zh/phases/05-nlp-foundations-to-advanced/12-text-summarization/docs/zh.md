# 文本摘要

> 抽取式系统告诉你文档说了什么，生成式系统告诉你作者想表达什么。两类任务，不同陷阱。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 11（机器翻译）
**Time:** 约 75 分钟

## 问题

信息流中出现了一篇 2000 词的新闻文章，你需要用 120 个词概括它。可以从文章中选出最重要的三个句子（抽取式），也可以用自己的话重新表述内容（生成式）。二者都称为摘要，却是完全不同的问题。

抽取式摘要是排序问题。为每个句子评分，返回得分最高的 `k` 个。输出直接逐字取自原文，因此语法一定正确；风险是漏掉分散在全文各处的信息。

生成式摘要是生成问题。Transformer 以输入为条件产生新文本。输出流畅、压缩程度高，却可能虚构源文中不存在的事实；风险是自信地捏造内容。

本课将构建两种方法，并分别说明它们固有的失败模式。

## 概念

![抽取式 TextRank 与生成式 Transformer](../assets/summarization.svg)

**抽取式。** 把文章视作一张图，节点是句子，边表示相似度。在图上运行 PageRank（或类似算法），根据每个句子与其他内容的连接程度进行评分。得分最高的句子组成摘要。经典实现是 **TextRank**（Mihalcea 与 Tarau，2004）。

**生成式。** 在文档—摘要对上微调 Transformer 编码器—解码器（BART、T5、Pegasus）。推理时，模型读取文档，再通过交叉注意力逐词元生成摘要。Pegasus 尤其采用了缺口句子预训练目标，因此即使不进行大量微调，也非常擅长摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）评估。ROUGE-1 与 ROUGE-2 分别计算一元语法和二元语法重叠，ROUGE-L 计算最长公共子序列。分数越高越好，但 40 ROUGE-L 算“良好”，50 算“极佳”。每篇论文都会报告三项指标。请使用 `rouge-score` 包。

```figure
summarize-collapse
```

## 动手构建

### 第 1 步：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

这里有两点值得说明。相似度函数采用经过对数归一化的词语重叠，这是原始 TextRank 的变体；也可以使用 TF-IDF 向量的余弦相似度。阻尼系数 0.85 和迭代次数沿用 PageRank 默认值。

### 第 2 步：使用 BART 生成摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调，可以开箱即用地生成新闻风格摘要。对于其他领域（科研论文、对话、法律），应使用对应的 Pegasus 检查点，或在目标数据上微调。

### 第 3 步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。否则，“running”和“run”会被视为不同词，导致 ROUGE 低估匹配程度。

### 超越 ROUGE（2026 年的摘要评估）

ROUGE 已经主导摘要评估二十年，但到 2026 年，单独使用它已经不够。一项针对自然语言生成论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年前持续普及，如今大多数摘要论文都会将它与 ROUGE 一起报告。
- **BARTScore** 把评估视作生成问题：根据源文，计算预训练 BART 为摘要赋予的似然分数。
- **MoverScore**（上下文嵌入上的地球移动距离）在 2025 年摘要基准中位居首位，因为它比 ROUGE 更能捕捉语义重叠。
- **FactCC** 与**基于问答的忠实度**在 2021～2023 年很常见，现在往往被 **G-Eval** 取代。G-Eval 使用 GPT-4 提示链，通过思维链推理评估连贯性、一致性、流畅性和相关性。
- 当评分标准设计良好时，**G-Eval** 和类似的大语言模型裁判与人类判断的一致率约为 80%。

生产建议：报告 ROUGE-L 以便与历史结果比较，使用 BERTScore 衡量语义重叠，使用 G-Eval 衡量连贯性与事实准确性；并用 50～100 篇经过人工标注的摘要完成校准。

### 第 4 步：事实准确性问题

生成式摘要容易产生幻觉。抽取式摘要直接逐字采用源文，幻觉风险要低得多；但如果源句脱离上下文、已经过时或顺序被打乱，它仍可能误导读者。这是合规相关生产系统仍偏爱抽取式方法的首要原因。

需要知道名称的幻觉类型包括：

- **实体替换。** 源文写“John Smith”，摘要却写“John Brown”。
- **数字漂移。** 源文写“25,000”，摘要却写“25 million”。
- **极性反转。** 源文写“rejected the offer”，摘要却写“accepted the offer”。
- **事实编造。** 源文没有提到 CEO，摘要却声称 CEO 已批准。

有效的评估方法包括：

- **FactCC。** 在源句与摘要句之间的蕴含关系上训练的二元分类器，预测内容是否符合事实。
- **基于问答的事实性评估。** 向问答模型提出答案位于源文中的问题。如果摘要支持不同答案，就标记异常。
- **实体级 F1。** 比较源文与摘要中的命名实体。只出现在摘要中的实体值得怀疑。

对于任何重视事实准确性的用户侧内容（新闻、医疗、法律、金融），抽取式方法是更安全的默认选择。生成式方法必须在流程中加入事实性检查。

## 学以致用

2026 年的技术栈：

| 用例 | 推荐方案 |
|---------|-------------|
| 英语新闻，3～5 句摘要 | `facebook/bart-large-cnn` |
| 科研论文 | `google/pegasus-pubmed` 或经过调优的 T5 |
| 多文档、长篇内容 | 任何上下文达到 32k 以上的大语言模型，通过提示调用 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式、从机制上降低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

如果算力不是限制，长上下文大语言模型在 2026 年通常能胜过专用模型。代价是成本与可复现性；专用模型的输出更一致。

## 交付成果

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank，把得分最高的三个句子与参考摘要比较，并测量 ROUGE-L。在 CNN/DailyMail 风格文章上，你应当看到 30～45 的 ROUGE-L。
2. **中等。** 实现实体级事实性检查：从源文与摘要中提取命名实体（使用 spaCy），计算摘要对源实体的召回率，以及摘要实体相对于源文的精确率。高精确率、低召回率意味着安全但简略；低精确率意味着出现了虚构实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与大语言模型（Claude 或 GPT-4）。报告 ROUGE-L、事实准确性（通过实体 F1）和每篇摘要的成本，并记录各自胜出的场景。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 | 选择句子 | 逐字返回源文中的句子，不会凭空生成内容。 |
| 生成式 | 重写 | 以源文为条件生成新文本，可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 n 元语法/最长公共子序列重叠。 |
| TextRank | 基于图的抽取方法 | 在句子相似度图上运行 PageRank。 |
| 事实准确性 | 内容是否正确 | 摘要中的陈述是否得到源文支持。 |
| 幻觉 | 编造的内容 | 摘要中出现源文无法支持的内容。 |

## 延伸阅读

- [Mihalcea 与 Tarau（2004），TextRank：让文本井然有序](https://aclanthology.org/W04-3252/)——经典的抽取式论文。
- [Lewis 等（2019），BART：用于自然语言生成、翻译和理解的去噪序列到序列预训练](https://arxiv.org/abs/1910.13461)——BART 论文。
- [Zhang 等（2019），PEGASUS：使用抽取式缺口句子进行预训练](https://arxiv.org/abs/1912.08777)——Pegasus 与缺口句子目标。
- [Lin（2004），ROUGE：自动摘要评估工具包](https://aclanthology.org/W04-1013/)——ROUGE 论文。
- [Maynez 等（2020），生成式摘要中的忠实度与事实准确性](https://arxiv.org/abs/2005.00661)——事实性研究综述。
