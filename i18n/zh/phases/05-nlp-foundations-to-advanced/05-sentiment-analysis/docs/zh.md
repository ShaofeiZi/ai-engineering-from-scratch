# 情感分析

> 这是自然语言处理的经典任务。经典文本分类中需要了解的大部分知识，都能在这里看到。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 02（BoW + TF-IDF）、阶段 2 · 14（朴素贝叶斯）
**Time:** 约 75 分钟

## 问题

“这顿饭不算好。”是正面还是负面？

情感分析听起来很简单。评论者说自己喜欢或不喜欢某样东西，你只需为句子打上标签。它之所以成为自然语言处理的经典任务，是因为每个貌似简单的例子背后都藏着难题。否定会反转含义，讽刺也会表达相反意思。“一点也不差”虽然包含两个带负面色彩的词，整体却是正面的。表情符号携带的信号可能比周围文本更强。领域词汇也很重要（音乐评论中的 `tight` 与服装评论中的 `tight` 含义不同）。

情感分析是经典自然语言处理的实践实验室。如果你理解每一种朴素基线为何都有特定的失败模式，就能理解每一种更丰富的模型为何会被发明。本课将从零构建一个朴素贝叶斯基线，再加入逻辑回归，并说明那些会让生产级情感分析上升为合规级问题的陷阱。

## 概念

经典情感分析由两个步骤组成。

1. **表示。** 把文本转换成特征向量，可以使用 BoW、TF-IDF 或 n 元语法。
2. **分类。** 在带标签样本上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是能奏效的最简单模型。它假设给定标签后，每个特征都相互独立；根据计数估算 `P(word | positive)` 和 `P(word | negative)`，推理时再将概率相乘。这种“朴素”的独立性假设错得可笑，结果却强得惊人。原因在于：面对稀疏文本特征和中等规模数据，分类器更关心每个词倾向于哪一类，而不是倾向程度有多大。

逻辑回归不再采用独立性假设。它为每个特征学习一个权重，其中也包括负权重。作为二元语法特征的 `not good` 会得到负权重。对于从未标注过的二元语法，朴素贝叶斯无法做到这一点。

```figure
sentiment-logits
```

## 动手构建

### 第 1 步：一个真实的微型数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

数据集特意保持很小。实际工作会使用数万个样本（IMDb、SST-2、Yelp polarity），但数学原理完全相同。

### 第 2 步：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）就是拉普拉斯平滑。如果没有它，某类别中从未出现过的词概率为零，取对数时就会出错。实际应用中常用 `alpha=0.01`，`alpha=1.0` 则是教学时的默认值。

### 第 3 步：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里十分重要。文本特征是稀疏的，不使用 L2 时，模型会死记训练样本。可以从 `0.01` 开始，再进行调优。

### 第 4 步：处理否定（失败模式）

考虑“not good”和“not bad”。词袋分类器只会看到 `{not, good}` 和 `{not, bad}`，并根据哪一组在训练数据中出现得更多来学习。二元语法分类器则会看到 `not_good` 与 `not_bad`，把它们当作不同特征学习。通常这样就足够了。

如果不能使用二元语法，还有一种更粗糙但有效的修复方法：**否定范围标记**。从否定词开始，为后续词元加上 `NOT_` 前缀，直到遇到下一个标点。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在，`good` 和 `NOT_good` 是两个不同的特征，分类器可以为它们赋予方向相反的权重。只需三行预处理代码，就能在情感分析基准上取得可测量的准确率提升。

### 第 5 步：真正重要的评估指标

类别不平衡时，仅看准确率会产生误导。真实情感语料库通常有 70%～80% 的正面样本，或 70%～80% 的负面样本；永远预测多数类别的分类器也能得到 80% 准确率，却毫无价值。以下各项都应报告：

- **逐类别精确率与召回率。** 每个类别各一组，再计算宏平均，以得到尊重类别平衡的单一数值。
- **Macro-F1（不平衡数据的首要指标）。** 各类别 F1 分数的等权平均。类别不平衡时应使用它，而不是准确率。
- **Weighted-F1（备选指标）。** 与宏平均相同，但按类别频率加权。如果类别不平衡本身具有业务含义，应与 Macro-F1 一并报告。
- **混淆矩阵。** 原始计数。在相信任何标量指标前都要查看它；它能揭示模型混淆了哪一对类别。
- **逐类别错误样本。** 每个类别抽取 5 个错误预测并亲自阅读。没有任何东西能取代阅读真实错误。

对于严重不平衡的数据（比例超过 95:5），应报告 **AUROC** 和 **AUPRC**，而不是准确率。AUPRC 对少数类别更敏感，而少数类通常正是你真正关心的对象（垃圾邮件、欺诈、罕见情感）。

**需要避免的常见错误。** 在不平衡数据上报告 Micro-F1 而不是 Macro-F1，会得到一个看似很高的数值，因为它由多数类主导。Macro-F1 会迫使你正视少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 学以致用

scikit-learn 用六行代码就能正确完成这件事。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

请留意三点。`stop_words=None` 会保留否定词。`ngram_range=(1, 2)` 会加入二元语法，使 `not_good` 成为特征。`sublinear_tf=True` 会削弱重复词的影响。在 SST-2 上，这三个参数足以拉开准确率 75% 的基线与准确率 85% 的基线之间的差距。

### 何时应改用 Transformer

- 讽刺检测。经典模型在这里一定会失败。
- 情感倾向会在文档中途变化的长篇评论。
- 基于方面的情感分析。“Camera was great but battery was terrible.”你需要把情感归因到具体方面，只能使用 Transformer 或结构化输出模型。
- 非英语、低资源语言。多语言 BERT 可以免费提供零样本基线。

如果你需要以上任意能力，请直接跳到阶段 7（深入 Transformer）。否则，基于 TF-IDF 的朴素贝叶斯或逻辑回归，再加上二元语法和否定处理，就是你在 2026 年的生产基线。

### 再谈可复现性陷阱

重新训练情感模型是常规操作，重新评估却不是。论文中的准确率使用特定的数据划分、预处理和分词器。如果没有采用完全相同的流水线，就拿新模型与论文中的基线数值比较，你会得到误导性的差异。一定要在自己的流水线上重新生成基线，而不要直接引用论文中的数字。

## 交付成果

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单。** 把 `apply_negation` 作为预处理步骤加入 scikit-learn 流水线，并测量它在小型情感数据集上带来的 F1 变化。
2. **中等。** 实现类别加权逻辑回归（向 scikit-learn 传入 `class_weight="balanced"`，或自行推导梯度）。在合成的 90:10 类别不平衡数据上测量效果。
3. **困难。** 在情感模型的残差上训练第二个分类器，构建讽刺检测器。记录实验设置。当准确率低于随机水平时提醒读者（二分类讽刺任务的随机水平约为 50%，而大多数首次尝试都会落在这里）。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 极性 | 正面或负面 | 二元标签；有时会扩展为中性或细粒度标签（五星制）。 |
| 基于方面的情感分析 | 逐方面极性 | 把情感归因到文本中提到的具体实体或属性。 |
| 否定范围标记 | 反转附近词元 | 在“not”之后的词元前加 `NOT_`，直到遇到标点。 |
| 拉普拉斯平滑 | 为计数加 1 | 防止朴素贝叶斯中出现概率为零的特征。 |
| L2 正则化 | 收缩权重 | 在损失中加入 `lambda * sum(w^2)`，对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang 与 Lee（2008），意见挖掘与情感分析](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html)——奠基性综述。篇幅虽长，但前四节已经涵盖全部经典内容。
- [Wang 与 Manning（2012），基线与二元语法：简单而有效的情感和主题分类](https://aclanthology.org/P12-2018/)——这篇论文证明了二元语法 + 朴素贝叶斯在短文本上很难被击败。
- [scikit-learn 文本特征提取文档](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)——`CountVectorizer`、`TfidfVectorizer` 以及所有待调参数的参考资料。
