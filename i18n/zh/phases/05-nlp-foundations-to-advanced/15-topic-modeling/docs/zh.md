# 主题建模——LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词语的概率分布。BERTopic：文档在嵌入空间中聚类，簇就是主题。目标相同，分解方式不同。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 03（Word2Vec）
**Time:** 约 45 分钟

## 问题

你手中有 1 万张客户支持工单、5 万篇新闻文章或 20 万条推文。你需要在不逐一阅读的情况下了解这批内容在谈论什么。你没有预先标注的类别，甚至不知道究竟存在多少个类别。

主题建模可以无监督地回答这个问题。输入语料库，得到一小组语义连贯的主题，以及每篇文档在这些主题上的概率分布。

目前有两个占主导地位的算法家族。LDA（2003）把每篇文档视为多个潜在主题的混合，把每个主题视为词语的概率分布，并使用贝叶斯推断。在需要混合成员主题分配和可解释的词级概率分布时，它仍被用于生产环境。

BERTopic（2020）用 BERT 编码文档，使用 UMAP 降维，通过 HDBSCAN 聚类，再用基于类别的 TF-IDF 提取主题词。对于短文本、社交媒体，以及语义相似性比词语重叠更重要的内容，它表现更佳。每篇文档只得到一个主题，这是它处理长篇内容时的局限。

本课将帮助你建立对两种方法的直觉，并说明面对具体语料库时应选择哪一种。

## 概念

![LDA 混合模型与 BERTopic 聚类](../../../../../../phases/05-nlp-foundations-to-advanced/15-topic-modeling/assets/topic-modeling.svg)

**LDA 的生成过程。** 每个主题都是词语的概率分布，每篇文档都是主题的混合。要在文档中生成一个词，先从该文档的主题混合中采样一个主题，再从这个主题的词语分布中采样一个词。推断过程则反过来：根据观察到的词，推断每篇文档的主题分布以及每个主题的词语分布。其数学计算可由折叠吉布斯采样或变分贝叶斯完成。

LDA 的关键输出：

- `doc_topic`：形状为 `(n_docs, n_topics)` 的矩阵，每行之和为 1（文档的主题混合）。
- `topic_word`：形状为 `(n_topics, vocab_size)` 的矩阵，每行之和为 1（主题的词语分布）。

**BERTopic 流水线。**

1. 使用句子 Transformer（例如 `all-MiniLM-L6-v2`）编码每篇文档，得到 384 维向量。
2. 使用 UMAP 把维度降至约 5 维。BERT 嵌入的维度太高，不适合直接聚类。
3. 使用 HDBSCAN 聚类。它基于密度，可以生成大小不一的簇，并提供一个“离群点”标签。
4. 对每个簇中的文档计算基于类别的 TF-IDF，提取排名最高的词。

每篇文档的输出是一个主题（外加 -1 离群点标签）。也可以选择通过 HDBSCAN 的概率向量获得软成员关系。

```figure
topic-drift
```

## 动手构建

### 第 1 步：通过 scikit-learn 使用 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

请注意：这里移除了停用词，使用 min_df 与 max_df 过滤罕见词和无处不在的词，并选择 CountVectorizer（而不是 TfidfVectorizer），因为 LDA 需要原始计数。

### 第 2 步：BERTopic（生产用法）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 过滤掉 BERTopic 的离群点桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 的库默认值为 10。本例针对课程规模明确将其设为 15。语料库超过 1 万篇文档时，应提高到 50 或 100。

### 第 3 步：评估

两种方法都会输出主题词。问题在于这些词是否连贯。

- **主题一致性（c_v）。** 在滑动窗口上下文中计算高排名主题词对的 NPMI（归一化逐点互信息），把分数聚合为主题向量，再通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel`，并设置 `coherence="c_v"`。
- **主题多样性。** 所有主题高排名词中唯一词语的比例。越高越好，意味着主题之间不重叠。
- **定性检查。** 阅读每个主题排名最高的词。它们是否指向一个真实事物？人工判断仍是最后一道防线。

## 如何选择

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 没有 GPU / 算力有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 使用大语言模型标注主题 | BERTopic（直接支持） |
| 资源受限的边缘端部署 | LDA |
| 追求最高语义一致性 | BERTopic |

最大的实际考量是文档长度。BERT 嵌入会截断输入；LDA 的计数可以处理任意长度。文档超过嵌入模型的上下文窗口时，要么分块后聚合，要么使用 LDA。

## 学以致用

2026 年的技术栈：

- **BERTopic。** 短文本和任何重视语义的任务的默认选择。
- **`gensim.models.LdaModel`。** 用于生产的经典 LDA，成熟且经受过实战检验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 适合实验的便捷 LDA。
- **NMF。** 非负矩阵分解。LDA 的快速替代方案，在短文本上质量相近。
- **Top2Vec。** 与 BERTopic 设计相似，社区规模较小，但在某些基准上表现良好。
- **FASTopic。** 更新的方案，在超大语料库上比 BERTopic 更快。
- **基于大语言模型的命名。** 使用任意方法完成聚类，再提示模型为每个簇命名。

## 交付成果

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上拟合包含 5 个主题的 LDA。打印每个主题排名最高的 10 个词，再手工为各主题命名。算法找到了真实类别吗？
2. **中等。** 在同一份 20 Newsgroups 子集上拟合 BERTopic。将找到的主题数量、排名最高的词和定性一致性与 LDA 比较。哪种方法能更清楚地呈现真实类别？
3. **困难。** 在你的语料库上分别计算 LDA 与 BERTopic 的 c_v 一致性。令每种方法依次使用 5、10、20、50 个主题，绘制一致性随主题数量变化的曲线，并报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 主题 | 语料库谈论的事物 | 词语的概率分布（LDA），或相似文档组成的簇（BERTopic）。 |
| 混合成员关系 | 文档属于多个主题 | LDA 为每篇文档分配覆盖全部主题的概率分布。 |
| UMAP | 降维 | 保留局部结构的流形学习方法，用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 找出大小不一的簇，并为离群点生成“噪声”标签（-1）。 |
| c_v 一致性 | 主题质量指标 | 计算滑动窗口中高排名主题词的逐点互信息平均值。 |

## 延伸阅读

- [Blei、Ng、Jordan（2003），潜在狄利克雷分配](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)——LDA 论文。
- [Grootendorst（2022），BERTopic：采用基于类别的 TF-IDF 过程进行神经主题建模](https://arxiv.org/abs/2203.05794)——BERTopic 论文。
- [Röder、Both、Hinneburg（2015），探索主题一致性指标空间](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)——提出 c_v 等指标的论文。
- [BERTopic 文档](https://maartengr.github.io/BERTopic/)——生产参考，示例非常出色。
