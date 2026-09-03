# 问答系统

> 三类系统塑造了现代问答。抽取式系统寻找文本跨度，检索增强系统用文档为答案提供依据，生成式系统直接生成答案。每个现代 AI 助手都是三者的组合。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 11（机器翻译）、阶段 5 · 10（注意力机制）
**Time:** 约 75 分钟

## 问题

用户输入“第一代 iPhone 是什么时候发布的？”，期望得到“2007 年 6 月 29 日。”不是“苹果的历史悠久而复杂”，也不是孤零零的“2007”，而是一个直接、有依据且正确的答案。

过去十年间，三种架构主导了问答系统。

- **抽取式问答。** 给定一个问题和一段已知包含答案的文本，找出答案在文本中的起止索引。SQuAD 是经典基准。
- **开放域问答。** 不提供文本段落。先检索相关段落，再抽取或生成答案。这是当今每条 RAG 流水线的基础。
- **生成式/闭卷问答。** 大语言模型从参数记忆中回答，不进行检索。推理最快，对事实最不可靠。

2026 年的趋势是混合方案：检索最相关的几个段落，再提示生成模型依据这些段落作答。这就是 RAG，第 14 课会深入讲解检索部分。本课则构建问答部分。

## 概念

![问答架构：抽取式、检索增强式、生成式](../../../../../../phases/05-nlp-foundations-to-advanced/13-question-answering/assets/qa.svg)

**抽取式。** 使用 Transformer（BERT 家族）联合编码问题和文本段落。训练两个头，分别预测答案跨度的起始与结束词元索引。损失是在有效位置上计算的交叉熵。输出是原文中的一个跨度。它从机制上不会产生幻觉，也从机制上无法回答原文中没有答案的问题。

**检索增强式（RAG）。** 分为两个阶段。首先，检索器从语料库中找出排名前 `k` 的段落；然后，阅读器（抽取式或生成式）利用这些段落生成答案。拆分检索器与阅读器后，可以分别训练和评估二者。现代 RAG 通常还会在二者之间加入重排器。

**生成式。** 仅解码器大语言模型（GPT、Claude、Llama）依靠学习到的权重回答，不执行检索。对于常识表现出色，对于罕见或近期事实则可能灾难性地失败。幻觉率与事实在预训练数据中的出现频率负相关。

```figure
qa-span
```

## 动手构建

### 第 1 步：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在包含不可回答问题的 SQuAD 2.0 上训练。默认情况下，即使模型的空答案得分最高，`question-answering` 流水线也会返回得分最高的文本跨度——它*不会*自动返回空答案。若要获得明确的“无答案”行为，请在调用流水线时传入 `handle_impossible_answer=True`；只有当空答案得分超过所有跨度得分时，流水线才会返回空答案。无论如何，都要检查 `score` 字段。

### 第 2 步：检索增强流水线（概要）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

这是一条两阶段流水线。稠密检索器（Sentence-BERT）通过语义相似度找到相关段落，抽取式阅读器（RoBERTa-SQuAD）再从合并后的高排名段落中抽取答案跨度。它适用于小型语料库。面对百万文档规模的语料库，应使用 FAISS 或向量数据库。

### 第 3 步：结合 RAG 的生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式非常重要。与朴素提示相比，明确要求模型以给定上下文为依据，并在上下文不足时回答“我不知道”，可以把幻觉率降低 40%～60%。更复杂的模式还会加入引用、置信度和结构化抽取。

### 第 4 步：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**词元级 F1**。EM 会先进行规范化（转小写、移除标点、删除冠词），然后严格匹配——预测要么完全相同而得分 1，要么得分 0。F1 根据预测与参考答案的词元重叠度计算，可以给出部分分数。二者都会低估释义：“June 29, 2007”与“June 29th, 2007”通常得到 0 EM（序数形式无法通过规范化消除），但重叠词元仍会带来可观的 F1。

对于生产问答系统，应评估：

- **答案准确率**（由大语言模型或人工判断，因为传统指标无法识别语义等价）。
- **引用准确率。** 引用段落是否真的支持答案？可以通过生成的引用与检索段落之间的字符串匹配自动检查。
- **拒答校准。** 当检索段落中没有答案时，系统能否正确回答“我不知道”？应测量错误自信率。
- **检索召回率。** 评估阅读器之前，先测量检索器是否把正确段落放入排名前 `k` 的结果中。阅读器无法弥补缺失的段落。

### RAGAS：2026 年的生产评估框架

`RAGAS` 专为 RAG 系统构建，是 2026 年的交付默认方案。它不需要标准参考答案，就能评估四个维度：

- **忠实度。** 答案中的每项陈述是否来自检索上下文？通过基于 NLI 的蕴含关系衡量。这是首要幻觉指标。
- **答案相关性。** 答案是否回应了问题？方法是根据答案生成假设问题，再与真实问题比较。
- **上下文精确率。** 检索到的文本块中有多大比例真正相关？精确率低意味着提示中噪声较多。
- **上下文召回率。** 检索结果是否包含所有必要信息？召回率低意味着阅读器不可能成功。

无参考评分让你可以直接在生产流量上评估，而无须整理标准答案。对于精确匹配指标毫无意义的开放式问题，再叠加大语言模型裁判。

`pip install ragas`。接入检索器和阅读器，每次查询就能得到四个标量，并可针对回归发出告警。

## 学以致用

2026 年的技术栈如下。

| 用例 | 推荐方案 |
|---------|-------------|
| 给定段落，寻找答案跨度 | `deepset/roberta-base-squad2` |
| 在固定语料库上问答，不能接受闭卷方式 | RAG：稠密检索器 + 大语言模型阅读器 |
| 在文档库上实时问答 | RAG，使用混合（BM25 + 稠密）检索器和重排器（第 14 课） |
| 对话式问答（包含后续追问） | 大语言模型携带对话历史，并在每轮执行 RAG |
| 对事实要求很高的监管领域 | 在权威语料库上执行抽取式问答，绝不单独使用生成式方法 |

2026 年，抽取式问答已经不再流行，因为 RAG 加大语言模型能够处理更多情况。但在必须逐字引用的场景中，它仍然会被部署，例如法律研究、监管合规和审计工具。

## 交付成果

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 个 Wikipedia 段落上配置上述 SQuAD 抽取式流水线，手工编写 10 个问题，并测量答案正确率。如果段落和问题足够清晰，应有 7～9 个回答正确。
2. **中等。** 增加拒答分类器。当最高检索分数低于某个阈值（例如余弦相似度 0.3）时，返回“I don't know”，而不调用阅读器。在留出集上调节阈值。
3. **困难。** 在任选的 1 万篇文档语料库上构建 RAG 流水线。实现结合 BM25 与稠密检索的混合检索，并通过 RRF 融合（见第 14 课）。分别测量使用与不使用混合步骤时的答案准确率，并记录哪些问题类型受益最多。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答 | 寻找答案跨度 | 预测给定段落中答案的起止索引。 |
| 开放域问答 | 在语料库上问答 | 不提供指定段落，必须先检索再回答。 |
| RAG | 先检索，再生成 | 检索增强生成，由检索器与阅读器组成的流水线。 |
| SQuAD | 经典基准 | 斯坦福问答数据集，使用 EM + F1 指标。 |
| 幻觉 | 编造的答案 | 阅读器输出不受检索上下文支持。 |
| 拒答校准 | 知道何时闭嘴 | 系统无法回答时，能正确地说“I don't know”。 |

## 延伸阅读

- [Rajpurkar 等（2016），SQuAD：用于机器文本理解的十万多个问题](https://arxiv.org/abs/1606.05250)——基准论文。
- [Karpukhin 等（2020），用于开放域问答的稠密段落检索](https://arxiv.org/abs/2004.04906)——DPR，问答领域经典的稠密检索器。
- [Lewis 等（2020），用于知识密集型自然语言处理任务的检索增强生成](https://arxiv.org/abs/2005.11401)——为 RAG 命名的论文。
- [Gao 等（2023），大语言模型的检索增强生成综述](https://arxiv.org/abs/2312.10997)——全面的 RAG 综述。
