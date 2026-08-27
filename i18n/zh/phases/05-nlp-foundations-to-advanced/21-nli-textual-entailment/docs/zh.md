# 自然语言推断——文本蕴含

> “t 蕴含 h”意味着人类读完 t 后会认为 h 为真。NLI 的任务是预测蕴含、矛盾或中立。表面枯燥，却支撑着生产系统。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 05（情感分析）、阶段 5 · 13（问答系统）
**Time:** 约 60 分钟

## 问题

你构建了一个摘要器，它生成了一段摘要。如何知道摘要中没有幻觉？

你构建了一个聊天机器人，它回答“是”。如何知道检索到的段落确实支持这个答案？

你需要按主题分类 1 万篇新闻，却没有训练标签。能否复用现有模型？

这三个问题都可以归结为自然语言推断。NLI 会问：给定前提 `t` 和假设 `h`，`h` 是由 `t` 蕴含、与它矛盾，还是中立（无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要中的陈述。不构成蕴含 = 幻觉。
- **有依据的问答：** `t` = 检索段落，`h` = 生成答案。不构成蕴含 = 编造。
- **零样本分类：** `t` = 文档，`h` = 用自然语言表达的标签（“这篇内容与体育有关”）。构成蕴含 = 预测该标签。

一个任务，三种生产用途。这就是每个 RAG 评估框架都会在底层配备 NLI 模型的原因。

## 概念

![NLI：前提与假设之间的三分类](../assets/nli.svg)

**三个标签。**

- **蕴含。** `t` → `h`。“猫在垫子上”蕴含“这里有一只猫”。
- **矛盾。** `t` → ¬`h`。“猫在垫子上”与“这里没有猫”矛盾。
- **中立。** 两个方向都无法推断。“猫在垫子上”对于“这只猫饿了”是中立的。

**不是逻辑蕴含。** NLI 是*自然*语言推断——它关注普通人类读者会作出的推断，而不是严格逻辑。“约翰在遛狗”在 NLI 中蕴含“约翰有一条狗”，但严格的一阶逻辑只有在公理化所有权关系后才会接受它。

**数据集。**

- **SNLI**（2015）。57 万个人工标注样本对，以图像说明文字作为前提，领域较窄。
- **MultiNLI**（2017）。来自 10 种文体的 43.3 万个样本对，是 2026 年的标准训练语料库。
- **ANLI**（2019）。对抗式 NLI。人类专门编写能击败现有模型的样本，难度更高。
- **DocNLI、ConTRoL**（2020～2021）。文档长度的前提，用于测试多跳与长距离推断。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示进入三分类 softmax。在 MNLI 上训练、在留出基准上评估，可以在同分布样本对上获得 90% 以上的准确率。

**通过 NLI 实现零样本分类。** 给定一篇文档和一组候选标签，把每个标签改写成一个假设（“这段文本与体育有关”），计算每个假设的蕴含概率，并选择最大者。这就是 Hugging Face `zero-shot-classification` 流水线背后的机制。

```figure
nli-router
```

## 动手构建

### 第 1 步：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产级 NLI 的开放默认模型是 `facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli`，其中 DeBERTa-v3 位居排行榜前列。

### 第 2 步：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板为“This example is about {label}.”，可以通过 `hypothesis_template` 自定义。不需要训练数据，也不需要微调，开箱即可使用。

### 第 3 步：RAG 忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度评估的核心。先把生成答案拆成原子陈述，再逐条对照检索上下文检查是否构成蕴含，最后报告构成蕴含的比例。

### 第 4 步：手写 NLI 分类器（概念演示）

`code/main.py` 中提供了一个仅使用标准库的玩具实现：通过词法重叠和否定检测比较前提与假设。它无法与 Transformer 模型竞争，却展示了任务的基本形态：输入两段文本，输出三分类标签，损失是在 `{entail, contradict, neutral}` 上计算的交叉熵。

## 陷阱

- **仅靠假设的捷径。** 在 SNLI 上，仅根据假设就能以约 60% 的准确率预测标签，因为“not”“nobody”“never”与矛盾标签相关。这是检测标签泄漏的强力基线。
- **词法重叠启发式。** 子序列启发式（“每个子序列都被蕴含”）能通过 SNLI，却会在 HANS/ANLI 上失败。应使用对抗基准。
- **文档长度导致退化。** 单句 NLI 模型在文档级前提上会损失 20 个以上的 F1 点。长上下文应使用在 DocNLI 上训练的模型。
- **零样本模板敏感性。** “This example is about {label}”“{label}”“The topic is {label}”之间的切换，可能让准确率波动 10 个百分点以上。应调优模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科研文本需要领域专用 NLI 模型（例如 SciNLI、MedNLI）。

## 学以致用

2026 年的技术栈：

| 用例 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘端 | `cross-encoder/nli-deberta-v3-base` |
| 零样本文本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 幻觉检测 | RAGAS / DeepEval 中的 NLI 层 |

2026 年的元模式是：NLI 是文本理解领域的万能胶。只要需要回答“A 是否支持 B？”或“A 是否与 B 矛盾？”，都应先考虑 NLI，而不是再调用一次大语言模型。

## 交付成果

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 个手工编写、覆盖全部三个类别的（前提、假设、标签）三元组上运行 `facebook/bart-large-mnli`。测量准确率，再加入针对“子序列启发式”的对抗陷阱（“I did not eat the cake”与“I ate the cake”），观察模型是否会出错。
2. **中等。** 在 100 条 AG News 标题上，比较零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 与 `"{label}"`，报告准确率波动。
3. **困难。** 构建 RAG 忠实度检查器：拆分原子陈述，再对每条陈述执行 NLI。在 50 个带标准上下文的 RAG 生成答案上评估，与人工标签比较假阳性率和假阴性率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推断 | 对前提—假设关系进行三分类。 |
| RTE | 识别文本蕴含 | NLI 的旧称，任务相同。 |
| 蕴含 | “t 推出 h” | 给定 t 后，普通读者会认为 h 为真。 |
| 矛盾 | “t 排除 h” | 给定 t 后，普通读者会认为 h 为假。 |
| 中立 | “无法确定” | 无法从 t 推断 h 为真或为假。 |
| 零样本分类 | 把 NLI 当作分类器 | 将标签表述为假设，选择蕴含概率最高者。 |
| 忠实度 | 答案是否有依据？ | 对（检索上下文，生成答案）执行 NLI。 |

## 延伸阅读

- [Bowman 等（2015），用于学习自然语言推断的大型标注语料库](https://arxiv.org/abs/1508.05326)——SNLI。
- [Williams、Nangia、Bowman（2017），通过推断实现句子理解的广覆盖挑战语料库](https://arxiv.org/abs/1704.05426)——MultiNLI。
- [Nie 等（2019），对抗式 NLI](https://arxiv.org/abs/1910.14599)——ANLI 基准。
- [Yin、Hay、Roth（2019），零样本文本分类基准测试](https://arxiv.org/abs/1909.00161)——把 NLI 用作分类器。
- [He 等（2021），DeBERTa：采用解耦注意力的增强解码 BERT](https://arxiv.org/abs/2006.03654)——2026 年的 NLI 主力。
