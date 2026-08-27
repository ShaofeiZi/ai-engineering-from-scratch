# 共指消解

> “她给他打了电话。他没有接。那位医生当时在吃午饭。”三处指称涉及两个人，却没有出现任何姓名。共指消解负责判断谁是谁。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 06（NER）、阶段 5 · 07（词性标注与句法分析）
**Time:** 约 60 分钟

## 问题

从一篇 300 词的文章中提取所有对 Apple Inc. 的提及。文章直接写“Apple”时很简单，写成“这家公司”“他们”“来自库比蒂诺的科技巨头”或“乔布斯的公司”时却很难。如果不能把这些提及解析到同一个实体，你的 NER 流水线就会漏掉 60%～80% 的提及。

共指消解把所有指向同一现实世界实体的表达链接成一个簇。它是连接表层自然语言处理（NER、句法分析）与下游语义任务（信息抽取、问答、摘要、知识图谱）的黏合剂。

它在 2026 年仍然重要，原因包括：

- 摘要：“CEO 宣布了……”与“Tim Cook 宣布了……”——摘要应该说出这位 CEO 的姓名。
- 问答：“她给谁打了电话？”需要解析“她”指谁。
- 信息抽取：如果知识图谱把“PER1 创办了 Apple”和“Jobs 创办了 Apple”记录为两个不同事实，它就是错的。
- 多文档信息抽取：合并不同文章中对同一事件的提及，就是跨文档共指消解。

## 概念

![共指聚类：提及 → 实体](../assets/coref.svg)

**任务。** 输入一篇文档，输出一组提及（文本跨度）的聚类，每个簇都指向一个实体。

**提及类型。**

- **命名实体。** “Tim Cook”
- **名词性提及。** “the CEO”“the company”
- **代词性提及。** “he”“she”“they”“it”
- **同位语。** “Tim Cook, Apple's CEO,”

**架构。**

1. **基于规则（Hobbs，1978）。** 利用语法规则，在句法树上进行代词消解。它是很好的基线，在代词问题上出人意料地难以击败。
2. **提及对分类器。** 对每一对提及（m_i, m_j），预测它们是否共指，再通过传递闭包聚类。这是 2016 年前的标准方法。
3. **提及排序。** 对每个提及，为候选先行词（包括“无先行词”）排序，并选择排名最高者。
4. **基于跨度的端到端方法（Lee 等，2017）。** 使用 Transformer 编码器，枚举不超过长度上限的所有候选跨度，预测提及分数，再为每个跨度预测先行词概率，最后贪心聚类。这是现代默认方案。
5. **生成式方法（2024+）。** 提示大语言模型：“List every pronoun in this text and its antecedent.”它在简单情况中表现良好，但不擅长长文档和罕见指称对象。

**评估指标。** 标准指标共有五种（MUC、B³、CEAF、BLANC、LEA），因为没有任何单一指标可以完整衡量聚类质量。将前三种指标的平均值报告为 CoNLL F1。2026 年在 CoNLL-2012 上的顶尖水平约为 83 F1。

**已知难例。**

- 指向数页前所引入实体的限定描述。
- 桥接照应（“the wheels”→此前提到的一辆车）。
- 中文、日语等语言中的零照应。
- 后指（代词位于指称对象之前）：“When **she** walked in, Mary smiled.”

```figure
coref-links
```

## 动手构建

### 第 1 步：预训练神经共指消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

对于较长文档，你会得到类似下面的结果：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 第 2 步：基于规则的代词消解器（教学用）

`code/main.py` 中有一个仅使用标准库的实现：

1. 提取提及：命名实体（首字母大写的文本跨度）、代词（字典查找）、限定描述（“the X”）。
2. 对于每个代词，查看前 K 个提及，并根据以下因素评分：
   - 性别/数一致（启发式）
   - 新近程度（越近越优先）
   - 句法角色（优先选择主语）
3. 链接得分最高的先行词。

它无法与神经模型竞争，却展示了端到端模型必须面对的搜索空间与决策。

### 第 3 步：使用大语言模型进行共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要注意两种失败模式。第一，大语言模型会过度合并（把分别指向两个人的“him”和“her”并入一组）。第二，大语言模型会在长文档中悄然漏掉提及。始终使用跨度偏移检查进行验证。

### 第 4 步：评估

标准 conll-2012 脚本会计算 MUC、B³、CEAF-φ4，并报告三者的平均值。对于内部评估，可以先在标注测试集上计算跨度级精确率与召回率，再加入提及链接 F1。

## 陷阱

- **单例膨胀。** 某些系统会把每个提及都作为独立簇报告。B³ 对此较宽容，MUC 会予以惩罚，因此必须同时检查三项指标。
- **长上下文中的代词。** 文档超过 2000 个词元后，性能会下降约 15 个 F1 点。需要谨慎分块。
- **性别假设。** 硬编码的性别规则会在非二元指称、组织和动物上失效。应使用学习式模型或中性评分。
- **大语言模型在长文档中的漂移。** 单次 API 调用无法可靠地对 50 多个段落中的提及进行聚类。应使用滑动窗口再合并。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、单篇文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经共指消解 |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专用端到端模型（2025～2026 年顶尖方案） |
| 快速建立大语言模型基线 | GPT-4o / Claude + 结构化输出共指提示 |
| 生产对话系统 | 规则式后备 + 神经主模型 + 关键槽位人工复核 |

2026 年投入生产的集成模式是：先运行 NER，再运行共指消解，最后把共指簇合并进 NER 实体。下游任务看到的是每个簇对应的一个实体，而不是每次提及各自成为一个实体。

## 交付成果

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 个手工编写的段落上运行 `code/main.py` 中的规则式消解器，对照真实答案测量提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经共指消解模型，将得到的簇与你的人工标注比较。它在哪里失败了？
3. **困难。** 构建共指增强的 NER 流水线：先运行 NER，再通过共指簇合并结果。在 100 篇文章上测量相对于仅 NER 方案的实体覆盖率提升。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 提及 | 一次指称 | 指向某个实体的一段文本（名称、代词、名词短语）。 |
| 先行词 | “it”指向什么 | 后续提及与之共指的较早提及。 |
| 簇 | 一个实体的全部提及 | 指向同一现实世界实体的一组提及。 |
| 回指 | 向后指称 | 后续提及指向前文（“he”→“John”）。 |
| 后指 | 向前指称 | 前面的提及指向后文（“When he arrived, John...”）。 |
| 桥接 | 隐式指称 | “I bought a car. The wheels were bad.”（这辆车的车轮。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 三项 F1 的平均值。 |

## 延伸阅读

- [Jurafsky 与 Martin，《语音与语言处理》第 3 版第 26 章——共指消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf)——权威教材章节。
- [Lee 等（2017），端到端神经共指消解](https://arxiv.org/abs/1707.07045)——基于跨度的端到端方法。
- [Joshi 等（2020），SpanBERT](https://arxiv.org/abs/1907.10529)——改善共指消解的预训练方法。
- [Pradhan 等（2012），CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/)——基准。
- [Hobbs（1978），代词指称消解](https://www.sciencedirect.com/science/article/pii/0024384178900064)——经典规则方法。
