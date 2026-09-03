# 实体链接与消歧

> NER 找到了“Paris”。实体链接需要判断：法国巴黎？Paris Hilton？美国得克萨斯州巴黎？还是特洛伊王子帕里斯？没有链接，你的知识图谱仍然充满歧义。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 06（NER）、阶段 5 · 24（共指消解）
**Time:** 约 60 分钟

## 问题

一个句子写道：“Jordan beat the press.”你的 NER 把“Jordan”标成 PERSON。不错，但它指的是*哪个* Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利机器学习教授——在机器学习论文中确实会发生这种混淆）？
- Jordan（约旦这个国家）？
- Jordan（希伯来语人名）？

实体链接（EL）会把每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域知识库。它包含两个子任务：

1. **候选生成。** 给定“Jordan”，哪些知识库条目有可能对应它？
2. **消歧。** 给定上下文，哪个候选项才是正确答案？

两个步骤都可以学习，也都有对应基准。组合后的流水线已经稳定使用了十年，持续变化的是消歧器的质量。

## 概念

![实体链接流水线：提及 → 候选项 → 消歧后的实体](../../../../../../phases/05-nlp-foundations-to-advanced/25-entity-linking/assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（“Jordan”），在别名索引中查找候选项。Wikipedia 别名字典覆盖大多数命名实体：“JFK”→ John F. Kennedy、Jacqueline Kennedy、JFK airport、JFK（电影）。典型索引会为每个提及返回 10～30 个候选项。

**三种消歧方法。**

1. **先验 + 上下文（Milne 与 Witten，2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无须训练。
2. **基于嵌入（ESS / REL / Blink）。** 编码提及与上下文，再编码每个候选实体的描述，选择余弦相似度最高者。这是 2020～2024 年的默认方案。
3. **生成式（GENRE，2021；基于大语言模型，2023+）。** 逐字符解码实体的 Wikipedia 标准标题，并通过有效实体名称构成的字典树进行约束，确保输出一定是有效的知识库 ID。

**端到端与流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）会在一次处理中完成 NER、候选生成和消歧。生产环境仍以流水线系统为主，因为其中的组件可以独立替换。

### 两项测量

- **提及召回率（候选生成）。** 正确知识库条目出现在候选列表中的真实提及比例。这是整条流水线的上限。
- **消歧准确率/F1。** 在候选项正确的前提下，排名第一的结果有多大比例正确。

始终同时报告二者。一个消歧率 99%、候选召回率 80% 的系统，整条流水线只有 80%。

```figure
gx-entity-linking
```

## 动手构建

### 第 1 步：根据 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据包含约 1800 万个（别名，实体）对。可从 Wikidata 转储中下载，并存储为倒排索引。

### 第 2 步：基于上下文消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

这里的 Jaccard 重叠只是玩具实现。实际应用中应替换为嵌入的余弦相似度（Transformer 版本见 `code/main.py` 的第 2 步）。

### 第 3 步：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

建立索引时，每个知识库实体只需嵌入一次。查询时，只需嵌入一次提及及其上下文，再与候选池做点积，选择最大者。

### 第 4 步：生成式实体链接（概念）

GENRE 会逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）保证输出只能是有效标题，并与知识库支持的字典树紧密集成。现代后继方案包括 REL-GEN，以及使用结构化输出提示的大语言模型实体链接。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

再配合允许列表（Outlines `choice`），这就是 2026 年最容易交付的实体链接流水线。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准实体链接基准：1393 篇 Reuters 文章、3.4 万个提及、Wikipedia 实体。应报告知识库内准确率（`P@1`）和知识库外 NIL 检测率。

## 陷阱

- **NIL 处理。** 有些提及不在知识库中（新出现的实体、鲜为人知的人物）。系统必须预测 NIL，而不是猜错实体，并单独测量这一能力。
- **提及边界错误。** 上游 NER 漏掉部分跨度（“Bank of America”只标出“Bank”），实体链接的召回率就会下降。
- **流行度偏差。** 训练后的系统会过度预测高频实体。机器学习论文中的“Michael I. Jordan”经常被链接到篮球运动员 Jordan。
- **跨语言实体链接。** 把中文文本中的提及映射到英语 Wikipedia 实体，需要多语言编码器或翻译步骤。
- **知识库陈旧。** 新公司、新事件、新人物不在去年的 Wikipedia 转储中。生产流水线需要刷新机制。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语 + Wikipedia | BLINK 或 REL |
| 跨语言，知识库 = Wikipedia | mGENRE |
| 适合大语言模型、每天只有少量提及 | 向 Claude/GPT-4 提供候选列表 + 约束 JSON |
| 领域专用知识库（医学、法律） | 自定义 BERT + 感知知识库的检索，并在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅使用精确匹配先验（Milne-Witten 基线） |
| 研究前沿 | GENRE / ExtEnD / 生成式大语言模型实体链接 |

2026 年可交付的生产模式是：NER → 共指消解 → 对每个提及执行实体链接 → 把簇折叠为每簇一个标准实体。输出应是文档中每个实体一个知识库 ID，而不是每次提及一个 ID。

## 交付成果

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 10 个有歧义的提及（Paris、Jordan、Apple）上实现 `code/main.py` 中的先验 + 上下文消歧器。手工标注正确实体并测量准确率。
2. **中等。** 使用句子 Transformer 编码 50 个有歧义的提及，再嵌入每个候选实体的描述。将基于嵌入的消歧与 Jaccard 上下文重叠进行比较。
3. **困难。** 构建一个包含 1000 个实体的领域知识库（例如你公司的员工与产品），端到端实现 NER + 实体链接，并在 100 个留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到 Wikipedia | 把一个提及映射到唯一的知识库条目。 |
| 候选生成 | 它可能是谁？ | 为一个提及返回可能的知识库条目短名单。 |
| 消歧 | 选出正确对象 | 使用上下文为候选项评分，并选择胜出者。 |
| 别名索引 | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在知识库中 | 明确预测没有任何知识库条目与之匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准 | 包含标准实体链接的 1393 篇 Reuters 文章。 |

## 延伸阅读

- [Milne、Witten（2008），借助 Wikipedia 学习实体链接](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf)——奠基性的先验 + 上下文方法。
- [Wu 等（2020），使用稠密实体检索进行零样本实体链接（BLINK）](https://arxiv.org/abs/1911.03814)——基于嵌入的主力模型。
- [De Cao 等（2021），自回归实体检索（GENRE）](https://arxiv.org/abs/2010.00904)——使用约束解码的生成式实体链接。
- [Hoffart 等（2011），文本中命名实体的稳健消歧（AIDA）](https://www.aclweb.org/anthology/D11-1072.pdf)——基准论文。
- [REL：站在巨人肩膀上的实体链接器（2020）](https://arxiv.org/abs/2006.01969)——开放的生产级技术栈。
