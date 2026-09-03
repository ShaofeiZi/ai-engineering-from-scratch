# 关系抽取与知识图谱构建

> NER 找到了实体，实体链接把它们锚定下来，关系抽取则找出实体之间的边。知识图谱由节点、边及其溯源信息共同构成。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 06（NER）、阶段 5 · 25（实体链接）
**Time:** 约 60 分钟

## 问题

分析师读到：“Tim Cook 在 2011 年成为 Apple 的 CEO。”其中包含四项事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）把自由文本转换成结构化三元组 `(subject, relation, object)`。在整个语料库上汇总它们，就得到了知识图谱；汇总后再查询，就得到了一套可用于 RAG、分析或合规审计的推理基础。

2026 年的问题在于：大语言模型会非常积极地抽取关系——积极过头了。它们会虚构源文并不支持的三元组。没有溯源信息，你就无法分辨真实三元组和看似可信的虚构内容。2026 年的解决方案是 AEVS 风格的锚定与验证流水线。

## 概念

![文本 → 三元组 → 知识图谱](../../../../../../phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg/assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系可以来自封闭本体（Wikidata 属性、FIBO、UMLS），也可以来自开放集合（OpenIE 风格，任何关系都允许）。

**三种抽取方法。**

1. **基于规则/模式。** Hearst 模式：“X such as Y”→`(Y, isA, X)`，再配合人工编写的正则表达式。脆弱，但精确且可解释。
2. **监督式分类器。** 给定句子中的两个实体提及，从固定集合中预测二者的关系。在 TACRED、ACE、KBP 上训练，是 2015～2022 年的标准方案。
3. **生成式大语言模型。** 提示模型输出三元组，开箱即可工作；但必须带溯源信息，否则会生成貌似合理的垃圾内容。

**AEVS（锚定—抽取—验证—补充，2026）。** 当前用于缓解幻觉的框架：

- **锚定。** 用精确位置找出每个实体跨度与关系短语跨度。
- **抽取。** 生成链接到锚定跨度的三元组。
- **验证。** 把每个三元组元素匹配回源文本，拒绝任何没有依据的内容。
- **补充。** 通过覆盖检查，确保没有遗漏任何已锚定跨度。

幻觉会显著减少。它需要更多计算，但可以审计。

**开放与封闭之间的权衡。**

- **封闭本体。** 固定的属性列表（例如 Wikidata 的 1.1 万多个属性）。可预测、可查询，也很难凭空发明。
- **开放信息抽取。** 任何动词短语都可以成为关系。召回率高、精确率低，也难以查询。

生产知识图谱通常会混合使用二者：先用开放信息抽取发现关系，再把关系规范到封闭本体上，然后才合并进主图谱。

```figure
relation-triples
```

## 动手构建

### 第 1 步：基于模式抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst 模式至今仍用于领域专用流水线，因为它们易于调试。

### 第 2 步：监督式关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是序列到序列关系抽取器：输入文本，输出已经使用 Wikidata 属性 ID 表示的三元组。它在远程监督数据上微调，是标准的开放权重基线。

### 第 3 步：使用大语言模型提示与锚定进行抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

逐一对照源文验证返回的跨度。如果 `text[start:end] != triple_entity`，就拒绝对应结果。这是最简形式的 AEVS“验证”步骤。

### 第 4 步：规范到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化往往占整个工程工作的 60%～80%，务必为它预留预算。

### 第 5 步：构建小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个知识图谱 RAG 系统的基本单元。可以使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储扩展规模。

## 陷阱

- **先做共指消解，再做关系抽取。** “He founded Apple”——关系抽取需要先知道“he”指谁。先运行共指消解（第 24 课）。
- **实体规范化。** “Apple Inc”与“Apple”必须解析到同一节点。先做实体链接（第 25 课）。
- **虚构三元组。** 大语言模型会输出源文没有支持的三元组。必须强制执行跨度验证。
- **关系规范化漂移。** 开放信息抽取的关系并不一致（“was born in”“came from”“is a native of”）。应归并到标准 ID，否则图谱将无法查询。
- **时间错误。** “Tim Cook is CEO of Apple”现在为真，在 2005 年却不成立。许多关系都有时间范围。应使用限定符（Wikidata 中 `P580` 表示开始时间，`P582` 表示结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科研文本通常需要针对领域微调的关系抽取模型。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用领域、快速投入生产 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域专用（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| 大语言模型提示、可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 高吞吐新闻信息抽取 | 基于模式 + 监督模型的混合方案 |
| 从零构建知识图谱 | 开放信息抽取 + 人工规范化步骤 |
| 时态知识图谱 | 抽取时加入限定符（开始/结束时间、时间点） |

集成模式为：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 加载图谱。每个阶段都可以成为质量门禁。

## 交付成果

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 5 个新闻文章句子上运行 `code/main.py` 中的模式抽取器，人工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小型大语言模型），比较抽取的三元组。哪个抽取器精确率更高？哪个召回率更高？
3. **困难。** 构建 AEVS 流水线：使用大语言模型抽取，再对照源文验证跨度。在 50 个 Wikipedia 风格的句子上，测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组 | 主语—关系—宾语 | `(s, r, o)` 元组，是知识图谱的原子单元。 |
| 开放信息抽取 | 什么都抽取 | 开放词表关系短语；召回率高，精确率低。 |
| 封闭本体 | 固定模式 | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化 | 统一所有内容 | 把表面名称和关系映射为标准 ID。 |
| AEVS | 有依据的抽取 | 锚定—抽取—验证—补充流水线（2026）。 |
| 溯源 | 指向事实来源 | 每个三元组都携带源文档 ID 与字符跨度。 |
| 远程监督 | 低成本标签 | 把文本与现有知识图谱对齐，生成训练数据。 |

## 延伸阅读

- [Mintz 等（2009），无需标注数据的关系抽取远程监督](https://www.aclweb.org/anthology/P09-1113.pdf)——远程监督论文。
- [Huguet Cabot、Navigli（2021），REBEL：通过端到端语言生成进行关系抽取](https://aclanthology.org/2021.findings-emnlp.204.pdf)——序列到序列关系抽取主力。
- [Wadden 等（2019），使用上下文化跨度表示抽取实体、关系与事件（DyGIE++）](https://arxiv.org/abs/1909.03546)——联合信息抽取。
- [AEVS——锚定—抽取—验证—补充框架](https://www.mdpi.com/2073-431X/15/3/178)——2026 年的幻觉缓解设计。
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial)——标准图查询。
