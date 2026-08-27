# 多语言自然语言处理

> 一个模型覆盖 100 多种语言，其中大多数语言都没有训练数据。跨语言迁移是 2020 年代真正实用的奇迹。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 04（GloVe、FastText、子词）、阶段 5 · 11（机器翻译）
**Time:** 约 45 分钟

## 问题

英语拥有数十亿个带标签样本，乌尔都语只有数千个，迈蒂利语则几乎没有。任何面向全球用户的实用自然语言处理系统，都必须覆盖那些缺乏任务专用训练数据的长尾语言。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示使模型能够把从高资源语言中学到的技能迁移到低资源语言。只用英语情感分析数据微调模型，它就能开箱即用地对乌尔都语作出出人意料地准确的情感预测。这就是零样本跨语言迁移，它重塑了自然语言处理服务全球用户的方式。

本课将说明其中的权衡、经典模型，以及最容易让刚接触多语言任务的团队踩坑的一项决策：选择哪种源语言进行迁移。

## 概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表由各语言共享：相关语言中的同一词素会使用同一个子词单元。例如英语与意大利语中的 `anti-` 会得到相同词元。

**共享表示。** 在多种语言上通过掩码语言建模预训练的 Transformer，会让不同语言中语义相似的句子产生相似隐藏状态。mBERT、XLM-R 和 NLLB 都呈现这种性质。英语“cat”的嵌入会聚集在法语“chat”和西班牙语“gato”附近，完整句子嵌入也同样如此。

**零样本迁移。** 使用一种语言（通常是英语）的带标签数据微调模型，推理时再直接用于模型支持的任何其他语言，不需要目标语言标签。对于类型学相近的语言，结果很强；对于距离较远的语言，结果较弱。

**少样本微调。** 在目标语言中增加 100～500 个带标签样本。在分类任务上，准确率会跃升至英语基线的 95%～98%。这是多语言自然语言处理中成本效益最高的单一手段。

## 模型

| 模型 | 年份 | 覆盖范围 | 说明 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言语言模型，在低资源语言上较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（规模远大于 Wikipedia）。确立了跨语言基线。Base 为 270M，Large 为 550M。 |
| XLM-V | 2023 | 100 种语言 | 采用 100 万词元词表（XLM-R 为 25 万）的 XLM-R，在低资源语言上更好。 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型，包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 以多语言方式训练的开放 176B 大语言模型。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言大语言模型，擅长阿拉伯语、印地语和斯瓦希里语。 |

应根据用例选择。分类任务以 XLM-R-base 作为稳妥默认方案；生成任务根据是翻译还是开放生成选择 mT5 或 NLLB；大语言模型式任务则可以选 Aya-23 或 Claude，并使用明确的多语言提示。

## 源语言决策（2026 年研究）

大多数团队默认用英语作为微调源语言。近期研究（2026）表明，这往往是错误的。

语言相似性对迁移质量的预测能力胜过原始语料规模。对于斯拉夫语族目标，德语或俄语往往胜过英语；对于印度语族目标，印地语往往胜过英语。**qWALS** 相似度指标（2026，基于《世界语言结构地图集》的特征）可以量化这种关系。**LANGRANK**（Lin 等，ACL 2019）是另一种更早出现的方法，它综合语言学相似性、语料规模和谱系关系，对候选源语言排序。

实用规则：如果目标语言存在类型学相近的高资源语言，先尝试在这种语言上微调，再与英语微调结果比较。

```figure
n5-crosslingual-bridge
```

## 动手构建

### 第 1 步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型、三种语言、同一套 API。XLM-R 在 NLI 数据上训练，通过蕴含关系技巧可以很好地迁移到分类任务。

### 第 2 步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

互为翻译的句子会在嵌入空间中彼此靠近，不同的英语句子则离得更远。跨语言检索、聚类与相似度计算正是由此实现。

### 第 3 步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100～500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。更高的学习率会破坏多语言对齐，让你得到一个只会英语的模型。

## 真正有效的评估

- **在留出集上逐语言计算准确率。** 不要聚合。聚合值会掩盖长尾问题。
- **与单语言基线比较。** 对于数据充足的语言，从零训练的单语言模型有时能胜过多语言模型。必须测试。
- **实体级测试。** 检查目标语言中的命名实体。多语言模型对与拉丁文字差异很大的文字体系通常分词较弱。
- **跨语言一致性。** 同一含义在两种语言中应产生相同预测，测量其中的差距。

## 学以致用

2026 年的技术栈：

| 任务 | 推荐方案 |
|-----|-------------|
| 覆盖 100 种语言的分类 | 微调 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 覆盖 200 种语言的翻译 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 多语言生成 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V，或在相关高资源语言上进行领域微调 |

如果表现很重要，始终要为目标语言微调预留预算。零样本只是起点，不是最终答案。

### 分词成本（低资源语言会出什么问题）

多语言模型让所有语言共享一个分词器，而这个词表在由英语、法语、西班牙语、中文和德语占主导的语料库上训练。对于主流集合之外的语言，三种成本会悄然叠加：

- **繁衍率成本。** 低资源语言文本中，每个词被切分出的词元远多于英语。一个印地语句子可能需要相同英语句子的 3～5 倍词元。这 3～5 倍会吞噬上下文窗口、训练效率和延迟预算。
- **变体恢复成本。** 每个拼写错误、变音符号变体、Unicode 归一化差异或大小写变化，都会在嵌入空间中变成互不相关的冷启动序列。模型无法学到母语使用者眼中显而易见的拼写对应关系。
- **容量溢出成本。** 前两项成本会占用上下文位置、层深度和嵌入维度。对于同一个模型，留给实际推理的容量系统性地少于高资源语言。

实际症状是：模型在印地语上看似正常地训练，损失曲线正确，评估困惑度也很合理，但生产输出仍会出现微妙错误。句子中途的形态结构崩溃，罕见屈折形式始终无法恢复。**有缺陷的分词器无法单靠增加数据规模来补救。**

缓解方法：选择能良好覆盖目标语言的分词器（XLM-V 的 100 万词元词表就是直接修复）；训练前在留出的目标语言文本上验证分词繁衍率；对于真正长尾的文字体系，使用字节级回退（SentencePiece `byte_fallback=True`、GPT-2 风格字节级 BPE），确保任何内容都不会成为 OOV。

## 交付成果

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语中各选 10 个句子，运行零样本分类流水线，并报告每种语言的准确率。你应当看到法语表现很强，印地语尚可，阿拉伯语波动较大。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料库上构建跨语言检索器。使用英语查询，检索任意语言的文档，并测量 Recall@5。
3. **困难。** 对一个印地语分类任务，比较以英语和印地语作为源语言进行微调的效果。两种方案都使用 500 个目标语言样本进行少样本微调。报告哪种源语言产生了更高的印地语准确率，以及高出多少。这就是 LANGRANK 论点的缩影。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| 跨语言迁移 | 在一种语言上训练，在另一种语言上运行 | 在源语言上微调，无须目标语言标签即可在目标语言上评估。 |
| 零样本 | 没有目标语言标签 | 不在目标语言上微调即可迁移。 |
| 少样本 | 少量目标语言标签 | 使用 100～500 个目标语言样本进行微调。 |
| mBERT | 第一个多语言语言模型 | 在 Wikipedia 上预训练、覆盖 104 种语言的 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练、覆盖 100 种语言的 RoBERTa。 |
| NLLB | Meta 的 200 语言机器翻译模型 | No Language Left Behind，包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau 等（2019），大规模无监督跨语言表示学习](https://arxiv.org/abs/1911.02116)——XLM-R 论文。
- [Pires、Schlinger、Garrette（2019），多语言 BERT 究竟有多“多语言”？](https://arxiv.org/abs/1906.01502)——开启跨语言迁移研究路线的分析论文。
- [Costa-jussà 等（2022），不让任何语言掉队](https://arxiv.org/abs/2207.04672)——NLLB-200 论文。
- [Üstün 等（2024），Aya 模型：经过指令微调的开放访问多语言语言模型](https://arxiv.org/abs/2402.07827)——Cohere 的多语言大语言模型 Aya。
- [语言相似性可以预测跨语言迁移学习表现（2026）](https://www.mdpi.com/2504-4990/8/3/65)——关于 qWALS / LANGRANK 与源语言选择的论文。
