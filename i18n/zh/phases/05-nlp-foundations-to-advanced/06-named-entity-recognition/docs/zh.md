# 命名实体识别

> 把名称提取出来。听起来很简单，直到你遇到边界歧义、嵌套实体和领域术语。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 03（词嵌入）
**Time:** 约 75 分钟

## 问题

“Apple 因其在美国的 iPhone 搜索协议起诉了 Google。”包含五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、搜索协议（是否算实体仍有争议）和美国（GPE）。优秀的 NER 系统能提取所有实体并正确标注类型；糟糕的系统会漏掉 iPhone，把作为水果的 Apple 与 Apple 公司混为一谈，还会把“美国”标成 PERSON。

NER 是每条结构化抽取流水线背后的主力：解析简历、扫描合规日志、匿名化医疗记录、理解搜索查询、为聊天机器人回答提供依据、提取法律合同信息。你几乎看不见它，却始终依赖它。

本课将沿着经典路线（基于规则、HMM、CRF）走向现代路线（BiLSTM-CRF，再到 Transformer）。每一步都会解决前一种方法的一个具体限制，这种演进模式本身就是本课的重点。

## 概念

**BIO 标注**（或 BILOU）把实体抽取转换为序列标注问题。为每个词元分配 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）标签。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词元实体会串联起来：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以提取任意跨度。

架构演进如下：

- **基于规则。** 正则表达式 + 地名词典查找。对已知实体精确率高，对新实体覆盖率为零。
- **HMM。** 隐马尔可夫模型。计算给定标签时词元的发射概率，以及标签之间的转移概率，再使用维特比算法解码，通过带标签数据训练。
- **CRF。** 条件随机场。与 HMM 相似，但采用判别式学习，因此可以混合任意特征（词形、大小写、相邻词）。到 2026 年，它仍是低资源部署中的经典生产主力。
- **BiLSTM-CRF。** 用神经特征取代人工特征。LSTM 从两个方向读取句子，顶部的 CRF 层负责保证标签序列一致。
- **基于 Transformer。** 使用词元分类头微调 BERT。准确率最高，计算量也最大。

```figure
ner-bio-tagging
```

## 动手构建

### 第 1 步：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第 2 步：人工设计特征

对于经典（非神经）NER，特征决定成败。实用特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`，`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词有很强的指示作用。

### 第 3 步：简单的规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级词典会包含从 Wikipedia 和 DBpedia 抓取的数百万条记录。覆盖率不错，消歧能力（公司 `Apple` 与水果 Apple）却很差。这正是统计模型胜出的原因。

### 第 4 步：CRF（结构示意，而非完整实现）

如果没有概率论基础，用 50 行代码从零实现完整 CRF 并不能帮助理解。这里改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 分别是 L1 与 L2 正则化。`all_possible_transitions=True` 允许模型学习非法序列（例如 `I-ORG` 跟在 `O` 后面）不太可能发生；CRF 正是借此保证 BIO 一致，而无须你手写约束。

### 第 5 步：BiLSTM-CRF 增加了什么

特征变成了学习所得。输入是词元嵌入（GloVe 或 fastText）。LSTM 从左到右、从右到左读取句子，拼接后的隐藏状态进入 CRF 输出层。CRF 仍负责保证标签序列一致；LSTM 则用学习到的特征替代人工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层可以使用 `torchcrf.CRF`（pip install pytorch-crf）。与人工特征 CRF 相比，它确实有所提升，但除非拥有数万个带标签句子，否则增益会比你预想的小。

## 学以致用

spaCy 开箱即用地提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意，`iPhone` 被标成了 `ORG`，而不是 `PRODUCT`——spaCy 的小模型对产品实体覆盖较弱。大模型（`en_core_web_lg`）表现更好，Transformer 模型（`en_core_web_trf`）则更进一步。

使用 Hugging Face 执行基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X 词元合并成一个跨度。不使用它时，你只能得到词元级标签，必须自行合并。

### 基于大语言模型的 NER（2026 年的选择）

在许多领域，零样本和少样本大语言模型 NER 已经可以与微调模型竞争；当带标签数据稀缺时，它的优势尤其显著。

- **零样本提示。** 给大语言模型一组实体类型和一个示例模式，要求输出 JSON。开箱即可使用；在新领域上的准确率中等。
- **ZeroTuneBio 风格提示。** 把任务分解为候选提取 → 含义解释 → 判断 → 复查。多阶段提示（而非一次性提示）可以显著提升生物医学 NER 的准确率，同一模式也适用于法律、金融和科研领域。
- **结合 RAG 的动态提示。** 每次推理都从小型已标注种子集中检索最相似的标注示例，动态构建少样本提示。在 2026 年的基准中，这让 GPT-4 生物医学 NER 的 F1 比静态提示高出 11%～12%。
- **按实体类型拆分。** 对于长文档，一次调用同时抽取所有实体类型时，召回率会随长度增长而下降。应为每种实体类型分别执行一次抽取。推理成本更高，准确率也显著更高；这是临床记录和法律合同的标准做法。

截至 2026 年的生产建议是：在收集训练数据前，先建立大语言模型零样本基线。其 F1 往往已经足够好，让你根本无须微调。

### 经典 NER 仍能胜出的场景

即使可以使用大语言模型，经典 NER 在以下情况中仍然更合适：

- 延迟预算低于 50 毫秒。
- 已拥有数千个带标签样本，并且需要 98% 以上的 F1。
- 领域本体稳定，预训练 CRF 或 BiLSTM 可以良好迁移。
- 监管约束要求使用本地部署、非生成式模型。

### 它会在哪里失效

- **领域漂移。** 在 CoNLL 上训练的 NER 用于法律合同时，表现可能还不如词典。应在自己的领域数据上微调。
- **嵌套实体。** “Bank of America Tower”同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠跨度，需要嵌套 NER（多轮或基于跨度的模型）。
- **长实体。** “United States Federal Deposit Insurance Corporation.”词元级模型有时会把它拆开。应使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医疗 NER 中的 DRUG_BRAND、ADVERSE_EVENT、DOSE 等标签，通用模型完全不了解。这类任务应从 Scispacy 和 BioBERT 起步。

## 交付成果

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证往返转换的一致性。
2. **中等。** 在 CoNLL-2003 英语 NER 数据集上训练上面的 sklearn-crfsuite CRF。使用 `seqeval` 报告每种实体的 F1。典型结果约为 84 F1。
3. **困难。** 在特定领域的 NER 数据集（医学、法律或金融）上微调 `distilbert-base-cased`，并与 spaCy 小模型进行比较。记录数据泄漏检查，并写下令你意外的发现。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NER | 提取名称 | 为词元跨度标注类型（PERSON、ORG、GPE、DATE……）。 |
| BIO | 标注方案 | `B-X` 表示开头，`I-X` 表示延续，`O` 表示实体外部。 |
| BILOU | 更完善的 BIO | 增加 `L-X`（末尾）和 `U-X`（单元）以获得更清晰的边界。 |
| CRF | 结构化分类器 | 对标签之间的转移建模，而不只建模发射；可保证序列有效。 |
| 嵌套 NER | 重叠实体 | 一个跨度与其子跨度分别属于不同实体，BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度完全一致；词元级 F1 会夸大准确率。 |

## 延伸阅读

- [Lample 等（2016），用于命名实体识别的神经架构](https://arxiv.org/abs/1603.01360)——经典的 BiLSTM-CRF 论文。
- [Devlin 等（2018），BERT：深度双向 Transformer 的预训练](https://arxiv.org/abs/1810.04805)——介绍后来成为标准的词元分类模式。
- [spaCy 语言学特征——命名实体](https://spacy.io/usage/linguistic-features#named-entities)——`Doc.ents` 与 `Span` 上每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval)——正确的指标库，务必始终使用。
