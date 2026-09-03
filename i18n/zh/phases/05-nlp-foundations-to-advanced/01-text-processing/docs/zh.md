# 文本处理——分词、词干提取与词形还原

> 语言是连续的，模型是离散的。预处理是连接二者的桥梁。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 2 · 14（朴素贝叶斯）
**Time:** 约 45 分钟

## 问题

模型无法直接读懂“那些猫当时正在奔跑”，它读取的是整数。

每个自然语言处理系统都会从三个相同的问题开始：一个词从哪里开始？它的词根是什么？我们该如何在有帮助时把“run”“running”“ran”视为同一个词，而在不该合并时保留它们的差异？

分词一旦出错，模型学到的就是垃圾。如果分词器把 `don't` 当作一个词元，却把 `do n't` 当作两个词元，训练分布便会割裂。如果词干提取器把 `organization` 和 `organ` 归并为同一个词干，主题建模就会失效。如果词形还原器需要词性上下文，而你没有提供，动词就会被当作名词处理。

本课将从零构建这三项预处理步骤，再展示 NLTK 与 spaCy 如何完成同样的工作，让你看清其中的权衡。

## 概念

三种操作，各有自己的职责和失败模式。

**分词**把字符串拆分为词元。“词元”这个词有意保持宽泛，因为合适的粒度取决于任务。经典自然语言处理使用词级粒度，Transformer 使用子词粒度，没有空格的语言则可以使用字符粒度。

**词干提取**按照规则砍掉后缀。它快速、激进，却很粗糙。`running -> run`。`organization -> organ`。后一个例子正是它的失败模式。

**词形还原**利用语法知识把词还原成词典形式。它更慢、更准确，需要查找表或形态分析器。`ran -> run`（需要知道“ran”是“run”的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且可以容忍噪声时使用词干提取（搜索索引、粗略分类）；当含义重要时使用词形还原（问答、语义搜索，以及任何最终会由用户阅读的内容）。

```figure
edit-distance
```

## 动手构建

### 第 1 步：正则表达式单词分词器

最简单实用的分词器会在非字母数字字符处分割，同时把标点保留为独立词元。它并不完美，也不是最终方案，但只需一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

这里有三种按优先顺序匹配的模式：内部可以带撇号的单词（`don't`、`it's`）；纯数字；以及任何单独出现的非空白、非字母数字字符（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

留意它的失败模式。`3pm` 会被拆成 `['3', 'pm']`，因为我们的模式在连续字母和连续数字之间交替匹配。对大多数任务来说已经足够。URL、电子邮件地址和话题标签都会被拆坏。在生产环境中，应当把这些模式加在通用模式之前。

### 第 2 步：Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法包含五个规则阶段。只实现步骤 1a 就能覆盖最常见的英语后缀，并帮助你理解这种模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

按从上到下的顺序阅读这些规则。`ies -> i` 规则会让 `ponies -> poni`，而不是 `pony`。真正的 Porter 算法会在步骤 1b 中修正它。规则之间会相互竞争，排在前面的规则优先。因此，规则顺序比任何一条规则本身都更重要。

### 第 3 步：基于查找表的词形还原器

真正的词形还原需要形态学知识。便于教学的可行版本可以使用一张小型词形还原表，并提供后备规则。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键。`watched` 不在表中，而我们的后备规则只处理 `ing`。真正的词形还原还要处理 `ed`、不规则动词、形容词比较级，以及伴随语音变化的复数（`children -> child`）。正因如此，生产系统会使用 WordNet、spaCy 的形态分析组件或完整的形态分析器。

### 第 4 步：把它们串成流水线

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的环节是词性标注器。阶段 5 · 07（词性标注）会构建一个。现在先默认所有词都是 `NOUN`，并明确承认这一限制。

## 学以致用

NLTK 和 spaCy 提供了生产级版本，各自只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 可以处理缩写、Unicode 以及正则表达式容易漏掉的边缘情况。`PorterStemmer` 会运行全部五个阶段。`WordNetLemmatizer` 需要先把 NLTK 的 Penn Treebank 词性体系转换为 WordNet 的缩写集合。上面的转换衔接代码，正是大多数教程会跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整条流水线隐藏在 `nlp(text)` 背后。分词、词性标注和词形还原会全部执行。大规模处理时，它比 NLTK 更快，开箱即用的准确率也更高。代价是你无法轻松替换其中的单个组件。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、重视速度 | spaCy |
| Transformer 流水线（反正会使用模型自己的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人提醒你的两种失败模式

大多数教程讲完算法就结束了。真正的预处理流水线中有两个问题必然会让你吃亏，却几乎从来没人提及。

**可复现性漂移。** NLTK 和 spaCy 会在版本之间改变分词与词形还原行为。在 spaCy 2.x 中产生 `['do', "n't"]` 的输入，到 3.x 中可能变成 `["don't"]`。你的模型是在一种分布上训练的，推理现在却运行在另一种分布上。准确率会悄然下降，没人知道原因。应在 `requirements.txt` 中固定库版本，并编写预处理回归测试，冻结 20 个示例句子的预期分词结果。每次升级都运行这项测试。

**训练/推理不一致。** 训练时使用激进预处理（转小写、删除停用词、提取词干），部署时却直接接收原始用户输入，性能就会断崖式下降。这是生产级自然语言处理最常见的失败。如果训练时做了预处理，推理时就必须运行完全相同的函数。把预处理作为函数打包进模型，而不要把它留在由服务团队重新编写的笔记本单元格中。

## 交付成果

下面这个可复用提示词能帮助工程师选择预处理策略，无须先读完三本教科书。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize`，让 URL 保持为单一词元。测试：`tokenize("Visit https://example.com today.")` 应当生成一个 URL 词元。
2. **中等。** 实现 Porter 步骤 1b。如果单词包含元音并以 `ed` 或 `ing` 结尾，就移除这个后缀。同时处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个用 WordNet 作为查找表的词形还原器，但在 WordNet 中没有词条时回退到你的 Porter 词干提取器。在带标注的语料库上，将它与纯 WordNet 和纯 Porter 方法比较准确率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 词元 | 一个单词 | 模型使用的任意单位，可以是词、子词、字符或字节。 |
| 词干 | 单词的词根 | 按规则删除后缀所得的结果，不一定是真实存在的单词。 |
| 词元原形（Lemma） | 词典原形 | 可以在词典中查到的规范词形，需要语法上下文才能正确确定。 |
| 词性标签 | 词类 | NOUN、VERB、ADJ 等类别，是准确还原词形所必需的。 |
| 形态学 | 词形规则 | 单词如何根据时态、数和格改变形式；词形还原依赖这些规则。 |

## 延伸阅读

- [Porter, M. F.（1980），后缀剥离算法](https://tartarus.org/martin/PorterStemmer/def.txt)——原始论文只有五页，至今仍是最清晰的说明。
- [spaCy 101——语言学特征](https://spacy.io/usage/linguistic-features)——真实流水线如何连接。
- [NLTK 图书第 3 章](https://www.nltk.org/book/ch03.html)——你还没有想到过的分词边缘情况。
