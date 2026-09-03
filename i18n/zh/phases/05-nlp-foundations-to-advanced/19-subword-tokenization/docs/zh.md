# 子词分词——BPE、WordPiece、Unigram、SentencePiece

> 单词分词器会被未见词难住，字符分词器又会让序列长度暴涨。子词分词器取二者之间的平衡，每个现代大语言模型都依赖其中一种。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 01（文本处理）、阶段 5 · 04（GloVe / FastText / 子词）
**Time:** 约 60 分钟

## 问题

你的词表包含 5 万个词，用户输入“untokenizable”，分词器却返回 `[UNK]`。模型因此无法获得关于这个词的任何信号。更糟的是：语料库中处于第 90 百分位的文档包含 40 个罕见词，这意味着每篇文档都会丢失 40 比特的信息。

子词分词解决了这个问题。常见词保持为单一词元，罕见词则拆成有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据可以覆盖一切，因为任何字符串最终都能表示成字节序列。

2026 年的每个前沿大语言模型都采用三种算法之一（BPE、Unigram、WordPiece），并由三种库之一封装（tiktoken、SentencePiece、HF Tokenizers）。要交付语言模型，就必须从中作出选择。

## 概念

![逐字符对比 BPE、Unigram 与 WordPiece](../../../../../../phases/05-nlp-foundations-to-advanced/19-subword-tokenization/assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词表开始，统计每一对相邻字符，把出现频率最高的一对合并成新词元，不断重复，直到达到目标词表大小。它是主流算法，GPT-2/3/4、Llama、Gemma、Qwen2、Mistral 都使用它。

**字节级 BPE。** 算法相同，但操作对象是原始字节（256 个基础词元），而不是 Unicode 字符。它保证不会产生 `[UNK]` 词元——任何字节序列都可以编码。GPT-2 使用 50257 个词元（256 个字节 + 50000 次合并 + 1 个特殊词元）。

**Unigram。** 从一个巨大词表开始，为每个词元分配一元概率，再反复剪掉那些移除后对语料库对数似然影响最小的词元。推理时具有概率性：可以采样不同的分词结果（通过子词正则化进行数据增强时很有用）。T5、mBART、ALBERT、XLNet、Gemma 都使用它。

**WordPiece。** 合并能够最大化训练语料库似然的词对，而不是单纯选择原始频率最高的词对。BERT、DistilBERT、ELECTRA 使用这种方法。

**SentencePiece 与 tiktoken。** SentencePiece 是直接在原始 Unicode 文本上*训练*词表（BPE 或 Unigram）的库，用 `▁` 表示空格。tiktoken 是针对预建词表的 OpenAI 高速*编码器*，不提供训练功能。

经验法则：

- **训练新词表：** 使用 SentencePiece（多语言、无须预分词）或 HF Tokenizers。
- **针对 GPT 词表快速推理：** 使用 tiktoken（cl100k_base、o200k_base）。
- **两者都需要：** 使用 HF Tokenizers，一个库同时负责训练与服务。

```figure
bpe-merge
```

## 动手构建

### 第 1 步：从零实现 BPE

见 `code/main.py`。循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

这个算法编码了三个事实。`</w>` 标记词尾，使作为后缀的“low”和作为“lower”前缀的“low”保持区别。频率加权让高频词对更早被合并。合并列表有顺序——推理时必须按训练顺序应用合并。

### 第 2 步：使用学到的合并规则编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

这个朴素实现的复杂度为 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）通过带优先队列的合并排名查找，以近线性时间运行。

### 第 3 步：实际使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无须预分词；空格被编码为 `▁`；`character_coverage` 控制保留罕见字符的积极程度，以及将多少字符映射到 `<unk>`。

### 第 4 步：为 OpenAI 兼容词表使用 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

它只提供编码，速度很快（Rust 后端）。对于字节计数、成本估算和上下文窗口预算，其分词结果与 GPT-4/5 完全一致。

## 2026 年仍会进入生产的陷阱

- **分词器漂移。** 使用词表 A 训练，却用词表 B 部署。词元 ID 不同，模型输出变成乱码。应在 CI 中检查 `tokenizer.json` 的哈希值。
- **空白歧义。** BPE 会为“hello”和“ hello”生成不同词元。始终显式指定 `add_special_tokens` 与 `add_prefix_space`。
- **多语言训练不足。** 英语占主导的语料库会让非拉丁文字被拆成 5～10 倍的词元。相同提示在 GPT-3.5 中以日语或阿拉伯语表达，成本可能高 5～10 倍。o200k_base 部分修复了这一点。
- **表情符号拆分。** 一个表情符号可能占用 5 个词元。规划上下文预算时应检查表情符号的处理方式。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 从零训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API | tiktoken（GPT-4+ 使用 `o200k_base`） |
| 领域专用词表（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，再与基础词表合并 |
| 边缘端推理、小模型 | Unigram（较小词表表现更好） |

词表大小是一项缩放决策，而非常量。粗略经验是：小于 1B 参数使用 32k，1～10B 参数使用 50～100k，多语言/前沿模型使用 200k 以上。

## 交付成果

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的微型语料库上训练执行 500 次合并的 BPE，编码三个留出词语。分别有多少词恰好生成 1 个词元，多少生成了多个词元？
2. **中等。** 对 100 个英语 Wikipedia 句子，比较 `cl100k_base`、`o200k_base` 和一个使用 vocab=32k 训练的 SentencePiece BPE 的词元数量，报告各自的压缩率。
3. **困难。** 在同一语料库上分别使用 BPE、Unigram 与 WordPiece 训练。测量各分词方案用于小型情感分类器时的下游准确率。选择会让 F1 变化超过 1 个百分点吗？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | 字节对编码 | 贪心合并最常见的字符对，直到达到目标词表大小。 |
| 字节级 BPE | 永远没有未知词元 | 在原始 256 个字节上执行 BPE；GPT-2 / Llama 使用这种方法。 |
| Unigram | 概率分词器 | 使用对数似然从大型候选集中剪枝；T5、Gemma 使用这种方法。 |
| SentencePiece | 处理空白的那个 | 直接在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 速度快的那个 | OpenAI 使用 Rust 实现的预建词表 BPE 编码器，不提供训练。 |
| 合并列表 | 那些神奇数字 | 有序的 `(a, b) → ab` 合并列表；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见才算太罕见？ | 分词器必须覆盖的训练语料字符比例；典型值约为 0.9995。 |

## 延伸阅读

- [Sennrich、Haddow、Birch（2015），使用子词单元进行稀有词神经机器翻译](https://arxiv.org/abs/1508.07909)——BPE 论文。
- [Kudo（2018），使用 Unigram 语言模型进行子词正则化](https://arxiv.org/abs/1804.10959)——Unigram 论文。
- [Kudo、Richardson（2018），SentencePiece：简单且与语言无关的子词分词器](https://arxiv.org/abs/1808.06226)——介绍该库的论文。
- [Hugging Face——分词器概览](https://huggingface.co/docs/transformers/tokenizer_summary)——简明参考。
- [OpenAI tiktoken 代码库](https://github.com/openai/tiktoken)——使用手册与编码列表。
