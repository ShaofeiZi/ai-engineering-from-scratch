# BERT——掩码语言建模

> GPT 预测下一个词，BERT 预测缺失的词。仅此一处差异，便开启了此后五年的嵌入模型时代。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 05（完整 Transformer）、阶段 5 · 02（文本表示）
**Time:** 约 45 分钟

## 问题

2018 年，每项自然语言处理任务——情感分析、NER、问答、蕴含——都要在自己的带标签数据上从零训练模型。当时不存在可以微调的预训练“理解英语”检查点。ELMo（2018）证明了可以使用双向 LSTM 预训练上下文嵌入；它有所帮助，却无法广泛泛化。

BERT（Devlin 等，2018）提出：如果拿一个 Transformer 编码器，在互联网上的所有句子上训练，并迫使它根据左右两侧上下文预测缺失词语，会怎样？之后只需针对下游任务微调一个输出头。这种参数效率令人耳目一新。

结果是：仅仅 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）就主导了当时所有自然语言处理排行榜。到 2020 年，全球每个搜索引擎、内容审核流水线和语义搜索系统中几乎都有一个 BERT。

到 2026 年，仅编码器模型仍然是分类、检索与结构化抽取的正确工具——每词元运行速度比解码器快 5～10 倍，其嵌入也是每个现代检索技术栈的骨干。ModernBERT（2024 年 12 月）通过 Flash Attention + RoPE + GeGLU，把架构扩展到 8K 上下文。

## 概念

![掩码语言建模：选择词元、遮盖、预测原词](../../../../../../phases/07-transformers-deep-dive/06-bert-masked-language-modeling/assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机遮盖 15% 的词元：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测被遮盖位置上的原始词元。由于编码器是双向的，在位置 1 预测 `[MASK]` 时可以利用位置 2 之后的 `brown fox jumps`。这是 GPT 无法做到的事情。

### BERT 的掩码规则

在选中用于预测的 15% 词元中：

- 80% 替换为 `[MASK]`。
- 10% 替换为随机词元。
- 10% 保持不变。

为什么不始终使用 `[MASK]`？因为 `[MASK]` 在推理时从不出现。如果训练模型预期所有被选位置都出现 `[MASK]`，预训练与微调之间就会产生分布偏移。10% 随机替换 + 10% 保持不变，可以让模型面对更真实的输入。

### 下一句预测（NSP）——以及它为何被弃用

原始 BERT 还会训练 NSP：给定句子 A 与 B，预测 B 是否紧接着 A 出现。RoBERTa（2019）通过消融实验表明，NSP 不仅无益，反而有害。现代编码器不再使用它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文使用 2026 年的现代组件重建了整个模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置编码 | 学习式绝对编码 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | 预归一化 RMSNorm |
| 注意力 | 完全稠密 | 局部（128）与全局交替 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年技术栈不同，它原生支持 Flash Attention。在序列长度 8K 时，推理速度比 DeBERTa-v3 快 2～3 倍，GLUE 分数也更高。

### 2026 年仍应选择编码器的用例

| 任务 | 编码器胜过解码器的原因 |
|------|---------------------------|
| 检索/语义搜索嵌入 | 双向上下文 = 每词元的嵌入质量更高 |
| 分类（情感、意图、有害内容） | 一次前向传播，无生成开销 |
| NER/词元标注 | 逐位置输出，原生双向 |
| 零样本蕴含（NLI） | 编码器顶部的分类头 |
| RAG 重排器 | 交叉编码器评分，比大语言模型重排器快 10 倍 |

```figure
transformer-residual
```

## 动手构建

### 第 1 步：掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收词元 ID 列表、词表大小和掩码概率。它返回应用掩码后的输入 ID，以及标签（只在被遮盖位置保留标签，其他位置为 -100——这是 PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第 2 步：在微型语料库上运行 MLM 预测

在只有 20 个词的词表和 200 个句子上，训练一个双层编码器 + MLM 头。不计算梯度——这里只做前向传播健全性检查。完整训练需要 PyTorch。

### 第 3 步：比较掩码类型

展示三路规则如何让模型在不存在 `[MASK]` 的情况下仍可使用。分别在未遮盖句子与遮盖句子上预测。两者都应产生合理的词元分布，因为模型在训练中见过两种模式。

### 第 4 步：微调输出头

在玩具情感数据集上，用分类头替换 MLM 头。只训练输出头，冻结编码器。这就是每种 BERT 应用都遵循的模式。

## 学以致用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型就是经过微调的 BERT。** `sentence-transformers` 中的 `all-MiniLM-L6-v2` 等模型，是使用对比损失训练的 BERT。编码器相同，改变的是损失函数。

**交叉编码器重排器也是经过微调的 BERT。** 对 `[CLS] query [SEP] doc [SEP]` 执行样本对分类。查询与文档之间的双向注意力，正是交叉编码器质量高于双编码器的原因。

**2026 年不应选择 BERT 的场景。** 任何生成任务。编码器无法合理地自回归生成词元。此外，对于参数量小于 10 亿、可由小型解码器以更大灵活性达到相同质量的任务（Phi-3-Mini、Qwen2-1.5B），也不应选择它。

## 交付成果

见 `outputs/skill-bert-finetuner.md`。该技能会为新的分类或抽取任务界定 BERT 微调方案，包括骨干网络选择、输出头规格、数据、评估与停止条件。

## 练习

1. **简单。** 运行 `code/main.py`，打印 1 万个词元上的掩码分布。确认约 15% 被选中，其中约 80% 会变成 `[MASK]`。
2. **中等。** 实现全词掩码：如果一个词被拆成多个子词，就要么同时遮盖全部子词，要么全部保留。在包含 500 个句子的语料库上测量它是否改善 MLM 准确率。
3. **困难。** 在公共数据集的 1 万个句子上训练微型 BERT（两层，d=64）。使用 `[CLS]` 词元针对 SST-2 情感分类进行微调，并与参数量相同的仅解码器基线比较——哪一个胜出？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | “掩码语言建模” | 训练信号：随机把 15% 的词元替换为 `[MASK]`，预测原始词元。 |
| 双向 | “同时看两边” | 编码器注意力没有因果掩码——每个位置都能看到其他所有位置。 |
| `[CLS]` | “池化词元” | 添加到每个序列开头的特殊词元；其最终嵌入用作句子级表示。 |
| `[SEP]` | “片段分隔符” | 分隔成对序列（例如查询/文档、句子 A/B）。 |
| NSP | “下一句预测” | BERT 的第二个预训练任务；RoBERTa 证明它没有作用，2019 年后被弃用。 |
| 微调 | “适配任务” | 让编码器大体保持冻结，在顶部为下游任务训练一个小型输出头。 |
| 交叉编码器 | “重排器” | 同时接收查询与文档，并输出相关性分数的 BERT。 |
| ModernBERT | “2024 年更新版” | 使用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力和 8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin 等（2018），BERT：用于语言理解的深度双向 Transformer 预训练](https://arxiv.org/abs/1810.04805)——原始论文。
- [Liu 等（2019），RoBERTa：稳健优化的 BERT 预训练方法](https://arxiv.org/abs/1907.11692)——如何正确训练 BERT；证明 NSP 无益。
- [Clark 等（2020），ELECTRA：把文本编码器预训练为判别器而非生成器](https://arxiv.org/abs/2003.10555)——在计算量相同时，以替换词元检测胜过 MLM。
- [Warner 等（2024），更智能、更出色、更快速、更长：现代双向编码器](https://arxiv.org/abs/2412.13663)——ModernBERT 论文。
- [Hugging Face `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)——经典编码器参考实现。
