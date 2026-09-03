# T5、BART——编码器—解码器模型

> 编码器负责理解，解码器负责生成。把二者重新组合起来，就得到专为输入 → 输出任务设计的模型：翻译、摘要、改写、转写。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）、阶段 7 · 07（GPT）
**Time:** 约 45 分钟

## 问题

仅解码器 GPT 与仅编码器 BERT 分别为不同目标精简了 2017 年的架构。但许多任务天然是输入—输出形式：

- 翻译：英语 → 法语。
- 摘要：5000 词文章 → 200 词摘要。
- 语音识别：音频词元 → 文本词元。
- 结构化抽取：普通文本 → JSON。

对于这类任务，编码器—解码器是最自然的选择。编码器生成源序列的稠密表示，解码器在每一步通过交叉注意力关注该表示，并生成输出。训练目标是输出侧错位一位后的序列。损失与 GPT 相同，只是额外以编码器输出为条件。

两篇论文确立了现代方法：

1. **T5**（Raffel 等，2019）。“文本到文本迁移 Transformer”。把每项自然语言处理任务都重新表述为文本输入、文本输出；采用单一架构、单一词表、单一损失。通过掩码跨度预测预训练（破坏输入中的若干跨度，再在输出中解码这些跨度）。
2. **BART**（Lewis 等，2019）。“双向与自回归 Transformer”。去噪自动编码器：以多种方式破坏输入（打乱、遮盖、删除、旋转），再要求解码器重建原文。

2026 年，在输入结构至关重要的任务中，编码器—解码器形式仍然存在：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 一些输入上下文与编辑结果结构明确分离的代码补全/修复模型。
- 用于结构化推理任务的 Flan-T5 及其变体。

仅解码器模型占据了聚光灯，但编码器—解码器从未消失。

## 概念

![带交叉注意力的编码器—解码器](../../../../../../phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键在于：每个输入只运行一次编码器。解码器以自回归方式运行，但每一步都对*同一份*编码器输出执行交叉注意力。对编码器输出进行缓存，是长输入场景中无需代价即可获得的加速。

### T5 预训练——跨度破坏

随机选择输入中的若干跨度（平均长度 3 个词元，共占 15%），将每个跨度替换为唯一的哨兵词元：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出被破坏的跨度，并在前面加上对应哨兵：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

与预测整个序列相比，这种训练信号成本更低。在 T5 论文的消融实验中，它可以与 MLM（BERT）和前缀语言模型（UniLM）竞争。

### BART 预训练——多噪声去噪

BART 尝试五种加噪函数：

1. 词元掩码。
2. 词元删除。
3. 文本填空（遮盖一个跨度，由解码器推断正确长度与内容）。
4. 句子置换。
5. 文档旋转。

文本填空 + 句子置换的组合取得了最好的下游结果。解码器始终重建完整原文。BART 的输出是完整序列，而不只是被破坏的跨度，因此预训练计算量高于 T5。

### 推理

与 GPT 一样采用自回归生成。可以使用贪心、束搜索或 top-p 采样。翻译与摘要的输出分布比聊天更窄，因此通常使用宽度为 4～5 的束搜索。

### 2026 年何时选择哪种变体

| 任务 | 使用编码器—解码器？ | 原因 |
|------|------------------|-----|
| 翻译 | 通常是 | 源序列明确，输出分布固定，束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入与输出模态不同；编码器负责塑造音频特征 |
| 聊天/推理 | 否，仅解码器 | 没有持久的“输入”——对话本身就是序列 |
| 代码补全 | 通常否 | 长上下文仅解码器胜出；Qwen 2.5 Coder 等代码模型都是仅解码器 |
| 摘要 | 两者都可以 | BART、PEGASUS 胜过早期仅解码器基线；现代仅解码器大语言模型已能追平 |
| 结构化抽取 | 两者都可以 | T5 很自然，因为“文本 → 文本”可以容纳任意输出格式 |

约从 2022 年开始，仅解码器模型接管了以往由编码器—解码器承担的任务，原因是：（a）指令微调后的仅解码器大语言模型可以通过提示泛化到任何任务；（b）一种架构比两种架构更易扩展；（c）RLHF 以解码器为基础。当输入模态不同（语音、图像），或束搜索质量至关重要时，编码器—解码器仍保持优势。

```figure
encoder-decoder
```

## 动手构建

见 `code/main.py`。我们会为玩具语料库实现 T5 风格的跨度破坏——这是本课最值得掌握的单项内容，因为此后的每种编码器—解码器预训练方案都能看到它的影子。

### 第 1 步：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标采用 T5 格式：`<sent0> span0 <sent1> span1 ...`。被破坏的输入则在跨度原位置交错放置未修改词元与哨兵词元。

### 第 2 步：验证往返还原

根据被破坏的输入和目标重建原始句子。如果破坏过程可逆，前向传播才有明确定义。这是一项健全性检查——真实训练不会执行还原，但这项廉价测试可以发现跨度记录中的差一错误。

### 第 3 步：BART 加噪

实现五种函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两种并展示结果。

## 学以致用

Hugging Face 参考用法：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧是把任务名称也放进输入文本。同一个模型可以处理数十种任务，因为每项任务都是文本输入、文本输出。到 2026 年，指令微调的仅解码器模型已经推广了这种模式，但最先将其系统化的是 T5。

## 交付成果

见 `outputs/skill-seq2seq-picker.md`。该技能会根据输入—输出结构、延迟和质量目标，在编码器—解码器与仅解码器架构之间作出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对一个包含 30 个词元的句子应用跨度破坏；验证将源序列中的非哨兵词元与目标中解码出的跨度拼接后，可以还原原句。
2. **中等。** 实现 BART 的 `text_infill` 噪声：把随机跨度替换为单个 `<mask>` 词元，由解码器推断正确的跨度长度与内容。展示一个示例。
3. **困难。** 在包含 200 个样本对的英语 → Pig Latin 小型语料库上微调 `flan-t5-small`，在包含 50 个样本对的留出集上测量 BLEU。使用相同计算量与微调 `Llama-3.2-1B` 的结果比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器—解码器 | “序列到序列 Transformer” | 两个堆栈：处理输入的双向编码器，以及带交叉注意力、处理输出的因果解码器。 |
| 交叉注意力 | “源序列与目标序列交流的位置” | 解码器的 Q × 编码器的 K/V，是编码器信息进入解码器的唯一位置。 |
| 跨度破坏 | “T5 的预训练技巧” | 用哨兵词元替换随机跨度；解码器输出这些跨度。 |
| 去噪目标 | “BART 的游戏” | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵词元 | “`<extra_id_N>` 占位符” | 在源序列中标记被破坏跨度，并在目标序列中重新标记它们的特殊词元。 |
| Flan | “经过指令微调的 T5” | 在 1800 多项任务上微调的 T5，使编码器—解码器也擅长遵循指令。 |
| 束搜索 | “解码策略” | 每一步保留排名前 k 的部分序列；翻译和摘要的标准方案。 |
| 教师强制 | “训练时输入” | 训练时向解码器提供上一个真实输出词元，而不是采样得到的词元。 |

## 延伸阅读

- [Raffel 等（2019），使用统一文本到文本 Transformer 探索迁移学习的极限](https://arxiv.org/abs/1910.10683)——T5。
- [Lewis 等（2019），BART：用于自然语言生成、翻译和理解的去噪序列到序列预训练](https://arxiv.org/abs/1910.13461)——BART。
- [Chung 等（2022），扩展指令微调语言模型](https://arxiv.org/abs/2210.11416)——Flan-T5。
- [Radford 等（2022），通过大规模弱监督实现稳健语音识别](https://arxiv.org/abs/2212.04356)——Whisper，2026 年典型的编码器—解码器。
- [Hugging Face `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)——参考实现。
