# 序列到序列模型

> 两个 RNN 扮演翻译器。它们遇到的瓶颈，正是注意力机制存在的原因。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 08（用于文本的 CNN + RNN）、阶段 3 · 11（PyTorch 入门）
**Time:** 约 75 分钟

## 问题

分类把一个变长序列映射为单一标签，翻译则把一个变长序列映射为另一个变长序列。输入和输出位于不同词表中，甚至可能属于不同语言，长度也不保证一致。

序列到序列架构（Sutskever、Vinyals、Le，2014）用一套刻意保持简单的方法解决了这个问题：两个 RNN。一个读取源句子并生成定长上下文向量，另一个读取该向量并逐词元生成目标句子。它们就是你在第 08 课中写过的代码，只是以不同方式组合在一起。

学习它有两个原因。首先，上下文向量瓶颈是自然语言处理中最具教学价值的失败案例，它解释了注意力机制和 Transformer 所有优势的由来。其次，这套训练方法（教师强制、计划采样、推理时的束搜索）仍然适用于包括大语言模型在内的每一种现代生成系统。

## 概念

**编码器。** 一个读取源句子的 RNN。它最后的隐藏状态就是**上下文向量**——整个输入的定长摘要。据称除了源句子外，什么都没有丢失。

**解码器。** 另一个用上下文向量初始化的 RNN。每一步都以上一个生成的词元作为输入，并输出目标词表上的概率分布。通过采样或 argmax 选择下一个词元，再把它送回模型。重复这一过程，直到生成 `<EOS>` 词元或达到最大长度。

**训练：** 在解码器的每个步骤计算交叉熵损失，再沿序列求和。通过两个网络执行标准的随时间反向传播。

**教师强制。** 训练时，解码器在步骤 `t` 的输入是位置 `t-1` 上的*真实*词元，而不是解码器自己上一步的预测。这样可以稳定训练；如果不这样做，早期错误会层层累积，模型始终无法学会。推理时必须使用模型自己的预测，因此训练分布与推理分布之间始终存在差距。这种差距称为**暴露偏差**。

**瓶颈。** 编码器从源文本学到的一切，都必须挤进唯一一个上下文向量。长句会丢失细节，罕见词会变得模糊，语序重排（chat noir 与 black cat）也只能靠记忆，而不能现场计算。

注意力机制（第 10 课）允许解码器查看编码器的*每一个*隐藏状态，而不只是最后一个，从而解决这个问题。这就是它的核心主张。

```figure
lstm-gates
```

## 动手构建

### 第 1 步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]`——输入的每个位置对应一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]`——最后一个时间步的状态。第 08 课说过“用于分类时，在 outputs 上执行池化”。这里则把最后的隐藏状态保留为上下文向量，忽略逐时间步输出。

### 第 2 步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次只调用一步。输入是一批单词元和当前隐藏状态；输出是下一个词元的词表 logits 与更新后的隐藏状态。

### 第 3 步：采用教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

这里有两个值得点明的参数。`ignore_index=0` 会跳过填充词元上的损失。`teacher_forcing_ratio` 是每一步使用真实词元而非模型预测的概率。可以从 1.0（完全教师强制）开始训练，再逐渐退火到约 0.5，以缩小暴露偏差。

### 第 4 步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每一步都选择概率最高的词元，因此可能走入歧途：一旦选定某个词元，就无法收回。**束搜索**会保留得分最高的 `k` 个部分序列，最后再选出得分最高的完整序列。束宽通常设为 3～5。

### 第 5 步：直观展示瓶颈

在玩具复制任务上训练模型：源序列为 `[a, b, c, d, e]`，目标序列同样为 `[a, b, c, d, e]`。逐渐增加序列长度并观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损记住包含 40 个词元的输入。信息原本存在于编码器的每个时间步，但解码器只能看到最后一个状态。注意力机制会直接解决这个问题。

## 学以致用

PyTorch 提供 `nn.Transformer` 和基于 `nn.LSTM` 的序列到序列模板。Hugging Face 的 `transformers` 库则提供在数十亿词元上训练的完整编码器—解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器—解码器已经用 Transformer 取代 RNN。其高层形态（编码器、解码器、逐词元生成）与 2014 年的序列到序列论文完全相同，只是每个块内部的机制不同。

### 何时仍应选择基于 RNN 的序列到序列模型

对于新项目，几乎永远不应选择。具体例外包括：

- 流式翻译：一次读取一个输入词元，并保持有界内存。
- 设备端文本生成：Transformer 的内存成本高得无法承受。
- 教学。理解编码器—解码器瓶颈，是理解 Transformer 为何胜出的最快途径。

### 暴露偏差及其缓解方法

- **计划采样。** 在训练期间逐步降低教师强制比例，让模型学会从自己的错误中恢复。
- **最小风险训练。** 使用句子级 BLEU 分数而非词元级交叉熵进行训练，更接近真正的目标。
- **强化学习微调。** 用指标奖励序列生成器，现代大语言模型的 RLHF 也采用这种思路。

这三种方法同样适用于基于 Transformer 的生成。

## 交付成果

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在目标等于源序列的输入—输出对上训练 GRU 序列到序列模型，测量长度为 5、10、20 时的准确率，并复现瓶颈。
2. **中等。** 增加束宽为 3 的束搜索解码。在小型平行语料库上测量相对于贪心解码的 BLEU，记录束搜索在哪些位置胜出（通常是最后几个词元），以及在哪些位置没有差异。
3. **困难。** 在包含 1 万个样本对的释义数据集上微调 `facebook/bart-base`。比较微调模型与基础模型在留出输入上的 beam-4 输出，报告 BLEU，并挑选 10 个定性示例。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器 | 输入 RNN | 读取源序列，生成逐时间步隐藏状态和最终上下文向量。 |
| 解码器 | 输出 RNN | 由上下文向量初始化，每次生成一个目标词元。 |
| 上下文向量 | 摘要 | 编码器最终的隐藏状态，大小固定，也是注意力机制要解决的瓶颈。 |
| 教师强制 | 使用真实词元 | 训练时输入上一个真实词元，可以稳定学习。 |
| 暴露偏差 | 训练/测试差距 | 模型只在真实词元上训练，从未练习如何从自己的错误中恢复。 |
| 束搜索 | 更好的解码 | 每一步保留得分最高的 k 个部分序列，而不是贪心地立即作出不可逆选择。 |

## 延伸阅读

- [Sutskever、Vinyals、Le（2014），使用神经网络进行序列到序列学习](https://arxiv.org/abs/1409.3215)——原始 seq2seq 论文，只有四页。
- [Cho 等（2014），使用 RNN 编码器—解码器学习统计机器翻译的短语表示](https://arxiv.org/abs/1406.1078)——提出 GRU 和编码器—解码器框架。
- [Bahdanau、Cho、Bengio（2014），通过联合学习对齐和翻译实现神经机器翻译](https://arxiv.org/abs/1409.0473)——注意力机制论文，请在本课之后立即阅读。
- [PyTorch 从零开始学习 NLP 教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)——可以实际构建的 seq2seq + 注意力代码。
