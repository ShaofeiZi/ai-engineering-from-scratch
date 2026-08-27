# 用于文本的 CNN 与 RNN

> 卷积学习 n 元语法，循环记住历史。二者都已被注意力取代，却依然适用于资源受限的硬件。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 3 · 11（PyTorch 入门）、阶段 5 · 03（词嵌入）、阶段 4 · 02（从零实现卷积）
**Time:** 约 75 分钟

## 问题

TF-IDF 和 Word2Vec 生成的是忽略词序的扁平向量。建立在它们之上的分类器无法区分 `dog bites man` 与 `man bites dog`，而词序有时正是关键信号。

在 Transformer 出现之前，有两个架构家族填补了这一缺口。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为 3 的滤波器就是一个可学习的三元语法检测器：它跨越三个词并输出分数。叠加不同宽度（2、3、4、5）的滤波器以检测多尺度模式，再通过最大池化得到定长表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 每次处理一个词元，并维护向前携带信息的隐藏状态。顺序执行、具备记忆、支持灵活的输入长度。它在 2014 至 2017 年间主导序列建模，随后注意力机制出现了。

本课将构建这两种架构，再指出促使注意力机制诞生的失败模式。

## 概念

**TextCNN**（Kim，2014）。先嵌入词元。宽度为 `k` 的一维卷积让一个滤波器滑过连续的 `k` 元嵌入，生成特征图。对特征图执行全局最大池化，选出最强激活。把多种滤波器宽度的最大池化结果拼接起来，再送入分类头。

它为何有效？一个滤波器就是一个可学习的 n 元语法。最大池化与位置无关，因此“not good”位于评论开头或中部时都会触发同一特征。三种宽度各配 100 个滤波器，就能得到 300 个学习式 n 元语法检测器。训练可以并行，不存在顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态为 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在所有时间步共享。时间步 `T` 的隐藏状态是整个前缀的摘要。用于分类时，可以在 `h_1 ... h_T` 上执行池化（最大值、均值或末状态）。

普通 RNN 会遭遇梯度消失。**LSTM** 增加门机制，用于决定遗忘什么、保存什么、输出什么，从而稳定长序列中的梯度。**GRU** 把 LSTM 简化为两个门；参数更少，表现相近。

**双向 RNN** 同时运行一个正向 RNN 和一个反向 RNN，再拼接隐藏状态。每个词元的表示都能看到左右两侧的上下文，因此它对于标注任务至关重要。

```figure
rnn-unroll
```

## 动手构建

### 第 1 步：使用 PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 会把 `[batch, seq_len, embed_dim]` 重排为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间轴视为通道。无论输入长度如何，池化后的输出大小都固定不变。

### 第 2 步：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

这里在序列上执行最大池化，而不是只取最后状态。对于分类任务，最大池化通常优于最后隐藏状态，因为长序列末尾的信息往往会主导最后状态。

### 第 3 步：梯度消失演示（直觉）

没有门控的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测词元 `A` 是否在序列中的任意位置出现。如果 `A` 位于第 1 个位置，而序列长 100 个词元，损失的梯度就必须反向经过循环权重的 99 次乘法。权重小于 1，梯度便会消失；大于 1，梯度就会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 使用一个仅通过加性交互穿越网络的**细胞状态**来修复这个问题（遗忘门会以乘法方式缩放它，但梯度仍可沿着这条“高速公路”流动）。GRU 用更少的参数实现类似机制。二者都能在超过 100 个时间步的序列中稳定训练。

### 第 4 步：为何这仍然不够

即便使用 LSTM，仍有三个问题存在。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN，需要依次执行 1000 个前向/反向步骤，无法跨时间并行。
2. **编码器—解码器结构中的定长上下文向量。** 解码器只能看到编码器最后的隐藏状态，而整个输入都被压缩在其中。长输入会丢失细节，第 09 课会直接讨论这一点。
3. **远距离依赖的准确率上限。** LSTM 胜过普通 RNN，但在跨越 200 多个步骤传播特定信息时仍然吃力。

注意力机制解决了这三个问题，Transformer 则彻底抛弃循环。第 10 课是转折点。

## 学以致用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都已达到生产可用水平，训练代码也很标准。

Hugging Face 提供预训练嵌入，可以直接插入作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

符合以下约束时可以使用这些架构：

- **边缘端/设备端推理。** 搭配 GloVe 嵌入的 TextCNN 比 Transformer 小 10～100 倍。部署目标是手机时，应选择这一技术栈。
- **流式/在线分类。** RNN 每次处理一个词元；Transformer 需要完整序列。对于实时到达的文本，LSTM 仍然更合适。
- **用小模型建立基线。** 在新任务上快速迭代。使用 CPU，5 分钟即可训练一个 TextCNN。
- **数据有限的序列标注。** 对于 1000～10000 个带标签句子，BiLSTM-CRF（第 06 课）仍是生产级 NER 架构。

其他所有情况都应使用 Transformer。

## 交付成果

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在你自己编造的三分类玩具数据集上训练 TextCNN。验证滤波器宽度组合（2、3、4）的平均 F1 优于只使用单一宽度（3）。
2. **中等。** 为 LSTM 分类器实现最大池化、均值池化与末状态池化。在小型数据集上比较，记录哪种池化胜出，并推测原因。
3. **困难。** 构建 BiLSTM-CRF NER 标注器（结合第 06 课与本课）。在 CoNLL-2003 上训练，并与纯 CRF 基线及 BERT 微调方案比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 用于文本的 CNN | 在词嵌入上堆叠一维卷积，并执行全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 增加输入门、遗忘门、输出门和细胞状态，可以在长序列上稳定训练。 |
| GRU | 更简单的 LSTM | 用两个门替代三个门。准确率相近，参数更少。 |
| 双向 | 两个方向 | 拼接正向与反向 RNN，使每个词元都能看到上下文两侧。 |
| 梯度消失 | 训练信号消亡 | 普通 RNN 反复乘以小于 1 的权重，导致早期时间步的梯度趋近于零。 |

## 延伸阅读

- [Kim, Y.（2014），用于句子分类的卷积神经网络](https://arxiv.org/abs/1408.5882)——TextCNN 论文，八页，可读性很好。
- [Hochreiter, S. 与 Schmidhuber, J.（1997），长短期记忆](https://www.bioinf.jku.at/publications/older/2604.pdf)——LSTM 论文，出乎意料地清晰。
- [Olah, C.（2015），理解 LSTM 网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)——让所有人都能理解 LSTM 的经典图解。
