# 注意力机制——突破性进展

> 解码器不再费力盯着一份压缩摘要，而是开始查看整个源序列。此后的一切，都是注意力加工程优化。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 09（序列到序列模型）
**Time:** 约 45 分钟

## 问题

第 09 课以一次实际测得的失败收尾：在玩具复制任务上训练的 GRU 编码器—解码器，在长度为 5 时准确率为 89%，到长度 80 时却跌至接近随机水平。原因在于结构，而不是训练错误：编码器获取的每一比特信息都必须装进一个定长隐藏状态，解码器再也看不到其他内容。

Bahdanau、Cho 与 Bengio 在 2014 年发表了一个三行代码的修复方法。不要只把编码器的最终状态交给解码器，而应保留编码器的每个状态。在解码器的每一步，计算编码器状态的加权平均值，其中权重表示“解码器此刻需要查看编码器位置 `i` 的程度”。这个加权平均值就是上下文，并且会在解码器的每一步发生变化。

这就是全部思想。Transformer 扩展了它，自注意力把它应用于单一序列，多头注意力则并行执行它。但 2014 年的版本已经打破了瓶颈；理解它之后，从这里转向 Transformer 主要是工程问题，而不是概念问题。

## 概念

![Bahdanau 注意力：解码器查询编码器的全部状态](../../../../../../phases/05-nlp-foundations-to-advanced/10-attention-mechanism/assets/attention.svg)

在解码器的每个步骤 `t`：

1. 使用解码器上一步的隐藏状态 `s_{t-1}` 作为**查询**。
2. 将它与编码器的每个隐藏状态 `h_1, ..., h_T` 评分。每个编码器位置得到一个标量。
3. 对分数执行 softmax，得到总和为 1 的注意力权重 `α_{t,1}, ..., α_{t,T}`。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`，即编码器状态的加权平均值。
5. 解码器接收 `c_t` 和上一个输出词元，生成下一个词元。

关键就在这个加权平均值。当解码器需要把“Je”翻译成“I”时，它会为“Je”对应的编码器状态赋予高权重，降低其他位置的权重。需要生成“not”时，它会提高“pas”的权重。上下文向量会在每一步重新塑形。

## 形状（每个人都会踩坑的地方）

每个人第一次实现注意力时都会在这里犯错，请慢慢读。

| 对象 | 形状 | 说明 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果使用 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 在所有 `i` 上执行 softmax 后得到 |
| 上下文向量 `c_t` | `(d_h,)` | 与一个编码器状态形状相同 |

**Bahdanau（加性）评分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形状为 `(d_s,)`，`h_i` 的形状为 `(d_h,)`。
- `W_a` 的形状为 `(d_attn, d_s)`，`U_a` 的形状为 `(d_attn, d_h)`。
- 二者在 tanh 内相加后，形状为 `(d_attn,)`。
- `v_α` 的形状为 `(d_attn,)`。与 `v_α` 做内积后会压缩成一个标量。**这就是 `v_α` 的作用。** 它并不神秘，只是把注意力维向量投影为标量分数。

**Luong（乘性）评分。** 有三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`，这是硬性约束。如果编码器是双向的，就跳过这种形式。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 的形状为 `(d_s, d_h)`。它消除了维度相等的约束。
- `concat`：本质上就是 Bahdanau 形式。由于前两种形式更便宜，现在很少使用。

**一个值得点明的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词之前的解码器状态），Luong 使用 `s_t`（生成后的状态）。混淆二者会产生极其难以调试的微妙错误梯度。请选择一篇论文，并始终遵循它的约定。

```figure
attention-heatmap
```

## 动手构建

### 第 1 步：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

请对照上表检查形状。`encoder_states` 的形状是 `(T_enc, d_h)`；`projected_enc` 的形状是 `(T_enc, d_attn)`；`projected_dec` 的形状是 `(d_attn,)`，会通过广播参与运算；`combined` 的形状是 `(T_enc, d_attn)`；`scores` 的形状是 `(T_enc,)`；`weights` 的形状是 `(T_enc,)`；`context` 的形状是 `(d_h,)`。可以交付了。

### 第 2 步：Luong 点积与通用形式

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每种形式只需三行代码。这正是 Luong 论文产生影响的原因：在大多数任务上达到相同准确率，代码却少得多。

### 第 3 步：完整的数值示例

给定三个编码器状态（大致对应“cat”“sat”“mat”）和一个与第一个状态最对齐的解码器状态，注意力分布会集中在位置 0。如果解码器状态转而靠近第三个编码器状态，注意力也会移向位置 2，上下文向量会随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。接着让解码器状态更靠近第三个编码器状态，观察权重如何转移。就是这么简单：注意力是显式对齐。

### 第 4 步：它为何是通向 Transformer 的桥梁

把上面的语言改写为 Q/K/V：

- **Query** = 解码器状态 `s_{t-1}`
- **Key** = 编码器状态（用于评分的对象）
- **Value** = 编码器状态（用于加权求和的对象）

在经典注意力中，键和值是同一个对象。自注意力把二者分开：序列可以查询自身，并为 K 和 V 使用不同的学习式投影。多头注意力使用不同的学习式投影并行执行这一过程。Transformer 将整个阶段重复堆叠多次，并抛弃 RNN。

数学相同，形状也相同。从教学角度看，由 Bahdanau 注意力转向缩放点积注意力，主要只是记号发生了变化。

## 学以致用

PyTorch 与 TensorFlow 都直接提供注意力实现。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。查询批次包含 5 个位置，键/值批次包含 10 个位置，各为 128 维，共有 8 个头。`output` 是加入上下文后的新查询，`weights` 是可以可视化的 5×10 对齐矩阵。

### 经典注意力仍然重要的场景

- 教学。基于 RNN 的单头、单层版本让每个概念都清晰可见。
- Transformer 无法装入设备时的端侧序列任务。
- 阅读 2014～2017 年间的任何论文。不理解 Bahdanau 的约定，就会误读它们。
- 机器翻译中的细粒度对齐分析。即便在 Transformer 模型上，原始注意力权重仍是一种可解释性工具；要读懂它们，就必须知道其含义。

### 把注意力权重当作解释的陷阱

注意力权重看起来很容易解释。它们在各位置上的总和为一，可以绘图，数值高似乎就表示“看了这里”。评审者很喜欢它们。

但它们并不像看起来那样可解释。Jain 与 Wallace（2019）证明，在某些任务中，可以打乱注意力分布，甚至换成任意替代分布，而不改变模型预测。没有消融实验或反事实检查时，绝不要把注意力权重当作推理证据。

## 交付成果

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` 掩码，使批次中长度不一序列的填充词元获得零注意力权重。
2. **中等。** 为 Luong 的 `general` 形式增加多头注意力。把 `d_h` 拆成 `n_heads` 组，逐头运行注意力，再拼接结果。验证单头情况与先前实现一致。
3. **困难。** 在第 09 课的玩具复制任务上，训练带 Bahdanau 注意力的 GRU 编码器—解码器。绘制准确率随序列长度变化的曲线，并与无注意力基线比较。随着长度增加，二者差距应当不断扩大，从而证实注意力解除了瓶颈。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力 | 查看事物 | 对值序列进行加权平均，权重由查询与键的相似度计算。 |
| 查询、键、值 | QKV | 三种投影：Q 发问，K 用于匹配，V 是返回内容。 |
| 加性注意力 | Bahdanau | 前馈评分：`v^T tanh(W q + U k)`。 |
| 乘性注意力 | Luong 点积/通用形式 | 分数为 `q^T k` 或 `q^T W k`。成本更低，在大多数任务上准确率相同。 |
| 对齐矩阵 | 那张漂亮的图 | 形状为 `(T_dec, T_enc)` 的注意力权重网格，可以用来查看模型关注了什么。 |

## 延伸阅读

- [Bahdanau、Cho、Bengio（2014），通过联合学习对齐和翻译实现神经机器翻译](https://arxiv.org/abs/1409.0473)——原始论文。
- [Luong、Pham、Manning（2015），基于注意力的神经机器翻译有效方法](https://arxiv.org/abs/1508.04025)——三种评分变体及其比较。
- [Jain 与 Wallace（2019），注意力不是解释](https://arxiv.org/abs/1902.10186)——关于可解释性的注意事项。
- [《动手学深度学习》——Bahdanau 注意力](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)——可运行的 PyTorch 逐步讲解。
