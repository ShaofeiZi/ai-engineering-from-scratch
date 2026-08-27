# 多头注意力

> 一个注意力头一次学习一种关系，八个头就能学习八种。增加头数不增加总参数量，尽管多用。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 02（从零实现自注意力）
**Time:** 约 75 分钟

## 问题

单个自注意力头只计算一个注意力矩阵。这个矩阵只能捕捉一种关系——通常是最有助于降低训练损失的那一种。如果数据中同时纠缠着主谓一致、共指、长距离篇章关系和句法分块，一个头会把它们混进同一个 softmax 分布，丢掉一半信号。

Vaswani 2017 年论文给出的解决方法是：并行运行多个注意力函数，每个函数拥有独立的 Q、K、V 投影，再拼接输出。每个头在维度为 `d_model / n_heads` 的较小子空间中工作。总参数量不变，表达能力却得到提升。

多头注意力是 2026 年每个 Transformer 的默认配置。唯一需要争论的是究竟使用多少个头，以及键和值是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 概念

![多头注意力的拆分、计算与拼接](../assets/multi-head-attention.svg)

**拆分。** 取 `X`，其形状为 `(N, d_model)`。将其投影成形状均为 `(N, d_model)` 的 Q、K、V，再重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`，最后转置为 `(n_heads, N, d_head)`。

**并行计算注意力。** 在每个头内部运行缩放点积注意力。每个头产生形状为 `(N, d_head)` 的结果。各个头作用于嵌入的不同子空间，在注意力计算本身进行期间互不通信。

**拼接并投影。** 把各个头重新堆叠为 `(N, d_model)`，再乘以学习式输出矩阵 `W_o`，其形状为 `(d_model, d_model)`。`W_o` 是不同头彼此混合的地方。

**它为何有效。** 每个头都可以独立专门化，而不必与其他头争夺表示容量。2019～2024 年的探测研究发现了不同的头部角色：位置头、关注前一个词元的头、复制头、命名实体头，以及支撑上下文学习的归纳头。

**2026 年各种变体的演进脉络：**

| 变体 | Q 头数 | K/V 头数 | 使用者 |
|---------|---------|-----------|---------|
| 多头注意力（MHA） | N | N | GPT-2、BERT、T5 |
| 多查询注意力（MQA） | N | 1 | PaLM、Falcon |
| 分组查询注意力（GQA） | N | G（例如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| 多头潜在注意力（MLA） | N | 压缩为低秩表示 | DeepSeek-V2、V3 |

GQA 是现代默认方案，因为它能把 KV 缓存缩小 `N/G` 倍，同时几乎保持完整质量。MLA 更进一步，把 K/V 压缩进潜在空间，计算注意力时再投影回来——以更多浮点运算换取大幅内存节省。

```figure
multihead-split
```

## 动手构建

### 第 1 步：在已有单头注意力上拆分多个头

取出第 02 课的 `SelfAttention`，在外层加入拆分/拼接操作。NumPy 实现见 `code/main.py`，其逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 加一次 transpose，不需要循环。PyTorch 在 `nn.MultiheadAttention` 内部做的正是这件事。

### 第 2 步：逐头执行缩放点积注意力

每个头都取得自己对应的 Q、K、V 切片。注意力运算变为批量矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 看到的是形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单次批量矩阵乘法。增加头数几乎没有额外成本。

### 第 3 步：分组查询注意力变体

只有键和值的投影发生变化。Q 拥有 `n_heads` 组；K 和 V 只有 `n_kv_heads < n_heads` 组，再重复到与 Q 匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时可以节省内存，因为 KV 缓存中只需存放 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个查询头与 8 个 KV 头——把缓存缩小 8 倍。

### 第 4 步：探测每个头学到了什么

在短句上运行带 4 个头的 MHA。对每个头打印形状为 `(N, N)` 的注意力矩阵。即使随机初始化，你也会看到不同的头选择不同结构——一部分来自信号，一部分来自子空间中的旋转对称性。

## 学以致用

在 PyTorch 中，只需一行：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**应该使用多少个头？** 2026 年生产模型的经验法则：

| 模型大小 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型（约 125M） | 768 | 12 | 64 |
| 基础（约 350M） | 1024 | 16 | 64 |
| 大型（约 1B） | 2048 | 16 | 128 |
| 前沿（约 70B） | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它决定单个头可以“看到”多少信息。低于 32，注意力头会开始受到缩放因子 `sqrt(d_head)` 的影响而彼此竞争；高于 256，则会失去“众多小型专家”的优势。

## 交付成果

见 `outputs/skill-mha-configurator.md`。该技能会根据参数预算、序列长度和部署目标，为新的 Transformer 推荐注意力头数、KV 头数与投影策略。

## 练习

1. **简单。** 取 `code/main.py` 中的 MHA，把 `n_heads` 从 1 改为 16，同时保持 `d_model=64` 不变。绘制微型单层模型在合成复制任务上的损失。更多头会改善、趋于平稳，还是损害结果？
2. **中等。** 实现 MQA（所有查询头共享一个 KV 头）。测量相比完整 MHA 减少了多少参数，并计算 N=2048 时推理 KV 缓存缩小多少。
3. **困难。** 实现一个微型多头潜在注意力：把 K、V 压缩为秩 `r` 的潜变量，在 KV 缓存中存储潜变量，再在计算注意力时解压缩。当 `r` 取多大时，缓存内存可以低于完整 MHA 的八分之一，同时质量保持在验证困惑度相差 1 比特以内？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 头 | “一条独立的注意力回路” | 一个维度为 `d_head = d_model / n_heads`、拥有独立注意力矩阵的 Q/K/V 投影。 |
| d_head | “头维度” | 每个头的隐藏宽度；生产模型中几乎总是 64 或 128。 |
| 拆分/合并 | “重塑技巧” | 在注意力运算前后执行 `(N, d_model) ↔ (n_heads, N, d_head)` 的 reshape + transpose。 |
| W_o | “输出投影” | 拼接各个头后应用的 `(d_model, d_model)` 矩阵；各头在这里混合。 |
| MQA | “一个 KV 头” | 多查询注意力：共享一组 K/V 投影。KV 缓存最小，但会损失一些质量。 |
| GQA | “Llama 2 之后的默认方案” | 满足 `n_kv_heads < n_heads` 的分组查询注意力；通过重复与 Q 匹配。 |
| MLA | “DeepSeek 的技巧” | 多头潜在注意力：把 K、V 压缩为低秩潜变量，在计算注意力时解压缩。 |
| 归纳头 | “上下文学习背后的回路” | 一对能够发现先前出现位置，并复制其后续内容的注意力头。 |

## 延伸阅读

- [Vaswani 等（2017），Attention Is All You Need 第 3.2.2 节](https://arxiv.org/abs/1706.03762)——原始多头注意力规范。
- [Shazeer（2019），快速 Transformer 解码：一个写入头就够了](https://arxiv.org/abs/1911.02150)——MQA 论文。
- [Ainslie 等（2023），GQA：从多头检查点训练广义多查询 Transformer](https://arxiv.org/abs/2305.13245)——如何在训练后把 MHA 转换为 GQA。
- [DeepSeek-AI（2024），DeepSeek-V2 技术报告](https://arxiv.org/abs/2405.04434)——MLA 及其在缓存内存上胜过 MHA/GQA 的原因。
- [Olsson 等（2022），上下文学习与归纳头](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)——从机制层面观察注意力头究竟在做什么。
