# 注意力变体——滑动窗口、稀疏与差分注意力

> 完整注意力像一个圆。每个词元都能看到其他所有词元，内存则为此付出代价。四种变体改变了这个圆的形状，并省下过半成本。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 02（自注意力）、阶段 7 · 03（多头注意力）、阶段 7 · 12（KV 缓存 / Flash Attention）
**Time:** 约 60 分钟

## 问题

完整注意力的内存成本为 `O(N²)`，计算成本同样为 `O(N²)`，二者都随序列长度增长。对于上下文长度 128K 的 Llama 3 70B，每一层都有 160 亿个注意力条目，而模型共有 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，却没有改变算术成本——每个词元仍要关注其他所有词元。

有三类变体会改变注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA）。** 每个词元只关注固定窗口内的邻居，而不是完整前缀。内存和计算量降至 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 都采用它。
2. **稀疏/分块注意力。** 只对选定的位置对 `(i, j)` 评分，其余位置的权重被强制设为零。代表模型有 Longformer、BigBird、OpenAI sparse transformer。
3. **差分注意力。** 使用独立的 Q/K 投影计算两张注意力图，再从其中一张减去另一张。这样可以消除把权重泄漏给最前几个词元的“注意力汇点”。Microsoft 在 2024 年提出了 DIFF Transformer。

这些方法可以共存。2026 年的前沿模型经常混合使用：大多数层是 SWA-1024，每五层插入一个全局完整注意力层，再加入少量差分头以改善检索。Gemma 3 的 5:1 SWA 与全局注意力比例，是当前教科书式默认设置。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 上的每个查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）范围内的位置。窗口之外的词元在分数矩阵中被设为 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

当 `N = 8192`、`W = 1024` 时，分数矩阵预计有 1024 × 8192 个非零条目——减少 8 倍。

**SWA 也会缩小 KV 缓存。** 每一层只需保留最近 `W` 个词元的 K 与 V。对于类似 Gemma 3 的配置（窗口 1024，上下文 128K），KV 缓存会缩小 128 倍。

**质量代价。** 只使用 SWA 的 Transformer 不善于长距离检索。解决办法是在 SWA 层之间穿插完整注意力层。Gemma 3 使用 5:1 的 SWA:全局比例。Mistral 7B 采用因果 SWA 堆栈，信息会通过重叠窗口“向前流动”——每增加一层，有效感受野就向前扩展 `W`，经过 `L` 层后，模型可以关注此前 `L × W` 个词元。

### 稀疏/分块注意力

预先选择一个 `N × N` 稀疏模式。有三种经典形态：

- **局部 + 跨步稀疏（OpenAI sparse transformer）。** 关注最近 `W` 个词元，再关注之前每隔 `stride` 个位置的词元，以 `O(N · sqrt(N))` 计算量同时捕捉局部和长距离信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局词元（例如 `[CLS]`），这些词元关注所有位置，也被所有位置关注，再加随机稀疏连接。在质量相同时，实证可获得 2 倍上下文长度。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要，并在内核层面跳过全零块；与 FlashAttention 兼容。

稀疏注意力的关键在于内核工程。数学很简单（屏蔽分数矩阵），收益则来自根本不把零值条目加载到 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让自定义稀疏模式成为 PyTorch 的一等能力。

### 差分注意力（DIFF Transformer，2024）

普通注意力存在“注意力汇点”问题：softmax 强制每行之和为 1，因此没有明确关注对象的词元会把权重倾倒到第一个（或最前几个）词元上。这会偷走本应分配给真实内容的容量。

差分注意力通过计算**两张**注意力图并相减来解决问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中，`λ` 是学习得到的标量（通常为 0.5～0.8）。A1 捕捉真实内容权重，A2 捕捉汇点，相减后抵消汇点，把权重重新分配给相关词元。

Microsoft 2024 年报告的结果：困惑度降低 5%～10%，在相同训练长度下有效上下文延长 1.5～2 倍，大海捞针检索也更敏锐。

### 变体对比

| 变体 | 计算量 | KV 缓存 | 相对完整注意力的质量 | 生产用途 |
|---------|---------|----------|-----------------|----------------|
| 完整注意力 | O(N²) | 每层 O(N) | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | 困惑度差 0.1，搭配全局层时良好 | Gemma 2/3、Phi-3-Long |
| 局部 + 跨步稀疏 | O(N·√N) | 混合 | 与 SWA 相近 | OpenAI sparse transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 近似 O(N) | 混合 | 上下文扩大 2 倍时可匹配完整注意力 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 激活比例) | O(N) | 困惑度差小于 0.05 | DeepSeek-V3.2，2025 |
| 差分注意力 | O(2·N²) | O(2N) | 困惑度降低 5%～10% | DIFF Transformer、2026 年早期模型 |

```figure
gqa-kv-sharing
```

## 动手构建

见 `code/main.py`。我们实现一个因果掩码比较器，在玩具序列上并列展示完整注意力、SWA、局部 + 跨步稀疏注意力和差分注意力。

### 第 1 步：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

沿用第 07 课的基线。下三角区域有效，对角线上方权重为零。

### 第 2 步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

只有一个参数——`window`。当 `window >= n` 时，就恢复完整因果注意力；当 `window = 1` 时，每个词元只关注自身。

### 第 3 步：局部 + 跨步稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

它使用稠密局部窗口，并从当前位置向序列开头每隔 `stride` 个词元建立连接。增加层数后，感受野会以对数步数增长。

### 第 4 步：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

执行两次注意力，再用学习得到的混合系数相减。在代码中，我们会比较单一注意力与差分注意力的注意力汇点热力图，并观察汇点如何消失。

### 第 5 步：KV 缓存大小

打印每种变体在 `N = 131072` 时的逐层缓存大小。SWA 与稀疏变体可缩小 10～100 倍，差分注意力则会翻倍。应有意识地承担内存成本。

## 学以致用

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接收掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

它会被编译成自定义 Triton 内核。在常见模式下，速度与 FlashAttention-3 的差距不超过 10%，而掩码函数本身只是一个 Python 可调用对象。

**各变体的选择方式：**

- **纯完整注意力**——上下文不超过约 16K 的所有层，或检索质量至关重要的情况。
- **SWA + 全局混合**——长上下文（超过 32K），训练与推理受内存限制。这是 2026 年处理 32K 以上上下文的默认方案。
- **稀疏块注意力**——自定义内核、自定义模式，仅用于专门工作负载（检索、音频）。
- **差分注意力**——注意力汇点污染会造成损害的任何工作负载（长上下文 RAG、大海捞针）。

## 交付成果

见 `outputs/skill-attention-variant-picker.md`。该技能会根据目标上下文长度、检索要求和训练/推理计算特征，为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 会把每行最近 4 个词元之外的所有位置置零；验证 `window=n` 能逐比特还原完整因果注意力。
2. **中等。** 在第 07 课综合模型上实现 `window=1024` 的因果 SWA，在 tinyshakespeare 上训练 1000 步。与完整注意力相比，验证损失退化多少？峰值内存下降多少？
3. **困难。** 在综合模型中实现 Gemma 3 风格的 5:1 层混合（5 层 SWA、1 层全局）。在参数量相同的情况下，与纯 SWA 和纯全局基线比较损失、内存与生成质量。
4. **困难。** 为每个头实现带学习式 `λ` 的差分注意力。在一个合成检索任务（一根针、2000 个干扰项）上训练，并在参数量相同时与单注意力基线比较检索准确率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | “局部注意力” | 每个查询只关注最近 `W` 个词元；KV 缓存缩小为 `O(W)`。 |
| 有效感受野 | “模型能回看多远” | 在 `L` 层、窗口为 `W` 的 SWA 堆栈中，最多回看 `L × W` 个词元。 |
| Longformer / BigBird | “局部 + 全局 + 随机” | 使用少量始终参与注意力的全局词元的稀疏模式；早期长上下文方案。 |
| 原生稀疏注意力 | “DeepSeek 的内核技巧” | 学习块级稀疏性，在内核层面跳过全零块，同时保持质量。 |
| 差分注意力 | “两张图，相减一张” | DIFF Transformer：从第一张注意力图中减去学习式 `λ` 倍的第二张图，以消除注意力汇点。 |
| 注意力汇点 | “权重泄漏到词元 0” | softmax 强制每行之和为 1；无明确目标的查询会把权重倾倒到位置 0。 |
| FlexAttention | “用 Python 写掩码” | PyTorch 2.5+ API，可把任意掩码函数编译成类似 FlashAttention 的内核。 |
| 层类型混合 | “SWA 与全局注意力按 5:1 混合” | 在堆栈中交错使用稀疏与完整注意力层，以更低内存保持质量。 |

## 延伸阅读

- [Beltagy、Peters、Cohan（2020），Longformer：长文档 Transformer](https://arxiv.org/abs/2004.05150)——经典的滑动窗口 + 全局词元论文。
- [Zaheer 等（2020），Big Bird：处理更长序列的 Transformer](https://arxiv.org/abs/2007.14062)——局部 + 全局 + 随机注意力。
- [Child 等（2019），使用稀疏 Transformer 生成长序列](https://arxiv.org/abs/1904.10509)——OpenAI 的局部 + 跨步模式。
- [Gemma 团队（2024），Gemma 2：改进实用规模的开放语言模型](https://arxiv.org/abs/2408.00118)——1:1 SWA 与全局注意力混合。
- [Gemma 团队（2025），Gemma 3 技术报告](https://arxiv.org/abs/2503.19786)——窗口为 1024、如今成为教科书默认方案的 5:1 混合。
- [Ye 等（2024），差分 Transformer](https://arxiv.org/abs/2410.05258)——DIFF Transformer 论文。
- [Yuan 等（2025），原生稀疏注意力](https://arxiv.org/abs/2502.11089)——DeepSeek-V3.2 的学习式稀疏注意力。
- [PyTorch——FlexAttention 博客与文档](https://pytorch.org/blog/flexattention/)——“掩码即可调用函数”模式的 API 参考。
