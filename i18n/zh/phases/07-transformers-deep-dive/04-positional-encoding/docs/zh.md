# 位置编码——正弦、RoPE 与 ALiBi

> 注意力具有排列不变性。如果没有位置信号，“The cat sat on the mat”和“mat the on sat cat the”会产生相同的输出。三种算法解决了这个问题——它们对“位置”的含义各有不同假设。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 02（自注意力）、阶段 7 · 03（多头注意力）
**Time:** 约 45 分钟

## 问题

缩放点积注意力无法感知顺序。注意力矩阵 `softmax(Q K^T / √d) V` 根据成对相似度计算。打乱 `X` 的各行，输出的各行只会以相同方式重新排列。注意力内部没有任何东西关心位置。

对于词袋模型，这不算缺陷；但对语言、代码、音频、视频等任何顺序承载含义的数据来说，这都是致命问题。

解决办法是以某种方式把位置信息注入嵌入。三个时代给出了不同答案：

1. **绝对正弦编码**（Vaswani，2017）。把位置的 `sin/cos` 值加到嵌入中。简单、无需学习，但很难外推到训练长度以外。
2. **RoPE——旋转位置嵌入**（Su，2021）。按照与位置成比例的角度旋转 Q 和 K 向量，直接在点积中编码*相对*位置。它在 2026 年占据主导。
3. **ALiBi——带线性偏置的注意力**（Press，2022）。完全不使用位置嵌入，而是根据距离为注意力分数加入逐头线性惩罚。长度外推能力出色。

截至 2026 年，几乎每个前沿开放模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已经成为历史方案。

## 概念

![绝对正弦编码、RoPE 旋转与 ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦编码

预先计算固定矩阵 `PE`，其形状为 `(max_len, d_model)`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前计算 `X' = X + PE[:N]`。每个维度都是频率不同的正弦波，模型会学习从相位模式中读取位置。超过 `max_len` 后便会失效：如果模型只见过位置 0～2047，就没有任何信息告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量，而不是嵌入。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为 `pos_k` 的键应用相同旋转。点积 `q'_m · k'_n` 会变成只与 `(m - n)` 有关的函数。也就是说：**即使旋转由绝对位置决定，注意力分数也只依赖相对距离。** 这是一个漂亮的技巧。

扩展 RoPE 时，可以缩放 `base`（NTK-aware、YaRN、LongRoPE），从而无需重新训练即可外推到更长上下文。Llama 3 正是通过这种方式从 8K 扩展到 128K 上下文。

### ALiBi

跳过嵌入技巧，直接为注意力分数增加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中，`m_h` 是每个头特有的斜率（例如 `1 / 2^(8·h/H)`）。较近词元得到提升，较远词元受到惩罚。不增加训练成本。论文显示，它的长度外推能力胜过正弦编码，在原始训练长度上的效果则与 RoPE 相当。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 使用者 |
|---------|---------------|---------------|---------|
| 绝对正弦编码 | 差 | 无 | 原始 Transformer、早期 BERT |
| 学习式绝对编码 | 无 | 极低 | GPT-2、GPT-3 |
| RoPE | 配合缩放时良好 | 无 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 出色 | 需要微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 出色 | 无 | BLOOM、MPT、Baichuan |

RoPE 之所以胜出，是因为它可以直接接入注意力而无须改变架构、能够编码相对位置，并通过 `base` 超参数为长上下文微调提供清晰的调节旋钮。

```figure
rope-explorer
```

## 动手构建

### 第 1 步：正弦编码

见 `code/main.py`。只需四行核心计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前，把它加到嵌入矩阵中。

### 第 2 步：把 RoPE 应用于 Q、K

RoPE 会原地作用于 Q 和 K。对每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对位置 `m` 的 Q 与位置 `n` 的 K 应用相同函数。它们的点积会在每一对坐标上获得一个 `cos((m-n)·θ_i)` 因子。注意力无需额外成本即可学到相对位置。

### 第 3 步：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到形状为 `(seq_len, seq_len)` 的第 `h` 个头的注意力分数矩阵上，再执行 softmax。

### 第 4 步：验证 RoPE 的相对距离性质

任选两个随机向量 `a, b`，先分别按 `(pos_a, pos_b)` 旋转，再按 `(pos_a + k, pos_b + k)` 旋转。两次点积应当在浮点误差范围内相等。这正是 RoPE 的核心性质——它不受绝对偏移影响，只与相对间隔有关。

## 学以致用

PyTorch 2.5+ 在 `torch.nn.functional` 中提供 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，在注意力内核中应用 RoPE。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 从 4K 扩展到 16K 以上时，把 `base` 缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更聪明的插值方法，能在长上下文上保持注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 在 2024 年提出的方法，通过进化搜索选择逐维缩放因子。Phi-3-Long 使用它。
- **位置插值 + 微调。** 直接按扩展因子缩小位置，再微调 10 亿～50 亿词元，效果出人意料地好。

## 交付成果

见 `outputs/skill-positional-encoding-picker.md`。该技能会根据目标上下文长度、外推需求和训练预算，为新模型选择编码策略。

## 练习

1. **简单。** 绘制正弦 `PE` 矩阵的热力图，其中 `max_len=512, d=128`。确认“维度索引越大，条纹越宽”的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练微型语言模型，再分别在使用和不使用缩放的情况下测试长度 1024，并测量困惑度。
3. **困难。** 在同一个注意力模块中实现 ALiBi 与 RoPE。在长度为 512 的序列上训练四层 Transformer 完成复制任务，再在测试时外推到 2048，并比较性能退化。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码 | “告诉注意力顺序” | 添加到嵌入或注意力中的任意位置信号。 |
| 正弦编码 | “最初那个方案” | 把按几何级数分布频率的 `sin/cos` 加入嵌入；无法良好外推。 |
| RoPE | “旋转嵌入” | 按位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | “线性偏置技巧” | 向注意力分数加入 `-m·\|i-j\|`；无须嵌入，外推能力出色。 |
| base | “RoPE 的旋钮” | RoPE 中的频率缩放因子；增大它可在推理时扩展上下文。 |
| NTK-aware | “一种 RoPE 缩放技巧” | 缩放 `base`，避免上下文扩大时高频维度受到挤压。 |
| YaRN | “复杂的那一种” | 逐维组合插值与外推，并保持注意力熵。 |
| 外推 | “超过训练长度仍有效” | 位置方案能否在超过训练所见 `max_len` 后仍输出正确结果？ |

## 延伸阅读

- [Vaswani 等（2017），Attention Is All You Need 第 3.5 节](https://arxiv.org/abs/1706.03762)——最初的正弦编码。
- [Su 等（2021），RoFormer：使用旋转位置嵌入增强 Transformer](https://arxiv.org/abs/2104.09864)——RoPE 论文。
- [Press、Smith、Lewis（2021），短序列训练、长序列测试：带线性偏置的注意力实现输入长度外推](https://arxiv.org/abs/2108.12409)——ALiBi。
- [Peng 等（2023），YaRN：高效扩展大语言模型上下文窗口](https://arxiv.org/abs/2309.00071)——顶尖 RoPE 缩放方法。
- [Chen 等（2023），通过位置插值扩展大语言模型上下文窗口](https://arxiv.org/abs/2306.15595)——Meta 的 Llama 2 长上下文论文。
- [Ding 等（2024），LongRoPE：把大语言模型上下文窗口扩展到 200 万词元以上](https://arxiv.org/abs/2402.13753)——Phi-3-Long 使用且在“学以致用”部分提到的 Microsoft 方法。
- [Hugging Face Transformers——`modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)——所有 RoPE 缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）的生产级实现。
