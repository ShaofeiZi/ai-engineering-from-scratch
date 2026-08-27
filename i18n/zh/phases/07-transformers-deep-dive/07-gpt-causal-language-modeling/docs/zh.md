# GPT——因果语言建模

> BERT 能看见两侧，GPT 只能看见过去。这个三角掩码是现代 AI 中影响最深远的一行代码。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 02（自注意力）、阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）
**Time:** 约 75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个词元，第 `t` 个词元上的概率分布是什么？使用这一信号——下一词元预测——进行训练，就能得到一个每次生成一个词元、可以生成任意文本的模型。

要在完整序列上并行地端到端训练它，必须让每个位置的预测只能依赖更早的位置。否则，模型只需偷看答案就能作弊。

因果掩码正是为此而生。它是一个由 `-inf` 值组成的上三角矩阵，在 softmax 前加入注意力分数。执行 softmax 后，这些位置变为 0。每个位置只能关注自身与更早的位置。由于只需对完整序列应用一次，就能在一次前向传播中并行获得 N 个下一词元预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2025）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是采用同一核心循环的仅解码器因果 Transformer。真正区分它们的是数据质量、规模、架构改进和后训练（SFT、RLHF、DPO 及其后继方法）。

## 概念

![因果掩码形成三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 前把 `M` 加到原始注意力分数上。`exp(-inf) = 0`，因此被遮盖的位置贡献零权重。注意力矩阵的每一行，都只是在先前位置上的概率分布。

实现成本：调用一次 `torch.tril()`。计算时间：纳秒级。对整个领域的影响：无处不在。

### 三角形从何而来

因果掩码通常被描述为附加在注意力上的补丁。如果反过来推导，就不再神秘：注意力是前缀平均的第三次改良，而三角形就是该平均操作的循环边界写成矩阵后的形状。

**阶段 1——前缀平均。** 对序列进行最简单的因果摘要：位置 `i` 变成位置 `0…i` 的均值。写成循环就是 `out[i] = X[:i+1].mean(0)`。同一计算也可以通过一次矩阵乘法完成：取一个由 1 组成的下三角矩阵，用每行元素个数进行归一化，再相乘：

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

第 `i` 行的 `A` 是 `[1/(i+1), …, 1/(i+1), 0, …, 0]`。对角线上方的零值体现了因果性。未来并不是被遮掉了，而是从未进入求和范围。

**阶段 2——学习式权重。** 均匀平均把过去的每个词元视为同等相关。把这些 1 替换为学习得到的分数矩阵 `S`。此时各行不再天然和为一，因此不再除以计数，而是逐行用 softmax 归一化。softmax 永远不会输出精确的零，这会破坏因果性——除非把未来位置的分数设为 `-inf`，因为 `exp(-inf) = 0`：

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

同一个三角形，同一个行随机矩阵，同一次矩阵乘法。`-inf` 掩码并不是注意力新加的机制，而是第 1 阶段中的零值被转换到了 softmax 的输入域。

**阶段 3——依赖内容的权重。** 在第 2 阶段中，`S` 训练后就是固定的：无论词元内容是什么，位置 7 对位置 3 的权重都相同。让分数依赖词元本身：`S = Q @ K.T / sqrt(d_k)`。其他部分完全不变：掩码、softmax、矩阵乘法都相同。

三个阶段共享一个不变量：由一个下三角行随机矩阵乘以序列。从均匀平均，到学习式静态权重，再到依赖内容的权重。掩码从未被“添加”到注意力中，它只是从最初的平均操作中保留下来。

```figure
mask-derivation
```

### 并行训练，串行推理

训练时：一次前向传播处理完整的 `(N, d_model)` 序列，计算 N 个交叉熵损失（每个位置一个），求和后反向传播。序列维度可以并行。这就是 GPT 训练能够扩展的原因——一次 GPU 前向传播即可处理批次中的 100 万个词元。

推理时：逐个生成词元。输入 `[t1, t2, t3]`，得到 `t4`；输入 `[t1, t2, t3, t4]`，得到 `t5`；输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）会保存 `t1…tn` 的隐藏状态，避免每一步重新计算。但推理时的串行深度仍等于输出长度。这就是自回归税，也是每个大语言模型解码过程的延迟瓶颈。

### 损失——错位一位

给定词元 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i` 计算 `-log P(target_i | inputs[:i+1])`，再求和。这就是整个序列的交叉熵。

你听说过的每个 Transformer 语言模型都使用这种损失训练。预训练、微调、SFT——损失相同，数据不同。

### 解码策略

训练完成后，采样方式的重要性远超许多人的想象。

| 方法 | 做法 | 适用场景 |
|--------|--------------|-------------|
| 贪心 | 每一步取 argmax | 确定性任务、代码补全 |
| 温度 | 用 T 除 logits 后采样 | 创意任务；T 越高，多样性越大 |
| Top-k | 只从概率最高的 k 个词元中采样 | 截断低概率长尾 |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年后的默认方案；适应分布形状 |
| Min-p | 保留满足 `p > min_p * max_p` 的词元 | 2024 年以后；比 top-p 更善于排除长尾 |
| 推测解码 | 草稿模型提出 N 个词元，大模型并行验证 | 保持质量不变，将延迟降低 2～3 倍 |

到 2026 年，min-p + 温度 0.7 是开放权重模型的合理默认值。推测解码则是任何生产推理技术栈的基本配置。

### “GPT 配方”为何有效

1. **仅解码器。** 没有编码器开销，每层只执行一次注意力 + FFN。
2. **规模化。** 124M → 1.5B → 175B → 数万亿。Chinchilla 缩放定律（第 13 课）告诉你如何分配计算资源。
3. **上下文学习。** 大约在 6B～13B 出现。模型无须微调即可遵循少样本示例。
4. **RLHF。** 基于人类偏好进行后训练，把原始预训练文本模型转变为聊天助手。
5. **预归一化 + RoPE + SwiGLU。** 支持稳定的大规模训练。

核心架构自 GPT-2 以来没有太大变化。真正有趣的进展都发生在数据、规模和后训练上。

```figure
causal-mask
```

## 动手构建

### 第 1 步：因果掩码

见 `code/main.py`。只需一行：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 前把它加到注意力分数中，这就是全部机制。

### 第 2 步：一个两层、类似 GPT 的模型

堆叠两个解码器块（带掩码的自注意力 + FFN，不含交叉注意力）。增加词元嵌入、位置编码和反嵌入层（与词元嵌入矩阵共享权重——GPT-2 起沿用的标准技巧）。

### 第 3 步：端到端下一词元预测

在包含 20 个词元的玩具词表上，为每个位置生成 logits，并对错位一位后的目标计算交叉熵损失。不计算梯度——这里只做前向传播健全性检查。

### 第 4 步：采样

实现贪心、温度、top-k、top-p、min-p。对固定提示分别运行并比较输出。采样函数只需 10 行代码。

## 学以致用

PyTorch 在 2026 年的常用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 会执行前向传播，取出最后位置的 logits，采样下一个词元，将其追加后重复。每套生产级大语言模型推理技术栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都用大量优化实现了同一个循环——批量预填充、连续批处理、KV 缓存分页、推测解码。

**用一句话区分 GPT 与 BERT：** GPT 预测 `P(x_t | x_{<t})`，BERT 预测 `P(x_masked | x_unmasked)`。损失函数决定模型能否生成文本。

## 交付成果

见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并指出何时必须使用确定性解码。

## 练习

1. **简单。** 运行 `code/main.py`，验证执行 softmax 后的因果注意力矩阵为下三角。抽查第 3 行，它应该只在第 0～3 列有权重。
2. **中等。** 实现束宽为 4 的束搜索。在 10 个短提示上比较 beam-4 与贪心解码的困惑度。束搜索总能胜出吗？（提示：翻译通常如此，开放式聊天则不一定。）
3. **困难。** 实现推测解码：使用一个两层小模型作为草稿模型，使用一个六层模型作为验证器。对 100 次长度为 64 的补全测量墙钟加速比，并确认输出与验证器的贪心结果一致。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 因果掩码 | “那个三角形” | 加入注意力分数的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| 下一词元预测 | “那个损失” | 每个位置上，模型分布与真实下一词元之间的交叉熵。 |
| 自回归 | “一次生成一个” | 把输出反馈为输入；只能在训练时并行，生成时不能。 |
| Logits | “softmax 前的分数” | 语言模型头在 softmax 前的原始输出；采样在这些值上进行。 |
| 温度 | “创造力旋钮” | 用 T 除 logits；T→0 等于贪心，T→∞ 趋近均匀分布。 |
| Top-p | “核采样” | 把分布截断为累积概率 ≥ p 的最小集合。 |
| Min-p | “比 top-p 更好” | 保留满足 `p ≥ min_p × max_p` 的词元；阈值会适应分布的尖锐程度。 |
| 推测解码 | “草拟 + 验证” | 便宜模型提出 N 个词元，大模型并行验证。 |
| 教师强制 | “训练技巧” | 训练时输入真实的上一个词元，而不是模型预测；每个序列到序列语言模型都会使用。 |

## 延伸阅读

- [Radford 等（2018），通过生成式预训练改进语言理解](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)——GPT-1。
- [Radford 等（2019），语言模型是无监督多任务学习器](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)——GPT-2。
- [Brown 等（2020），语言模型是少样本学习器](https://arxiv.org/abs/2005.14165)——GPT-3 与上下文学习。
- [Leviathan、Kalman、Matias（2023），通过推测解码实现 Transformer 快速推理](https://arxiv.org/abs/2211.17192)——推测解码论文。
- [Hugging Face `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)——经典因果语言模型参考代码。
