# 完整 Transformer——编码器 + 解码器

> 注意力是主角，其他一切——残差、归一化、前馈网络、交叉注意力——都是让它可以深层堆叠的脚手架。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 02（自注意力）、阶段 7 · 03（多头注意力）、阶段 7 · 04（位置编码）
**Time:** 约 75 分钟

## 问题

单个注意力层只是特征提取器，不是完整模型。每层一次矩阵乘法不足以容纳语言能力。你需要深度——而如果没有正确的连接方式，深层网络就会崩溃。

Vaswani 2017 年的论文把六项设计决策组合起来，将单个注意力层变成可堆叠模块。此后的每个 Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器—解码器（T5）——都继承了同一骨架。到 2026 年，这些模块已经过改良（RMSNorm、SwiGLU、预归一化、RoPE），但骨架完全相同。

本课讲解这副骨架。后续课程会把它专门化——第 06 课讲编码器，第 07 课讲解码器，第 08 课讲编码器—解码器。

## 概念

![连接后的编码器与解码器块内部结构](../../../../../../phases/07-transformers-deep-dive/05-full-transformer/assets/full-transformer.svg)

### 六个组成部分

1. **嵌入 + 位置信号。** 词元 → 向量。通过 RoPE（现代方案）或正弦编码（经典方案）注入位置。
2. **自注意力。** 每个位置都关注其他所有位置；在解码器中使用掩码。
3. **前馈网络（FFN）。** 逐位置双层 MLP：`W_2 · activation(W_1 · x)`。默认扩展倍率为 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，超过约 6 层后梯度就会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代方案）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

观察一个向量如何流经一个块：注意力跨位置混合信息，残差将原信息向前传递，FFN 对其变换，归一化则保持信息流稳定。

```figure
transformer-block
```

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的，不使用掩码，所有位置都可以看到其他所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

每个解码器块包含三个子层。中间的交叉注意力，是信息从编码器流向解码器的唯一位置。在纯解码器架构（GPT）中，会省略交叉注意力，只保留带掩码的自注意力 + FFN。

### 预归一化与后归一化

原始论文比较的是 `x + sublayer(LN(x))` 与 `LN(x + sublayer(x))`。后归一化在约 2019 年失宠——如果没有精心设计的预热，很难训练深层模型。预归一化（在子层*之前*执行 `LN`）是 2026 年的默认方案：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年的现代化模块

Vaswani 2017 年使用 LayerNorm + ReLU，现代技术栈则替换了二者。生产模块实际采用：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展倍率 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量相当） |
| 位置 | 绝对正弦编码 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉 LayerNorm 的均值居中步骤（少一次减法），从而节省计算，而且实证表明至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中始终比 ReLU/GELU FFN 的困惑度低约 0.5 点。

### 参数量

对于 `d_model = d`、FFN 扩展倍率为 `r` 的一个模块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略不计

当 `d = 4096, r = 2.6, layers = 32` 时（大致相当于 Llama 3 8B），总量为：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（再加嵌入与输出头），与公开参数量相符。

## 动手构建

### 第 1 步：基础构件

使用第 03 课中的微型 `Matrix` 类（为保持独立性，复制到本文件）：

- `layer_norm(x, eps=1e-5)`——减去均值，再除以标准差。
- `rms_norm(x, eps=1e-6)`——除以 RMS，不减均值。
- `gelu(x)` 与 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 与 `decoder_block(x, enc_out, params)`。

完整连接方式见 `code/main.py`。

### 第 2 步：连接两层编码器和两层解码器

将它们堆叠起来，把编码器输出传入每一个解码器交叉注意力层，并在输出投影前加入最后一个 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第 3 步：在玩具示例上执行前向传播

输入一个包含 6 个词元的源序列与一个包含 5 个词元的目标序列，验证输出形状为 `(5, vocab)`。无需训练——本课关注架构，而不是损失函数。

### 第 4 步：换用 RMSNorm + SwiGLU

用 RMSNorm 与 SwiGLU 替换 LayerNorm 和 ReLU-FFN，确认形状仍然匹配。只需替换函数，就完成了 2026 年的现代化改造。

## 学以致用

PyTorch/TF 的参考实现是 `nn.TransformerEncoderLayer` 与 `nn.TransformerDecoderLayer`。不过，2026 年的大多数生产代码都会自行实现模块，原因包括：

- Flash Attention 在注意力内部调用，而不是通过 `nn.MultiheadAttention`。
- 标准库参考实现不包含 GQA / MLA。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认项。

Hugging Face `transformers` 中有值得阅读的清晰参考模块：`modeling_llama.py` 是 2026 年仅解码器模块的典型实现，约 500 行，值得完整阅读一次。

**编码器、解码器与编码器—解码器应如何选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器—解码器 | T5、BART、Whisper |

仅解码器架构在语言任务上胜出，因为它最容易扩展，同时能处理理解与生成。当输入有明确的“源序列”身份时（翻译、语音识别、结构化任务），编码器—解码器仍然最佳。

## 交付成果

见 `outputs/skill-transformer-block-reviewer.md`。该技能会按照 2026 年默认实践审查新的 Transformer 块实现，并标出缺失项（预归一化、RoPE、RMSNorm、GQA、FFN 扩展倍率）。

## 练习

1. **简单。** 计算 encoder_block 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时的参数量。实现这个模块，再使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从后归一化切换到预归一化。分别初始化两种模型，在随机输入上堆叠 12 层，测量最后的激活范数。后归一化的激活应当爆炸，预归一化则应保持有界。
3. **困难。** 在玩具复制任务（反向复制 `x`）上实现四层编码器—解码器，训练 100 步并报告损失。换用 RMSNorm + SwiGLU + RoPE 后，损失是否下降？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 模块 | “一层 Transformer” | 归一化 + 注意力 + 归一化 + FFN 的堆叠，外面包裹残差连接。 |
| 残差 | “跳跃连接” | 输出 `x + f(x)`，让梯度可以穿过深层网络。 |
| 预归一化 | “先归一化，而不是后归一化” | 现代形式：`x + sublayer(LN(x))`，无须复杂预热即可训练更深网络。 |
| RMSNorm | “不减均值的 LayerNorm” | 除以 RMS；少一次运算，实证稳定性相同。 |
| SwiGLU | “所有 FFN 都换成的激活” | `Swish(W1 x) ⊙ W3 x → W2`，在语言模型困惑度上胜过 ReLU/GELU。 |
| 交叉注意力 | “解码器如何看到编码器” | Q 来自解码器，K/V 来自编码器输出的 MHA。 |
| FFN 扩展倍率 | “中间 MLP 有多宽” | 隐藏大小与 d_model 的比值；LayerNorm 通常为 4，SwiGLU 通常为 2.6。 |
| 无偏置 | “删掉 +b 项” | 现代技术栈在线性层中省略偏置；困惑度略有改善，模型也更小。 |

## 延伸阅读

- [Vaswani 等（2017），Attention Is All You Need](https://arxiv.org/abs/1706.03762)——原始模块规范。
- [Xiong 等（2020），Transformer 架构中的层归一化](https://arxiv.org/abs/2002.04745)——预归一化为何能在深层网络中胜出。
- [Zhang、Sennrich（2019），均方根层归一化](https://arxiv.org/abs/1910.07467)——RMSNorm。
- [Shazeer（2020），GLU 变体改进 Transformer](https://arxiv.org/abs/2002.05202)——SwiGLU 论文。
- [Hugging Face `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)——2026 年仅解码器模块的典型实现。
