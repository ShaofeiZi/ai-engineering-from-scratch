# 从零构建 Transformer——综合项目

> 十三节课，一个模型，不走捷径。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 第 01～13 课。不要跳过。
**Time:** 约 120 分钟

## 问题

你已经读过每篇论文，实现过注意力、多头拆分、位置编码、编码器与解码器块、BERT 和 GPT 损失、MoE、KV 缓存。现在要让它们共同完成一项真实任务。

这个综合项目要求在字符级语言建模任务上，端到端训练一个小型仅解码器 Transformer。它阅读莎士比亚，再生成新的莎士比亚风格文本；规模小到可以在笔记本电脑上于 10 分钟内训练，又正确到只要换成更大的数据集并延长训练，就能得到真正的语言模型。

这是本课程的“nanoGPT”。它并非原创——Karpathy 2023 年的 nanoGPT 教程是每个学生都至少会实现一次的参考版本。我们沿用其整体形态，再根据此前学过的内容重新设计。

## 概念

![从零构建 Transformer 的模块图](../assets/capstone.svg)

带注释的架构如下：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们将交付什么

- `GPTConfig`——集中配置所有超参数。
- `MultiHeadAttention`——支持因果掩码和批处理，并可选择 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN`——现代 FFN。
- `Block`——采用预归一化，用残差连接包裹注意力与 FFN。
- `GPT`——嵌入、堆叠模块、语言模型头、generate()。
- 使用 AdamW、余弦学习率和梯度裁剪的训练循环。
- 针对莎士比亚文本的字符级分词器。

### 我们不会交付什么

- RoPE——第 04 课已经从概念上实现。为保持简洁，这里使用学习式位置嵌入；练习会要求你换成 RoPE。
- 生成期间的 KV 缓存——每个生成步骤都会在完整前缀上重新计算注意力。速度较慢，但实现简单；练习会要求你增加 KV 缓存。
- Flash Attention——只要输入满足条件，PyTorch 2.0+ 就会自动分派；我们使用 `F.scaled_dot_product_attention`。
- MoE——每个模块只有一个 FFN。第 11 课已经介绍 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，让一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2000 步：

- 训练损失会在约 6 分钟内从约 4.2（随机）收敛到约 1.5。
- 采样输出看起来像莎士比亚：古英语词语、换行、“ROMEO:”等专有名称会涌现出来。
- 验证损失（文本最后 10% 的留出部分）会紧跟训练损失；在这一规模与预算下不会过拟合。

```figure
n5-block-stack
```

## 动手构建

本课使用 PyTorch。安装 `torch` 即可（CPU 版本也能运行）。参见 `code/main.py`。脚本会负责：

- 在缺少 `tinyshakespeare.txt` 时下载它（或读取本地副本）。
- 字节级字符分词器。
- 按 90/10 划分训练集与验证集。
- 在支持的硬件上使用 bf16 自动混合精度的训练循环。
- 训练完成后采样。

### 第 1 步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

共有 65 个不同字符，词表很小，可以装进 4 字节的 vocab_size。没有 BPE，也没有分词器方面的麻烦。

### 第 2 步：模型

见 `code/main.py`。模块就是第 05 课中的教科书式设计——预归一化、RMSNorm、SwiGLU、因果 MHA。采用 4/4/128 配置时，参数量约为 80 万。

### 第 3 步：训练循环

随机抽取长度为 256 的词元窗口作为批次，执行前向传播，计算错位一位的交叉熵，再反向传播，执行 AdamW 步骤，记录日志，然后重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第 4 步：采样

给定提示后，重复执行前向传播，从 top-p logits 中采样，追加结果并继续，直到生成 500 个词元。

### 第 5 步：阅读输出

训练 2000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

这不是莎士比亚原作，却已经具有莎士比亚的形态。对于约 80 万参数、在笔记本电脑上训练 6 分钟的模型而言，这已经是明确的成功。

## 学以致用

这个综合项目是一份参考架构。要把它扩展成真正可用的模型，可以进行三项改造：

1. **替换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表会从 65 扩大到约 5 万，模型容量也要相应扩大。
2. **使用更大的语料库训练。** 使用 `OpenWebText` 或 `fineweb-edu`（Hugging Face）。在单张 A100 上，125M 参数的 GPT 使用 10B 词元训练约需 24 小时。
3. **加入 RoPE + KV 缓存 + Flash Attention。** 以下练习会引导你逐项完成。

最终会得到一个能够生成流畅英语的 125M 参数 GPT。它不是前沿模型，但 2026 年 Karpathy、EleutherAI 和 Allen Institute 训练研究检查点时使用的代码路径与此相同，只是规模更大。

## 交付成果

见 `outputs/skill-transformer-review.md`。该技能会根据此前 13 节课的知识，全面审查一个从零实现的 Transformer 是否正确。

## 练习

1. **简单。** 运行 `code/main.py`，验证训练完成时的验证损失低于 2.0。把 `max_steps` 从 2000 改为 5000——验证损失是否继续下降？
2. **中等。** 用 RoPE 替换学习式位置嵌入。在 `MultiHeadAttention` 内部对 Q 与 K 应用旋转，训练并确认验证损失至少同样低。
3. **中等。** 在采样循环中实现 KV 缓存。分别在使用和不使用缓存时生成 500 个词元，笔记本电脑上的墙钟速度应提高 5～20 倍。
4. **困难。** 为模型增加第二个输出头，用于预测下下个词元（MTP——DeepSeek-V3 的多词元预测），并联合训练。它有帮助吗？
5. **困难。** 把每个块中的单一 FFN 替换为包含 4 个专家的 MoE，使用路由器 + top-2 路由。在激活参数量相同时，观察验证损失如何变化。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | “Karpathy 的教程仓库” | 最小化的仅解码器 Transformer 训练代码，约 300 行；经典参考。 |
| tinyshakespeare | “标准玩具语料库” | 约 1.1 MB 文本；自 2015 年以来，每个字符语言模型教程都在使用。 |
| 权重绑定嵌入 | “共享输入/输出矩阵” | 语言模型头权重 = 词元嵌入矩阵的转置；节省参数并提高质量。 |
| bf16 自动混合精度 | “训练精度技巧” | 前向/反向传播使用 bf16，优化器状态保持 fp32；2021 年后的标准做法。 |
| 梯度裁剪 | “阻止尖峰” | 把全局梯度范数限制为 1.0，防止训练爆炸。 |
| 余弦学习率计划 | “2020 年后的默认方案” | 学习率先线性升高（预热），再按余弦曲线衰减至峰值的 10%。 |
| MFU | “模型 FLOP 利用率” | 实际 FLOPs / 理论峰值；2026 年稠密模型达到 40%、MoE 达到 30% 就很优秀。 |
| 验证损失 | “留出损失” | 模型从未见过的数据上的交叉熵，用于检测过拟合。 |

## 延伸阅读

- [The Annotated Transformer（Harvard NLP）](https://nlp.seas.harvard.edu/annotated-transformer/)——经典的带注释实现。
