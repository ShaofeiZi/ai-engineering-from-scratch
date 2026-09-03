# Transfusion：在一个 Transformer 中融合自回归文本与扩散图像

> Chameleon 与 Emu3 把全部赌注押在离散词元上。这种方法确实有效，但量化瓶颈清晰可见——图像质量在达到连续空间扩散模型的水平之前就已停滞。Transfusion（Meta，Zhou 等，2024 年 8 月）做出了相反选择：保留连续图像，彻底移除 VQ-VAE，并使用两个损失训练同一个 Transformer。文本词元使用下一词元预测，图像块使用流匹配/扩散损失，两项目标共同优化同一套权重。Stable Diffusion 3 背后的架构 MMDiT 与它非常相似。本课会解读 Transfusion 的核心观点，构建一个玩具式双损失训练器，并追踪让单个 Transformer 同时完成两项工作的注意力掩码。

**Type:** 构建
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** 第 12 阶段 · 第 11 课（Chameleon）、第 8 阶段（生成式 AI）
**Time:** 约 180 分钟

## 学习目标

- 在一个骨干网络上连接同时运行两项损失的 Transformer（文本词元使用 NTP，图像块使用扩散 MSE）。
- 解释为何图像块之间采用双向注意力、文本词元之间采用因果注意力，是正确的掩码选择。
- 从计算量、质量与代码复杂度三个方面比较 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 说出 MMDiT 的贡献：每个块使用模态专用权重，在残差流中执行联合注意力。

## 问题

关于离散与连续图像词元的争论，比大语言模型本身还要古老。连续表示（原始像素、VAE 潜变量）能够保留细节；离散词元（VQ 索引）符合 Transformer 原生词表，却会在量化步骤中损失细节。

Chameleon / Emu3 选择离散路线：一个损失、一种架构，但图像保真度受分词器质量限制。

扩散模型选择连续路线：图像质量卓越，但模型与大语言模型相互独立，需要复杂的噪声调度工程，也无法与文本生成自然融合。

Transfusion 提出：能否兼得二者？保留连续图像，仍只训练一个模型，把两个损失拼接到同一个梯度步骤中。

## 概念

### 双损失架构

一个仅解码器 Transformer 处理包含以下内容的序列：

- 文本词元（离散，来自 BPE 词表）。
- 图像块（连续，由 16x16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- 标记连续图像块所在位置的 `<image>` 与 `</image>` 标签。

前向传播只运行一次。损失会针对不同词元选择两个头之一：

- 文本词元：在词表 Logit 头上使用标准交叉熵。
- 图像块：在连续图像块上使用扩散损失——预测加入每个图像块的噪声。

梯度流经共享的 Transformer 主干，两项损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本词元必须使用因果注意力——不能让文本词元关注未来文本，否则教师强制会失效。然而，图像块表示的是同一个快照，应当在同一图像块区域内彼此双向关注。

掩码如下：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

训练和推理时，它都实现为分块三角掩码。

### Transformer 内部的扩散损失

这里使用标准扩散损失：向图像块加入噪声，要求模型预测所加噪声（或者等价地预测干净图像块）。Transfusion 使用流匹配版本——预测从噪声指向干净数据的速度场。

训练期间：
1. 对每个图像块 x0，随机采样一个时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配中的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本 NTP 损失一起反向传播。

推理期间的生成方式是：
- 文本词元：标准自回归采样。
- 图像块：在先前文本词元的条件下运行扩散采样循环（通常为 10～30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024 年 3 月）与 Transfusion 几乎同期推出了 MMDiT（Multimodal Diffusion Transformer）。两种架构是近亲。

MMDiT 的主要差异：

- 每个块使用模态专用权重。每个 Transformer 块为文本词元与图像块分别设置 Q、K、V 和 MLP 权重。注意力联合执行（跨模态），其余部分则按模态独立。
- 整流流训练。一种特定的流匹配变体，采样方式明确，数学形式比 DDPM 简单。
- 规模。MMDiT 是 SD3（2B 与 8B 参数变体）的骨干，Transfusion 论文则扩展到 7B。

二者汇聚到同一个核心思路：一个 Transformer 对文本执行 NTP，对连续图像表示执行扩散。

### 为什么它胜过 Chameleon 风格

连续扩散与离散 NTP 在图像生成上的质量差距可以测量。Transfusion 论文报告：

- 在 7B 参数下，FID 比同规模 Chameleon 风格模型好 3～5 分。
- 无须训练分词器——图像编码器更简单（线性投影到隐藏维度，与 ViT 输入层相同）。
- 图像块去噪可以并行执行，而自回归图像词元不能。

缺点是：Transfusion 是双损失模型，训练动态更难控制，需要调节损失权重。NTP 与扩散的调度不匹配，可能导致一个头占据主导地位。

### 后续发展

Janus-Pro（第 12.15 课）进一步改进 Transfusion 的思路，把理解与生成所用的视觉编码器解耦——一边使用 SigLIP，另一边使用 VQ，同时共享 Transformer 主干。Show-o（第 12.14 课）则用离散扩散（掩码预测）替代连续扩散。统一生成模型家族在 Transfusion 之后迅速分化。

2026 年能够输出图像的生产级 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定采用了这一家族的某种后继方案，具体细节并未公开。

```figure
cfg-guidance-scale
```

## 投入使用

`code/main.py` 在一个微型、类似 MNIST 的问题上构建玩具 Transfusion：

- 文本说明是描述数字（0～9）的短整数序列。
- 图像是由字节组成的 4x4 网格。
- 一对共享权重的线性投影充当 Transformer 替代品；文本使用 NTP 损失，带噪图像块使用 MSE 损失。
- 训练循环交替使用两项损失，并显式构造注意力掩码。
- 生成阶段在一次前向传播中同时产生文本说明与 4x4 图像。

Transformer 本身只是玩具模型，真正的交付物是双损失管道、注意力掩码构造方式与推理循环。

## 交付成果

本课会产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它会设计双损失调度（损失权重、掩码形状、共享块与模态专用块），并标记实现风险。

## 练习

1. 一个 Transfusion 风格模型的训练数据包含 70% 文本词元和 30% 图像块，而图像扩散损失的数值约为文本 NTP 损失的 10 倍。应设置怎样的损失权重来平衡二者？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现分块三角掩码，把每个条目标为 0 或 1。

3. MMDiT 使用模态专用 QKV 权重。相比 Transfusion 完全共享的 Transformer，这会增加多少参数？对于 7B 模型，是否值得？

4. 生成时，给定文本提示词，模型先运行 NTP 生成 50 个词元，随后遇到 `<image>`，再对 256 个图像块执行 20 步去噪。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流，以及它为什么能用比 DDPM 更少的推理步骤收敛。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| 双损失训练 | “NTP + 扩散” | 单个 Transformer 在同一个梯度步骤中，同时优化文本词元上的交叉熵和连续图像块上的 MSE |
| 流匹配 | “整流流” | 预测从噪声指向干净数据的速度场的扩散变体；数学形式比 DDPM 简单 |
| MMDiT | “多模态 DiT” | Stable Diffusion 3 的架构：联合注意力，以及模态专用的 MLP 与归一化层 |
| 分块三角掩码 | “因果文本 + 双向图像” | 文本之间使用因果关系、图像区域内部使用双向关系的注意力掩码 |
| 连续图像表示 | “无 VQ” | 图像块表示为实值向量，而不是整数码本索引 |
| 速度预测 | “v 参数化” | 网络输出噪声与数据之间的速度场，而不是噪声本身 |

## 延伸阅读

- [Zhou 等——Transfusion（arXiv:2408.11039）](https://arxiv.org/abs/2408.11039)
- [Esser 等——Stable Diffusion 3 / MMDiT（arXiv:2403.03206）](https://arxiv.org/abs/2403.03206)
- [Peebles 与 Xie——DiT（arXiv:2212.09748）](https://arxiv.org/abs/2212.09748)
- [Zhao 等——MonoFormer（arXiv:2409.16280）](https://arxiv.org/abs/2409.16280)
- [Xie 等——Show-o（arXiv:2408.12528）](https://arxiv.org/abs/2408.12528)
