# CLIP 与视觉—语言对比预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动此后五年发展的想法：只使用带噪声的 Web 图像—说明文字对和对比损失，把图像编码器与文本编码器对齐到同一个向量空间。不使用任何监督标签，只需要 4 亿个样本对。由此得到的嵌入空间能够执行零样本分类和图文检索，并作为视觉塔接入 2026 年的每种 VLM。SigLIP 2（2025）以 Sigmoid 取代 Softmax，用更低成本扩展到了 CLIP 之上。本课从 InfoNCE 一路推导到成对 Sigmoid 损失，并使用标准库 Python 构建训练步骤。

**Type:** 构建
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** 第 12 阶段 · 第 01 课（ViT 图像块）、第 7 阶段（Transformer）
**Time:** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释 Sigmoid 成对损失（SigLIP）为何无须 Softmax 所要求的全收集开销，就能扩展到 32768 以上的批大小。
- 通过构造文本模板（`a photo of a {class}`），并对余弦相似度取 argmax，执行 ImageNet 零样本分类。
- 说出 CLIP / SigLIP 预训练提供的四个调节杆：批大小、温度、提示词模板与数据质量。

## 问题

CLIP 之前的视觉模型依赖监督学习：收集带标签的数据集（ImageNet：120 万幅图像、1000 个类别），训练 CNN，然后发布。标签成本高昂，会偏向标注人员能够达成共识的内容，而且若不微调，就无法迁移到新任务。

图像—说明文字 Web 数据中免费存在超过十亿个松散标注的样本对。一张金毛寻回犬的照片配上替代文本“我的狗 Max 在公园里”，就携带了监督信号——文字描述了图像。问题是：怎样把这种信号转化为有效训练？

CLIP 的答案是：把图像—说明文字对视为匹配任务。给定一批 N 幅图像和 N 条说明文字，学习让每幅图像匹配自己的说明文字，并与其余 N-1 个干扰项区分开。监督信号只有“这两个对象属于一对，另外 N-1 个不属于”。不需要类别标签，不需要人工标注，只需要对比损失。

得到的嵌入空间能完成远超 CLIP 训练目标的任务。ImageNet 零样本分类之所以有效，是因为“a photo of a cat”的嵌入会靠近从未被显式标注为猫的猫照片。这项赌注催生了 2026 年的每一种 VLM。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，为每幅图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，为每条说明文字输出一个 D 维向量。

两个塔都会把输出归一化为单位长度。由于二者都是单位范数，相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于一批 N 个（图像、说明文字）样本对，构造相似度矩阵 `S`，其形状为 `(N, N)`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中，`tau` 是一个可学习温度参数（CLIP 初始化为 0.07，并在对数空间中学习）。

### InfoNCE 损失

CLIP 对矩阵的行和列分别执行交叉熵，再取对称平均：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵中的 Softmax 会迫使每幅图像与自己的说明文字比同批次中的其他所有说明文字更匹配。“负样本”就是批次中的所有其他项目。批次越大，负样本越多，训练信号越强。CLIP 使用 32k 批大小训练；规模至关重要。

### 温度

`tau` 控制 Softmax 分布的尖锐程度。tau 较低 → 分布尖锐，产生难负样本挖掘效果；tau 较高 → 分布平缓，所有样本都会产生贡献。CLIP 学习 log(1/tau)，并对其截断以防止坍缩。SigLIP 2 固定初始 tau，改为学习偏置。

### Sigmoid 为何更容易扩展（SigLIP）

Softmax 需要同步完整的相似度矩阵。在分布式训练中，你必须把每个嵌入全收集到每个副本，再执行 Softmax，其通信量会随全局规模呈二次方增长。

SigLIP 用逐元素 Sigmoid 替代 Softmax：对每个样本对 `(i, j)` 执行“二者是否匹配”的二元分类。对角线上的类别标签为正，其余均为负。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

当 `y_ij = 1` 时，表示 `i == j`；否则为 0。每个样本对的损失相互独立，不需要全收集。每张 GPU 计算并求和自己的局部块。SigLIP 2 可以低成本扩展到 32k～512k 的批大小，而 CLIP 则需要成比例增加通信量。

### 零样本分类

给定 N 个类别名称，为每个类别构造文本模板：

```
"a photo of a {class}"
```

使用文本编码器嵌入每个模板，使用图像编码器嵌入图像。余弦相似度的 argmax 就是预测类别，无须在目标类别上训练。

提示词模板会影响结果。CLIP 原始论文为每个类别使用了 80 个模板（普通照片、艺术作品、绘画等），再对嵌入取平均，使 ImageNet 分数提高了 3 个百分点。现代应用通常选择一两个模板。

### 线性探针与微调

零样本只是基线。在目标类别上训练一个位于冻结 CLIP 特征之上的线性层，即线性探针；它在领域内任务上会优于零样本。全量微调在领域内又会胜过线性探针，但可能损害零样本迁移能力。三种训练范式对应三种不同取舍。

### SigLIP 2：NaFlex 与稠密特征

SigLIP 2（2025）增加了：

- NaFlex：一个模型可以处理不同宽高比与分辨率。
- 更好的分割和深度估计稠密特征，目标是作为 VLM 中的冻结骨干使用。
- 多语言能力：使用 100 多种语言训练，而 CLIP 只支持英语。
- 10 亿参数规模，而 CLIP 最大为 4 亿参数。

在 2026 年的开放 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。如果纯图文检索场景与特定的 LAION-2B 训练分布相匹配，CLIP 仍是默认选择。

### ALIGN、BASIC、OpenCLIP 与 EVA-CLIP

ALIGN（Google，2021）：思路与 CLIP 相同，使用 18 亿样本对，其中 90% 带有噪声，证明了带噪数据也能随规模扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，提供多种规模，是首选开放检查点。EVA-CLIP：从掩码图像建模开始初始化，是 VLM 的强力骨干。BASIC：Google 的 CLIP + ALIGN 混合方案。它们都属于同一家族，只是数据与调优方式不同。

### 零样本上限

CLIP 类模型在 ImageNet 零样本任务上的上限约为 76%（CLIP-G、OpenCLIP-G）。想进一步提升，就需要大得多的数据（SigLIP 2 达到 80% 以上）或架构变更（监督头、更多参数）。这个基准正趋于饱和；真正的价值在于供下游 VLM 使用的嵌入空间。

```figure
multimodal-fusion
```

## 投入使用

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征、文本字符特征），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 使用纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定）。
3. 用于对比的 Sigmoid 成对损失。
4. 一套零样本分类流程：计算图像与一组文本提示词之间的余弦相似度，再以 argmax 得到预测结果。

运行它并观察损失曲线。绝对数值只是玩具示例，但形态与真实 CLIP 训练器的输出一致。

## 交付成果

本课会产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它会使用 CLIP 模板构建文本提示词，通过指定检查点（例如 `openai/clip-vit-large-patch14`）嵌入两侧内容，并返回带相似度分数的 top-1 / top-5 预测。这项技能会拒绝对不在提示词列表中的类别作出判断。

## 练习

1. 手工为一批 4 个样本对实现 InfoNCE。构造 4x4 相似度矩阵，执行 Softmax，取出对角线，再计算交叉熵。验证 Python 实现与手算结果一致。

2. 除温度外，SigLIP 还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次存在很大的类别不平衡（每行的负样本远多于正样本）时，`b` 发挥什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建猫与狗的零样本分类器。尝试两个提示词模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 幅测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算在 512 张 GPU、批大小 32k 的运行中，Softmax InfoNCE 与 Sigmoid 成对损失的通信成本。哪一种按 O(N) 扩展，哪一种按 O(N^2) 扩展？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等）。根据图表复现其数据缩放结论：模型大小固定时，ImageNet 零样本准确率与训练数据规模之间呈现什么样的对数线性关系？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| InfoNCE | “对比损失” | 对一个批次的相似度矩阵执行交叉熵；每个项目的正样本是其配对项目，其他所有项目都是负样本 |
| Sigmoid 损失 | “SigLIP 损失” | 逐样本对的二元交叉熵；无 Softmax、无全收集，可在分布式训练中低成本扩展 |
| 温度 | “tau” | 在执行 Softmax/Sigmoid 前缩放 Logit 的标量；控制分布的尖锐程度 |
| 零样本 | “无需微调的分类” | 使用文本提示词构造类别嵌入，并按余弦相似度分类；无须在目标类别上训练 |
| 提示词模板 | “a photo of a ...” | 包裹类别名的文本脚手架；会使零样本准确率变化 1～5 个百分点 |
| 双编码器 | “双塔” | 一个图像编码器 + 一个文本编码器；输出位于共享的 D 维空间中 |
| 难负样本 | “难以区分的干扰项” | 与正样本足够相似，迫使模型付出努力才能将其区分开的负样本 |
| 线性探针 | “冻结模型 + 一层” | 只在冻结特征之上训练一个线性分类器；用于衡量特征质量 |
| NaFlex | “原生灵活分辨率” | SigLIP 2 无须调整图像大小，即可接收任意宽高比与分辨率图像的能力 |
| 温度缩放 | “对数参数化 tau” | CLIP 对 `log(1/tau)` 进行参数化，使梯度行为更稳定，并通过截断防止 tau 坍缩到接近零 |

## 延伸阅读

- [Radford 等——Learning Transferable Visual Models From Natural Language Supervision（arXiv:2103.00020）](https://arxiv.org/abs/2103.00020)——CLIP 论文。
- [Zhai 等——Sigmoid Loss for Language Image Pre-Training（arXiv:2303.15343）](https://arxiv.org/abs/2303.15343)——SigLIP。
- [Tschannen 等——SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)——多语言 + NaFlex。
- [Jia 等——ALIGN（arXiv:2102.05918）](https://arxiv.org/abs/2102.05918)——使用带噪 Web 数据扩展规模。
- [Cherti 等——Reproducible scaling laws for contrastive language-image learning（arXiv:2212.07143）](https://arxiv.org/abs/2212.07143)——OpenCLIP 缩放定律。
