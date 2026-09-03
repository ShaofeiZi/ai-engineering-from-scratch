# 视觉 Transformer（ViT）

> 图像是图块组成的网格，句子是词元组成的网格。同一个 Transformer 可以处理二者。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 05（完整 Transformer）、阶段 4 · 03（CNN）、阶段 4 · 14（视觉 Transformer 入门）
**Time:** 约 45 分钟

## 问题

2020 年以前，计算机视觉意味着卷积。在 ImageNet、COCO 和各种检测基准上，每个顶尖模型都使用 CNN 骨干网络；Transformer 只用于语言。

Dosovitskiy 等人在 2020 年的论文《An Image is Worth 16x16 Words》中证明，可以彻底舍弃卷积。把图像切成大小固定的图块，以线性方式把每个图块投影为嵌入，再把这个序列送进普通 Transformer 编码器。当规模足够大时（在 ImageNet-21k 或更大数据集上预训练），ViT 可以达到或超过基于 ResNet 的模型。

ViT 开启了 2026 年的一种更广泛模式：同一种架构处理多种模态。Whisper 把音频变成词元，ViT 把图像变成词元，机器人使用动作词元，视频使用像素词元。Transformer 并不在乎——给它一个序列，它就能学习。

到 2026 年，ViT 及其后继模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大多数视觉任务。CNN 仍然在边缘设备和延迟敏感任务上胜出，除此之外的技术栈中几乎都有某种 ViT。

## 概念

![图像 → 图块 → 词元 → Transformer](../../../../../../phases/07-transformers-deep-dive/09-vision-transformers/assets/vit.svg)

### 第 1 步——划分图块

把一张 `H × W × C` 图像切成 `N × (P·P·C)` 的扁平图块序列。典型设置：`224 × 224` 图像、`16 × 16` 图块 → 196 个图块，每个包含 768 个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

图块大小是关键调节杆。图块越小，词元越多、分辨率越高，但注意力成本会按平方增长；图块越大，粒度越粗、成本越低。

### 第 2 步——线性嵌入

使用一个学习式矩阵把每个扁平图块投影到 `d_model`。这等价于卷积核大小为 `P`、步幅为 `P` 的卷积。在 PyTorch 中，它实际上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行即可实现。

### 第 3 步——前置 `[CLS]` 词元并加入位置嵌入

- 在序列前添加一个可学习的 `[CLS]` 词元，它最终的隐藏状态会作为图像表示用于分类。
- 加入可学习的位置嵌入（原始 ViT），或后续变体中的二维正弦编码。
- 从 2024 年开始，RoPE 被扩展到二维位置，有时不再需要显式嵌入。

### 第 4 步——标准 Transformer 编码器

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 块，与 BERT 完全相同，不包含任何视觉专用层。这正是论文在教学意义上的核心结论。

### 第 5 步——输出头

用于分类时：取 `[CLS]` 隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，则丢弃 `[CLS]`，直接使用图块嵌入。

### 产生重要影响的变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原始版本。固定图块大小、完全全局注意力。 |
| DeiT | 2021 | 知识蒸馏；只用 ImageNet-1k 即可训练。 |
| Swin | 2021 | 使用移位窗口的层次化结构，成本稳定地低于二次方。 |
| DINOv2 | 2023 | 自监督（无标签），最佳通用视觉特征。 |
| ViT-22B | 2023 | 220 亿参数；同样遵循缩放定律。 |
| SigLIP | 2023 | ViT + 语言对，使用 sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 它为何花了一段时间才流行

ViT 没有 CNN 的归纳偏置（平移不变性、局部性），因此需要*大量*数据才能追平 CNN。如果没有超过 1 亿张带标签图像或强大的自监督预训练，在计算量相同的情况下 CNN 仍然胜出。DeiT 在 2021 年用蒸馏技巧缓解了问题，DINOv2 则在 2023 年通过自监督彻底解决了它。

```figure
n5-patch-stream
```

## 动手构建

见 `code/main.py`。其中使用纯标准库完成图块划分、线性嵌入与健全性检查，不进行训练——任何实际规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 第 1 步：模拟图像

用行列表中的 `(R, G, B)` 元组表示一张 24 × 24 RGB 图像。我们使用 6×6 图块 → 16 个图块，每个图块形成 108 维嵌入向量。

### 第 2 步：划分图块

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

栅格顺序：按行优先遍历网格。每个 ViT 都采用这一顺序。

### 第 3 步：线性嵌入

将每个扁平图块乘以随机的 `(patch_flat_size, d_model)` 矩阵。验证前置词元后的输出形状为 `(N_patches + 1, d_model)`，其中包含 `[CLS]`。

### 第 4 步：计算实际 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 个头、d=768、patch=16。与 ResNet-50（约 2500 万）比较。ViT-Base 约为 8600 万，ViT-Large 约为 3.07 亿，ViT-Huge 约为 6.32 亿。

## 学以致用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干网络，只训练一个小型输出头，即可用于分类、检索、检测和图像描述。在所有不涉及文本的视觉任务上，Meta 的 DINOv2 检查点都胜过 CLIP。

**图块大小选择。** 小模型使用 16×16（ViT-B/16），密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2），超大模型使用 14×14。

## 交付成果

见 `outputs/skill-vit-configurator.md`。该技能会根据数据集大小、分辨率与计算预算，为新的视觉任务选择 ViT 变体和图块大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证图块数量等于 `(H/P) * (W/P)`，扁平图块维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入——分别为每个图块的 `row` 和 `col` 生成独立正弦编码，再拼接。把它们送入微型 PyTorch ViT，并在 CIFAR-10 上与可学习位置嵌入比较准确率。
3. **困难。** 构建三层 ViT（PyTorch），使用 4×4 图块在 1000 张 MNIST 图像上训练，并测量测试准确率。随后在同样的 1000 张图像上增加 DINOv2 预训练（简化方式：只训练编码器根据被遮盖图块预测图块嵌入）。准确率提高了吗？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 图块 | “视觉 Transformer 的词元” | 图像中 `P × P × C` 区域的扁平像素向量。 |
| 图块化 | “切分 + 展平” | 把图像切成不重叠的图块，再将每块展平为向量。 |
| `[CLS]` 词元 | “图像摘要” | 添加到序列前的可学习词元；其最终嵌入就是图像表示。 |
| 归纳偏置 | “模型作出的假设” | ViT 的先验少于 CNN，因此需要更多数据来弥补差距。 |
| DINOv2 | “自监督 ViT” | 使用图像增强 + 动量教师、无标签训练；2026 年最佳通用图像特征。 |
| SigLIP | “CLIP 的后继者” | 使用 sigmoid 对比损失训练的 ViT + 文本编码器；相同计算量下优于 CLIP。 |
| Swin | “窗口化 ViT” | 采用局部注意力 + 移位窗口的层次化 ViT，复杂度低于二次方。 |
| 寄存器词元 | “2023 年技巧” | 少量额外的可学习词元，用于吸收注意力汇点；可改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy 等（2020），一张图像等价于 16×16 个词：用于大规模图像识别的 Transformer](https://arxiv.org/abs/2010.11929)——ViT 论文。
- [Touvron 等（2021），通过注意力蒸馏训练数据高效的视觉 Transformer](https://arxiv.org/abs/2012.12877)——DeiT。
- [Liu 等（2021），Swin Transformer：使用移位窗口的层次化视觉 Transformer](https://arxiv.org/abs/2103.14030)——Swin。
- [Oquab 等（2023），DINOv2：无监督学习稳健视觉特征](https://arxiv.org/abs/2304.07193)——DINOv2。
- [Darcet 等（2023），视觉 Transformer 需要寄存器](https://arxiv.org/abs/2309.16588)——DINOv2 的寄存器词元修复方案。
