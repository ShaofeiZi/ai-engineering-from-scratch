# OCR 与文档理解

> OCR 是一条三阶段流水线——检测文本框、识别字符、再恢复版面。每一种现代 OCR 系统，都只是在重新排列这些阶段或把它们合并起来。

**Type:** 学习 + 使用
**Languages:** Python
**Prerequisites:** 第 4 阶段第 06 课（目标检测）、第 7 阶段第 02 课（自注意力）
**Time:** 约 45 分钟

## 学习目标

- 追踪经典 OCR 流水线（检测 -> 识别 -> 版面），并理解现代端到端替代方案（Donut、Qwen-VL-OCR）
- 实现用于序列到序列 OCR 训练的 CTC（连接时序分类）损失
- 使用 PaddleOCR 或 EasyOCR，无需训练即可完成生产级文档解析
- 区分 OCR、版面解析与文档理解，并为每种任务选择正确工具

## 问题所在

包含大量文字的图像随处可见：收据、发票、身份证件、扫描书籍、表单、白板、招牌和截图。从中提取结构化数据——不仅识别字符，还要判断“这是总金额”——是应用视觉中价值最高的问题之一。

这个领域可以分成三层能力：

1. **OCR 本身：** 把像素转换成文字。
2. **版面解析：** 把 OCR 输出组合成区域，例如标题、正文、表格和页眉。
3. **文档理解：** 从版面中提取结构化字段，例如“invoice_total = $42.50”。

每一层都有经典方法和现代方法，而且“我想从图像中获取文本”与“我需要这张收据的总金额”之间的差距，比大多数团队意识到的更大。

## 核心概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**会生成逐行或逐词的四边形区域。
- **文本识别**把每个区域裁剪到固定高度，再运行 CNN + BiLSTM + CTC，生成字符序列。
- **版面处理**重建阅读顺序。拉丁文字通常从上到下、从左到右，阿拉伯语和日语等语言则使用不同规则。

### 一段话理解 CTC

OCR 识别需要从固定长度的特征图生成长度可变的序列。CTC（Graves 等，2006）允许在没有字符级对齐信息时训练这种模型。模型会在每个时间步输出涵盖“词表 + 空白符”的概率分布；CTC Loss 会枚举并汇总所有经过“合并重复项、移除空白符”后能够还原目标文本的对齐路径。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 让 CRNN 在 2015 年取得成功，到 2026 年仍用于训练大多数生产级 OCR 模型。

### 现代端到端模型

- **Donut**（Kim 等，2022）——ViT 编码器 + 文本解码器；读取图像并直接输出 JSON，不需要文本检测器，也不需要版面模块。
- **TrOCR**——用于行级 OCR 的 ViT + Transformer 解码器。
- **Qwen-VL-OCR / InternVL**——针对 OCR 任务微调的完整视觉语言模型，在 2026 年的复杂文档上准确率最高。
- **PaddleOCR**——成熟生产套件中的经典 DB + CRNN 流水线，至今仍是开源主力。

端到端模型需要更多数据和计算资源，但避免了多阶段流水线中的误差累积。

### 版面解析

处理结构化文档时，应运行版面检测器（LayoutLMv3、DocLayNet），为每个区域标记 Title、Paragraph、Figure、Table 或 Footnote。阅读顺序随后就变成“按照版面顺序遍历各个区域并拼接”。

对于表单，应使用**键值提取**模型：视觉信息丰富的文档使用 Donut，普通扫描件使用 LayoutLMv3。它们接收图像、检测文本及其位置，并预测结构化键值对。

### 评估指标

- **字符错误率（CER）**——Levenshtein 距离 / 参考文本长度，越低越好。干净扫描件的生产目标是低于 2%。
- **词错误率（WER）**——同一指标，但在单词层面计算。
- **结构化字段 F1**——用于键值任务，衡量 `{invoice_total: 42.50}` 等结构是否正确出现。
- **JSON 编辑距离**——用于端到端文档解析；Donut 论文提出了归一化树编辑距离。

```figure
cv3-ctc-collapse
```

## 动手构建

### 第 1 步：CTC Loss 与贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 会在可用时采用高效的 CuDNN 实现。贪心解码器比 Beam Search 简单，而且 CER 通常只比它差 1% 以内。

### 第 2 步：微型 CRNN 识别器

这是用于行级 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入高度固定，CNN 通过最大池化把高度压到 1，宽度则成为 CTC 的时间维度。

### 第 3 步：合成 OCR

生成白底黑字的数字字符串，完成一次端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集还会加入字体、噪声、旋转、模糊和颜色变化，但使用的流水线完全相同。

### 第 4 步：训练概要

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这份简单合成数据上，200 步内损失应该从约 3 降低到约 0.2。

## 实际应用

生产环境有三条常见路径：

- **PaddleOCR**——成熟、快速、支持多语言。一行即可调用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**——原生 Python、多语言、使用 PyTorch 骨干网络。
- **Tesseract**——经典方案；模型难以处理旧扫描文档时仍然有用。

端到端文档解析可以使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于结构重复的收据、发票和表单，应微调 Donut；对于任意文档或需要推理的 OCR，Qwen-VL-OCR 等 VLM 是当前默认选择。

## 交付成果

本课会产出：

- `outputs/prompt-ocr-stack-picker.md`——根据文档类型、语言和结构，在 Tesseract / PaddleOCR / Donut / VLM-OCR 中作出选择的提示词。
- `outputs/skill-ctc-decoder.md`——从零编写贪心和 Beam Search CTC 解码器，并包含长度归一化的技能。

## 练习

1. **（简单）** 在 5 位随机数字字符串上训练 TinyCRNN 500 步，报告保留集上的 CER。
2. **（中等）** 用 Beam Search（beam_width=5）替换贪心解码，报告 CER 变化。Beam Search 在哪些输入上胜出？
3. **（困难）** 在一组 20 张收据上运行 PaddleOCR，提取明细项目，并根据手工标注的 {item_name, price} 真值对计算 F1。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| OCR | “从像素中读取文本” | 把图像区域转换成字符序列 |
| CTC | “无需对齐的损失” | 无需逐时间步标签即可训练序列模型，并对所有可能对齐求和的损失 |
| CRNN | “经典 OCR 模型” | 卷积特征提取器 + BiLSTM + CTC；2015 年提出的基线，至今仍用于生产 |
| Donut | “端到端 OCR” | 直接从图像输出 JSON 的 ViT 编码器 + 文本解码器；无需传统 OCR |
| 版面解析 | “寻找区域” | 检测文档中的标题、表格、图形、段落区域并分类 |
| 阅读顺序 | “文本序列” | 把识别出的区域排列成句子的规则；拉丁文本较简单，混合版面则很复杂 |
| CER / WER | “错误率” | 在字符或单词粒度上计算的 Levenshtein 距离 / 参考长度 |
| VLM-OCR | “会阅读的 LLM” | 经过 OCR 训练或提示的视觉语言模型，是复杂文档上的当前最佳方案 |

## 延伸阅读

- [《CRNN》（Shi 等，2015）](https://arxiv.org/abs/1507.05717)——CNN + RNN + CTC 原始架构
- [《CTC》（Graves 等，2006）](https://www.cs.toronto.edu/~graves/icml_2006.pdf)——CTC 原始论文，密集呈现了算法思想
- [《Donut》（Kim 等，2022）](https://arxiv.org/abs/2111.15664)——无需 OCR 的文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)——开源生产级 OCR 技术栈
