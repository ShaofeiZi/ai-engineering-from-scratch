# 音频分类——从基于 MFCC 的 k-NN 到 AST 与 BEATs

> 从“狗叫还是警笛”到“这是什么语言”，都属于音频分类。特征使用梅尔表示，架构每十年都会变化，评估却始终围绕 AUC、F1 与逐类别召回率。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图与梅尔特征）、阶段 3 · 06（CNN）、阶段 5 · 08（用于文本的 CNN 与 RNN）
**Time:** 约 75 分钟

## 问题

拿到一段 10 秒的音频，你想知道：“这是什么？”城市声音（警笛、电钻、狗叫）、语音命令（是/否/停止）、语言识别（英语/西班牙语/阿拉伯语）、说话人情绪（愤怒/中性），或环境声音（室内/室外、人群嘈杂声）。所有这些都是*音频分类*。2026 年的基线架构已经十分成熟：对数梅尔特征 → CNN 或 Transformer → softmax。

核心难点不在网络，而在数据。音频数据集存在严重的类别不平衡、显著的领域漂移（干净与嘈杂环境），以及标签噪声（由谁来界定“城市嘈杂声”与“餐厅噪声”？）。这个问题有 80% 在于数据整理、增强与评估，而不是把 CNN 换成 Transformer。

## 概念

![音频分类阶梯：从基于 MFCC 的 k-NN 到 AST 与 BEATs](../../../../../../phases/06-speech-and-audio/03-audio-classification/assets/audio-classification.svg)

**基于 MFCC 的 k-NN（20 世纪 90 年代基线）。** 将每个音频片段的 MFCC 展平，计算它与带标签样本库的余弦相似度，再返回前 K 个近邻的多数票。在干净的小型数据集（Speech Commands、ESC-50）上，它的表现出人意料地强，而且无须 GPU 即可运行。

**对数梅尔特征上的二维 CNN（2015～2019）。** 把形状为 `(T, n_mels)` 的对数梅尔特征视作图像，应用 ResNet-18 或 VGG 风格网络，对时间轴执行全局均值池化，再通过 softmax 分类。到 2026 年，它仍是大多数 Kaggle 竞赛的基线。

**音频频谱图 Transformer，AST（2021～2024）。** 把对数梅尔频谱图划分为图块（例如 16×16 图块），加入位置嵌入，再送入 ViT。在 AudioSet 监督学习上达到顶尖水平（mAP 0.485）。

**BEATs 与 WavLM-base（2024～2026）。** 在数百万小时音频上进行自监督预训练。只需以过去所需监督数据的 1%～10% 在你的任务上微调。到 2026 年，这是非语音音频任务的默认起点。BEATs-iter3 只使用 AST 四分之一的算力，却在 AudioSet 上高出 1～2 个 mAP 点。

**以 Whisper 编码器作为冻结骨干网络（2024）。** 取出 Whisper 编码器，移除解码器，再连接一个线性分类器。无须任何音频增强，就能在语言识别和简单事件分类上取得接近顶尖的表现。这是近乎“免费午餐”的基线。

### 类别不平衡才是真正的挑战

ESC-50：50 个类别，每类 40 个片段——平衡且简单。UrbanSound8K：10 个类别，不平衡比例为 10:1。AudioSet：632 个类别，呈现 100000:1 的长尾分布。有效方法包括：

- 训练时进行平衡采样（评估时不要）。
- Mixup：将两个音频片段及其标签线性插值，作为数据增强。
- SpecAugment：随机遮蔽时间带与频率带。简单，却至关重要。

### 评估

- 互斥多分类（Speech Commands）：top-1 准确率、top-5 准确率。
- 多标签分类（AudioSet、UrbanSound 风格）：平均精确率均值（mAP）。
- 严重不平衡：逐类别召回率 + Macro-F1。

应当了解的 2026 年数据：

| 基准 | 基线 | 2026 年顶尖水平 | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 排行榜 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

```figure
mfcc-pipeline
```

## 动手构建

### 第 1 步：提取特征

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第 2 步：定长摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单却强大：在时间轴上计算均值与方差，可以把 13 系数 MFCC 转换成 26 维定长嵌入。它能够瞬间运行，而且直到 2017 年还曾击败 ESC-50 上最先进的神经网络基线。

### 第 3 步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第 4 步：升级为对数梅尔特征上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万个参数。使用单张 RTX 4090，在 ESC-50 上训练约 10 分钟即可达到 80% 以上的准确率。

### 第 5 步：2026 年的默认方案——微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，应使用 `microsoft/BEATs-base`，并通过 `beats` 库加载；其 transformers API 的形态相同。

## 学以致用

2026 年的技术栈：

| 场景 | 起始方案 |
|-----------|-----------|
| 微型数据集（少于 1000 个片段） | 基于 MFCC 均值的 k-NN（你的基线）+ 音频增强 |
| 中型数据集（1000～10 万） | 微调 BEATs 或 AST |
| 大型数据集（超过 10 万） | 从零训练，或微调 Whisper 编码器 |
| 实时、边缘端 | 40-MFCC CNN，量化为 int8（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + Mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结的骨干网络开始，而不是从头训练新模型。** 微调 BEATs 的输出头只需数小时，就能达到顶尖水平的 95%，而不是耗费数周。

## 交付成果

保存为 `outputs/skill-classifier-designer.md`。根据具体音频分类任务，选择架构、数据增强、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它会在四分类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 用 [均值、方差、偏度、峰度] 替换 `summarize`。在相同合成数据集上，四阶矩池化能否胜过均值 + 方差？
3. **困难。** 使用 `torchaudio`，在 ESC-50 的第 1 折上训练二维 CNN。报告五折交叉验证准确率。加入 SpecAugment（时间掩码 = 20，频率掩码 = 10），再报告变化量。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频领域的 ImageNet | Google 收集的 200 万片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 个类别 × 每类 40 个环境声音片段。 |
| AST | 音频频谱图 Transformer | 在对数梅尔图块上运行的 ViT；2021 年的顶尖方案。 |
| BEATs | 自监督音频模型 | Microsoft 模型；截至 2026 年，iter3 在 AudioSet 上领先。 |
| Mixup | 样本对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的数据增强 | 将频谱图中的随机时间带和频率带置零。 |
| mAP | 主要多标签指标 | 在所有类别与阈值上计算平均精确率的均值。 |

## 延伸阅读

- [Gong、Chung、Glass（2021），AST：音频频谱图 Transformer](https://arxiv.org/abs/2104.01778)——2021～2024 年的代表性架构。
- [Chen 等（2022，2024 修订），BEATs：使用声学分词器进行音频预训练](https://arxiv.org/abs/2212.09058)——2024 年后的默认方案。
- [Park 等（2019），SpecAugment](https://arxiv.org/abs/1904.08779)——主流音频增强方法。
- [Piczak（2015），ESC-50 数据集](https://github.com/karolpiczak/ESC-50)——经久不衰的 50 类基准。
- [Gemmeke 等（2017），AudioSet](https://research.google.com/audioset/)——包含 632 类的 YouTube 分类体系，至今仍是黄金标准。
