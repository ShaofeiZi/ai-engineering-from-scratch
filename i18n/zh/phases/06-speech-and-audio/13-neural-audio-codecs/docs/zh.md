# 神经音频编解码器——EnCodec、SNAC、Mimi、DAC 与语义—声学分离

> 2026 年的音频生成几乎都建立在词元之上。EnCodec、SNAC、Mimi 和 DAC 把连续波形转换成 Transformer 可以预测的离散序列。语义词元与声学词元的分离——第一个码本表达语义，其余码本表达声学信息——是继 Transformer 之后音频领域最重要的架构转变。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图）、阶段 10 · 11（量化）、阶段 5 · 19（子词分词）
**Time:** 约 60 分钟

## 问题

语言模型处理离散词元，音频却是连续信号。要为语音或音乐构建类似大语言模型的系统——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——首先需要一个**神经音频编解码器**：通过学习得到的编码器把音频离散化为小型码本中的序列，再由匹配的解码器重建波形。

已经形成两个家族：

1. **重建优先的编解码器**——EnCodec、DAC。优化感知音频质量。词元属于“声学”表示，会捕捉说话人身份、音色、背景噪声等一切信息。
2. **语义优先的编解码器**——Mimi（Kyutai）、SpeechTokenizer。强制第一个码本编码语言/音素内容（通常通过 WavLM 蒸馏），后续码本则编码声学细节。

2024～2026 年的关键认识是：**尝试从文本生成时，纯重建编解码器会产生模糊语音。** 作用于编解码器词元的大语言模型必须在同一个码本中同时学习语言结构与声学结构，这种方案难以扩展。把二者分开——码本 0 表达语义，码本 1～N 表达声学信息——正是 Moshi 和 Sesame CSM 能够奏效的原因。

## 概念

![四种编解码器版图：EnCodec、DAC、SNAC（多尺度）、Mimi（语义 + 声学）](../../../../../../phases/06-speech-and-audio/13-neural-audio-codecs/assets/codec-comparison.svg)

### 核心技巧：残差向量量化（RVQ）

现代音频编解码器不会使用一个大型码本（高质量需要数百万个编码），而是都采用 **RVQ**：级联多个小型码本。第一个码本量化编码器输出，第二个量化剩余残差，依此类推。每个码本包含 1024 个编码。8 个码本构成的有效词表大小为 1024^8 = 10^24。

推理时，解码器把每一帧选中的所有编码相加，以重建音频。

### 2026 年最重要的四种编解码器

**EnCodec（Meta，2022）。** 基线方案。在波形上运行的编码器—解码器，以 RVQ 为瓶颈。采样率 24 kHz，最多可使用 32 个码本，默认以 4 个码本达到 1.5 kbps。架构为 `1D conv + transformer + 1D conv`，MusicGen 使用它。

**DAC（Descript，2023）。** 使用 L2 归一化码本、周期激活函数和改进损失的 RVQ。它是开放编解码器中重建保真度最高的方案——使用 12 个码本时，有时与原始语音难以分辨。支持 44.1 kHz 全频带。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ——粗粒度码本的帧率低于精细码本。它实际上按层次建模音频：约 12 Hz 的粗略“草图”，加上 50 Hz 的细节。Orpheus-3B 使用它，因为这种层次结构很适合基于语言模型的生成。

**Mimi（Kyutai，2024）。** 改变 2026 年格局的模型。帧率仅 12.5 Hz（极低），8 个码本达到 4.4 kbps。码本 0 **由 WavLM 蒸馏而来**——训练目标是预测 WavLM 的语音内容特征；码本 1～7 表达声学残差。这种分离支撑了 Moshi（第 15 课）与 Sesame CSM。

### 帧率对语言建模至关重要

帧率越低，序列越短，语言模型越快。

| 编解码器 | 帧率 | 1 秒 = N 帧 | 适用场景 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗粒度） | 约 12 Hz | 12 | 高效自回归语言模型 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，10 秒话语只有 125 个编解码器帧——Transformer 可以轻松预测。

### 语义词元与声学词元

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义词元（Mimi 中的码本 0）。** 编码说了什么——音素、词语、内容。通过辅助预测损失从 WavLM 蒸馏而来。
- **声学词元（码本 1～7）。** 编码音色、说话人身份、韵律、背景噪声和精细细节。

自回归语言模型先预测语义词元（以文本为条件），再预测声学词元（以语义 + 说话人参考为条件）。现代 TTS 之所以能零样本克隆声音，正是因为这种分解：语义模型处理内容，声学模型处理音色。

### 2026 年重建质量（每秒比特数，码率越低越好）

| 编解码器 | 码率 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

从单位比特的感知质量来看，Opus 等传统编解码器仍然胜出。神经编解码器的优势在于**离散词元**（Opus 无法提供）和**生成模型质量**（语言模型可以利用这些词元完成什么）。

```figure
rvq-codec-cascade
```

## 动手构建

### 第 1 步：使用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 时，`n_codebooks=8`。每个编码的取值为 0～1023（10 比特）。

### 第 2 步：解码并测量重建质量

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 第 3 步：语义—声学分离（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本 0 与 WavLM 对齐。可以训练一个文本到语义的 Transformer——它的词表比直接生成音频小得多。然后由独立的声学到波形解码器，以说话人参考作为条件完成生成。

### 第 4 步：为何在编解码器词元上运行自回归语言模型有效

对于一段 10 秒语音，采用 Mimi 的 12.5 Hz × 8 个码本：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个词元对 Transformer 而言只是很短的上下文。现代 GPU 上，一个 2.56 亿参数的 Transformer 可以在数毫秒内生成 10 秒语音。

## 学以致用

从问题映射到编解码器：

| 任务 | 编解码器 |
|------|-------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真度重建 | DAC-44.1k |
| 在语音上运行自回归语言模型（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文本描述的音效库 | EnCodec + T5 条件 |
| 精细音频编辑 | DAC + 局部重绘 |

经验法则：**构建生成模型时，从 Mimi 或 SNAC 开始；构建压缩流水线时，使用 Opus。**

## 陷阱

- **码本太多。** 增加码本会线性提高保真度，也会线性增加语言模型序列长度。应在 8～12 个停止。
- **帧率不匹配。** 在 12.5 Hz Mimi 上训练语言模型，再用 50 Hz EnCodec 微调，会悄然失败。
- **假设所有码本同等重要。** 在 Mimi 中，码本 0 携带内容；丢失它会摧毁可懂度，而丢失码本 7 几乎听不出变化。
- **把重建质量当成唯一指标。** 如果语义结构不佳，编解码器即使重建效果出色，也可能无法用于基于语言模型的生成。

## 交付成果

保存为 `outputs/skill-codec-picker.md`。针对具体生成或压缩任务选择编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。它实现一个玩具标量 + 残差量化器，并测量增加码本时重建误差的变化。
2. **中等。** 安装 `encodec`，在留出的语音片段上比较使用 1、4、8、32 个码本的效果。绘制 PESQ 或 MSE 随码率变化的曲线。
3. **困难。** 加载 Mimi 并编码一段音频。先把码本 0 替换为随机整数并解码，再以同样方式替换码本 7。比较两种破坏——码本 0 受损应摧毁可懂度，码本 7 受损则几乎不会改变内容。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | 残差量化 | 级联多个小型码本，每个码本量化上一层的残差。 |
| 帧率 | 编解码器速度 | 每秒有多少个词元帧；越低，语言模型越快。 |
| 语义码本 | 码本 0（Mimi） | 从自监督学习特征蒸馏而来，编码内容。 |
| 声学码本 | 其余所有码本 | 编码音色、韵律、噪声和精细细节。 |
| PESQ / ViSQOL | 感知质量 | 与 MOS 相关的客观指标。 |
| EnCodec | Meta 编解码器 | RVQ 基线，MusicGen 使用它。 |
| Mimi | Kyutai 编解码器 | 12.5 Hz 帧率；语义—声学分离；支撑 Moshi。 |

## 延伸阅读

- [Défossez 等（2023），EnCodec](https://arxiv.org/abs/2210.13438)——RVQ 基线。
- [Kumar 等（2023），Descript Audio Codec（DAC）](https://arxiv.org/abs/2306.06546)——开放模型中的最高保真度方案。
- [Siuzdak（2024），SNAC](https://arxiv.org/abs/2410.14411)——多尺度 RVQ。
- [Kyutai（2024），Mimi 编解码器](https://kyutai.org/codec-explainer)——语义—声学分离、WavLM 蒸馏。
- [Borsos 等（2023），AudioLM](https://arxiv.org/abs/2209.03143)——两阶段语义/声学范式。
- [Zeghidour 等（2021），SoundStream](https://arxiv.org/abs/2107.03312)——最初的可流式 RVQ 编解码器。
