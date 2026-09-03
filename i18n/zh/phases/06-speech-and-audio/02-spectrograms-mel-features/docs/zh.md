# 频谱图、梅尔尺度与音频特征

> 神经网络不善于直接处理原始波形，却很适合处理频谱图，尤其是梅尔频谱图。从 2010 年到 2026 年，每个 ASR、TTS 和音频分类器的成败都取决于这一项预处理选择。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 01（音频基础）
**Time:** 约 45 分钟

## 问题

取一段 16 kHz、10 秒的音频。它由 16 万个取值位于 `[-1, 1]` 的浮点数组成，与“狗叫”或“cat 这个词”等标签几乎毫无直接相关性。原始波形包含信息，却不是模型容易提取的形式。相隔 100 毫秒说出的两个相同音素，其原始采样值会完全不同。

频谱图解决了这个问题。它压缩人类感知会忽略的时间细节（微秒级抖动），保留感知真正关注的结构（在约 10～25 毫秒时间窗口内，哪些频率具有较强能量）。

梅尔频谱图更进一步。人类以对数方式感知音高：100 Hz 与 200 Hz 听起来的“距离”，和 1000 Hz 与 2000 Hz 相同。梅尔尺度会扭曲频率轴，使其符合这种感知。2010 至 2026 年间，梅尔频谱图一直是语音机器学习中最重要的单项特征。

## 概念

![从波形到 STFT、梅尔频谱图和 MFCC 的阶梯](../../../../../../phases/06-speech-and-audio/02-spectrograms-mel-features/assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 把波形切成相互重叠的帧（典型设置：25 毫秒窗口、10 毫秒步长，在 16 kHz 下分别为 400 和 160 个样本）。为每帧乘以窗口函数（Hann 是默认选择；Hamming 的权衡略有不同），再逐帧执行 FFT。将幅度谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵，这就是频谱图。

**对数幅度。** 原始幅度横跨 5～6 个数量级。使用 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 压缩动态范围。每条生产流水线都使用对数幅度，而不是原始幅度。

**梅尔尺度。** 以 Hz 为单位的频率 `f` 会映射为梅尔值 `m`，公式为 `m = 2595 * log10(1 + f / 700)`。该映射在 1 kHz 以下大致呈线性，在更高频率上大致呈对数。覆盖 0～8 kHz 的 80 个梅尔分箱，是 ASR 的标准输入。

**梅尔滤波器组。** 一组在梅尔尺度上等距排列的三角滤波器。每个滤波器都是相邻 FFT 分箱的加权和。将 STFT 幅度与滤波器组矩阵相乘，一次矩阵乘法即可得到梅尔频谱图。

**对数梅尔频谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入、Parakeet 的输入、SeamlessM4T 的输入，也是 2026 年通用的音频前端。

**MFCC。** 对对数梅尔频谱图执行 DCT（二型离散余弦变换），保留前 13 个系数。这会解相关并进一步压缩特征。它在约 2015 年前一直是主导特征，随后直接处理原始对数梅尔特征的 CNN/Transformer 追了上来。说话人识别（x-vector、ECAPA）中仍在使用它。

**分辨率权衡。** FFT 越大，频率分辨率越好，时间分辨率却越差。25 毫秒窗口/10 毫秒步长是音频机器学习默认值；音乐使用 50 毫秒/12.5 毫秒；瞬态检测（鼓点、爆破音）使用 5 毫秒/2 毫秒。

```figure
spectrogram-window
```

## 动手构建

### 第 1 步：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 16 kHz、10 秒的音频，在 `frame_len=400, hop=160` 时会得到 998 帧。

### 第 2 步：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在执行 FFT 前逐元素相乘。这可以消除在非零端点处截断信号所造成的频谱泄漏。

### 第 3 步：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（由 FFT 支持且已向量化）。这里的循环仅用于教学，可在 `code/main.py` 中处理短音频。

### 第 4 步：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

使用 `n_fft=400` 时，覆盖 0～8 kHz 的 80 个梅尔分箱会得到一个 `(80, 201)` 矩阵。将形状为 `(n_frames, 201)` 的 STFT 幅度乘以该矩阵的转置，即可得到形状为 `(n_frames, 80)` 的梅尔频谱图。

### 第 5 步：对数梅尔特征

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案包括 `librosa.power_to_db`（以参考值归一化的 dB）和 `10 * log10(power + eps)`。Whisper 使用了更复杂的裁剪 + 归一化流程（见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每一帧对数梅尔特征执行 DCT，保留前 13 个系数，就得到了 MFCC 矩阵。通常会丢弃第一个系数，因为它编码的是整体能量。

## 学以致用

2026 年的技术栈：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 维对数梅尔特征，10 毫秒步长，25 毫秒窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 维梅尔特征，5～12 毫秒步长以实现精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 维对数梅尔特征，10 毫秒步长 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 维对数梅尔特征或原始波形自监督学习 |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散词元（不是梅尔特征） |
| 关键词唤醒 | 用于微型设备的 40 维 MFCC |

经验法则：**如果处理的不是音乐，就从 80 维对数梅尔特征开始。** 任何偏离都需要给出证据。

## 2026 年仍会进入生产的陷阱

- **梅尔分箱数量不匹配。** 训练时使用 80 维梅尔特征，推理时却使用 128 维，系统会悄然失效。应在两端记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 下计算的梅尔特征不同于 16 kHz。必须在特征提取*之前*修正采样率。
- **dB 与 log 混淆。** Whisper 期望对数梅尔特征，而不是 dB 梅尔特征。某些 Hugging Face 流水线会自动检测，自定义代码不会。
- **归一化漂移。** 训练时逐话语归一化，推理时却全局归一化。这类生产错误会让词错误率翻倍。
- **填充泄漏。** 在音频片段末尾补零，会让尾部帧产生平坦频谱。应对称填充或复制边缘。

## 交付成果

保存为 `outputs/skill-feature-extractor.md`。这个技能会根据目标模型选择特征类型、梅尔分箱数、帧长/步长和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它会合成一个频率从 200 Hz 扫到 4000 Hz 的啁啾信号，并打印每一帧幅度最大的梅尔分箱。可以选择绘图，并确认分箱变化与扫频一致。
2. **中等。** 使用 `n_mels` ∈ `{40, 80, 128}` 和 `frame_len` ∈ `{200, 400, 800}` 重新运行。测量尖锐峰值在时间轴上的带宽。哪种组合最能解析啁啾信号？
3. **困难。** 实现 `power_to_db`，并在 AudioMNIST 上比较微型 CNN 分类器采用以下特征时的准确率：（a）原始对数梅尔；（b）`ref=max` 的 dB 梅尔；（c）MFCC-13 + 一阶差分 + 二阶差分。报告 top-1 准确率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 帧 | 一个切片 | 送入一次 FFT 的 25 毫秒波形片段。 |
| 步长 | 步幅 | 相邻帧之间的样本数；ASR 默认为 10 毫秒。 |
| 窗函数 | Hann/Hamming 那个东西 | 逐点乘法器，让一帧的边缘逐渐衰减到零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗的 FFT，生成时间 × 频率矩阵。 |
| 梅尔 | 扭曲后的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 | 那个矩阵 | 把 STFT 投影到梅尔分箱的三角滤波器。 |
| 对数梅尔 | Whisper 的输入 | `log(mel_spec + eps)`；2026 年的标准形式。 |
| MFCC | 传统特征 | 对数梅尔特征的 DCT；13 个彼此解相关的系数。 |

## 延伸阅读

- [Davis、Mermelstein（1980），单音节词识别的参数化表示比较](https://ieeexplore.ieee.org/document/1163420)——MFCC 论文。
- [Stevens、Volkmann、Newman（1937），测量音高心理量的尺度](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/)——最初的梅尔尺度。
- [OpenAI——Whisper 源码中的 log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py)——请阅读参考实现。
- [librosa 特征提取文档](https://librosa.org/doc/main/feature.html)——`mfcc`、`melspectrogram` 及步长/窗口的参考资料。
- [NVIDIA NeMo——音频预处理](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers)——Parakeet + Canary 的生产级流水线。
