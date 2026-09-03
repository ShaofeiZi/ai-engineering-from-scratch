# 音频基础——波形、采样与傅里叶变换

> 波形是原始信号，频谱图是它的表示，梅尔特征则是适合机器学习的形式。每条现代 ASR 和 TTS 流水线都会沿着这组阶梯向上走，而第一阶就是理解采样与傅里叶变换。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 1 · 06（向量与矩阵）、阶段 1 · 14（概率分布）
**Time:** 约 45 分钟

## 问题

麦克风产生的是压力随时间变化的信号，神经网络接收的却是张量。二者之间存在一整套约定；一旦违反，就会产生悄无声息的错误：模型训练看似正常，但词错误率翻倍；TTS 生成嘶嘶声；语音克隆系统记住的是麦克风，而不是说话人。

语音系统中的每个问题，都可以追溯到以下三个问题之一：

1. 数据使用什么采样率录制，模型期望什么采样率？
2. 信号是否发生了混叠？
3. 你是在处理原始采样，还是频率表示？

这些问题处理正确，阶段 6 的其他内容就容易掌握；处理错误，即使 Whisper-Large-v4 也只能输出垃圾。

## 概念

![波形、采样、DFT 与频率分箱可视化](../../../../../../phases/06-speech-and-audio/01-audio-fundamentals/assets/audio-fundamentals.svg)

**波形。** 一个取值位于 `[-1.0, 1.0]` 的一维浮点数组，按采样编号索引。除以采样率即可转换为秒：`t = n / sr`。一段 16 kHz、10 秒的音频包含 16 万个浮点数。

**采样率（sr）。** 每秒采集的样本数。2026 年常见采样率如下：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话与旧式 VOIP。奈奎斯特频率只有 4 kHz，会损失辅音；ASR 应避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都接收 16 kHz。 |
| 22.05 kHz | 较早模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特—香农定理。** 采样率 `sr` 可以无歧义地表示最高 `sr/2` 的频率。`sr/2` 这个边界称为*奈奎斯特频率*。高于奈奎斯特频率的能量会发生*混叠*——折返到较低频率——从而污染信号。降采样前必须先使用低通滤波器。

**位深度。** 16 位 PCM（有符号 int16，范围 ±32767）是通用交换格式；音乐使用 24 位，内部数字信号处理使用 32 位浮点数。`soundfile` 等库读取 int16，但会公开取值位于 `[-1, 1]` 的 float32 数组。

**傅里叶变换。** 任何有限信号都可以表示为不同频率正弦波之和。离散傅里叶变换（DFT）对 `N` 个样本计算 `N` 个复数系数——每个频率分箱对应一个。`bin k` 映射到频率 `k · sr / N` Hz。模长代表该频率的振幅，相角代表相位。

**FFT。** 快速傅里叶变换：一种复杂度为 `O(N log N)` 的 DFT 算法，要求 `N` 是 2 的幂。每个音频库都在底层使用 FFT。对 16 kHz 音频执行 1024 点 FFT，会得到 512 个可用频率分箱，覆盖 0～8 kHz，分辨率为 15.6 Hz。

**分帧 + 加窗。** 我们不会对整段音频执行一次 FFT，而是把它切成相互重叠的*帧*（通常为 25 毫秒帧长、10 毫秒步长），将每一帧乘以窗口函数（Hann、Hamming）以消除边缘不连续，再逐帧执行 FFT。这就是短时傅里叶变换（STFT）。第 02 课将从这里继续。

```figure
mel-scale
```

## 动手构建

### 第 1 步：读取音频片段并绘制波形

`code/main.py` 仅使用标准库 `wave` 模块，使演示不依赖第三方库。生产环境中应使用 `soundfile` 或 `torchaudio.load`（二者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第 2 步：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

在 16 kHz 下生成 1 秒的 440 Hz 正弦波（音乐会标准音 A），会得到 16000 个浮点数。使用 16 位 PCM 编码，通过 `wave.open(..., "wb")` 写入文件。

### 第 3 步：手工计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

复杂度为 `O(N²)`——用于 `N=256` 时可以验证正确性，处理真实音频则毫无实用价值。真实代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第 4 步：找出主导频率

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行时，应当在分箱 `440 * N / sr` 处找到峰值。

### 第 5 步：演示混叠

以 10 kHz 采样率采样 7 kHz 正弦波（奈奎斯特频率 = 5 kHz）。7 kHz 音调高于奈奎斯特频率，会折返到 `10 − 7 = 3 kHz`；FFT 峰值会出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配备砖墙式低通滤波器的原因。

## 学以致用

2026 年实际交付时使用的技术栈：

| 任务 | 库 | 原因 |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 速度最快且稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠处理。 |
| STFT / 梅尔特征 | `torchaudio` 或 `librosa` | 支持 GPU，与 PyTorch 生态集成。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | 命令行工具，速度快，可报告采样率、声道数与编解码器。 |

决策规则：**在匹配其他任何内容之前，先匹配采样率**。Whisper 期望输入 16 kHz 单声道 float32。如果传入 44.1 kHz 立体声，输出会像模型出错一样糟糕。

## 交付成果

保存为 `outputs/skill-audio-loader.md`。这个技能帮助你检查音频输入是否符合下游模型的预期，并在不符合时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成一秒钟的 220 Hz + 440 Hz + 880 Hz 混合音，运行 DFT，确认预期分箱位置出现三个峰值。
2. **中等。** 以 48 kHz 录制一段 3 秒语音。先使用 `torchaudio.transforms.Resample`（带抗混叠）降采样至 16 kHz，再使用朴素抽取（每三个样本取一个）降采样至 16 kHz。对二者执行 FFT，混叠出现在哪里？
3. **困难。** 只使用 `math` 和第 3 步中的 DFT 从零构建 STFT。帧大小为 400，步长为 160，采用 Hann 窗。使用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是第 02 课中的频谱图。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率 | 每秒采集多少样本 | ADC 测量信号的频率，单位为 Hz。 |
| 奈奎斯特频率 | 可以表示的最高频率 | `sr/2`；高于它的能量会向下混叠。 |
| 位深度 | 每个样本的分辨率 | `int16` = 65536 个电平；`float32` = `[-1, 1]` 范围内的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | 复杂度为 `O(N log N)`、要求 `N` 为 2 的幂的算法。 |
| 分箱 | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层机制 | 分帧 + 加窗 + 随时间执行 FFT。 |
| 混叠 | 奇怪的频率幽灵 | 高于奈奎斯特频率的能量镜像到较低频率分箱。 |

## 延伸阅读

- [Shannon（1949），噪声环境中的通信](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)——采样定理背后的论文。
- [Smith——《科学家与工程师的数字信号处理指南》](https://www.dspguide.com/ch8.htm)——免费、经典的 DSP 教材。
- [librosa 文档——音频入门](https://librosa.org/doc/latest/tutorial.html)——配有代码的实践教程。
- [Heinrich Kuttruff——《室内声学》（第 6 版）](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434)——解释现实音频为何不是干净正弦波的参考资料。
- [Steve Eddins——FFT 解读笔记](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/)——十分钟理清频率分箱的直觉。
