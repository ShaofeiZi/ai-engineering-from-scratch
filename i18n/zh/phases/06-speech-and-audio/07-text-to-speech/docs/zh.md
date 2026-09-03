# 文本转语音（TTS）——从 Tacotron 到 F5 与 Kokoro

> ASR 把语音转换为文本，TTS 则把文本转换为语音。2026 年的技术栈由三部分组成：文本 → 词元、词元 → 梅尔特征、梅尔特征 → 波形。每一部分都有可在笔记本电脑上运行的默认模型。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图与梅尔特征）、阶段 5 · 09（序列到序列）、阶段 7 · 05（完整 Transformer）
**Time:** 约 75 分钟

## 问题

你有一个字符串：“Please remind me to water the plants at 6 pm.”你需要生成一段 3 秒音频，要求听起来自然、韵律（停顿、重音）正确、以正确的元音发出“plants”，并且能在 CPU 上以低于 300 毫秒的时间运行，服务实时语音助手。你还需要切换声音、处理语码转换输入（“remind me at 6 pm, daijoubu?”），并且不要在姓名发音上出丑。

现代 TTS 流水线如下：

1. **文本前端。** 规范化文本（日期、数字、电子邮件），转换为音素或子词词元，并预测韵律特征。
2. **声学模型。** 文本 → 梅尔频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** 梅尔特征 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022），以及 2024 年后的神经编解码器声码器。

到 2026 年，端到端扩散和流匹配模型已经模糊了声学模型与声码器之间的界线。但在调试时，这个三部分心智模型仍然成立。

## 概念

![并列比较 Tacotron、FastSpeech、VITS、F5/Kokoro](../../../../../../phases/06-speech-and-audio/07-text-to-speech/assets/tts.svg)

**Tacotron 2（2017）。** 序列到序列：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器输出梅尔帧。速度慢（自回归），处理长文本时不稳定，但仍作为基线被引用。

**FastSpeech 2（2020）。** 非自回归。时长预测器为每个音素预测对应的梅尔帧数量。单次前向传播，比 Tacotron 快 10 倍。单调对齐会损失一些自然度，但它已被广泛部署。

**VITS（2021）。** 通过变分推断端到端联合训练编码器、基于流的时长模型和 HiFi-GAN 声码器。质量高，只需一个模型。它是 2022～2024 年占主导地位的开源 TTS。变体包括 YourTTS（多说话人零样本）和 XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer。韵律自然，只需 5 秒参考音频即可进行零样本声音克隆。在 2026 年开源 TTS 排行榜上居于首位，拥有 3.35 亿参数。

**Kokoro（2024）。** 小巧（8200 万参数）、可在 CPU 上运行，是实时场景中顶尖的英语 TTS。采用封闭英语词表，许可证为 Apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业领域的顶尖模型。ElevenLabs v2.5 的情绪标签（“[whispered]”“[laughing]”）和角色声音在 2026 年有声读物制作中占据主导。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅离线 | 发布时的顶尖水平 |
| 2018 | WaveRNN | 约实时 | 良好 |
| 2020 | HiFi-GAN | 100× 实时 | 接近人类 |
| 2022 | BigVGAN | 50× 实时 | 可泛化到不同说话人与语言 |
| 2024 | SNAC、DAC（神经编解码器） | 与自回归模型集成 | 离散词元、比特效率高 |

到 2026 年，大多数“TTS”模型都已实现文本到波形的端到端处理，梅尔频谱图成为内部表示。

### 评估

- **MOS（平均意见分）。** 1～5 分，由众包人员评分。仍是黄金标准，但速度极慢。
- **CMOS（比较式 MOS）。** A/B 偏好。每个标注可以获得更窄的置信区间。
- **UTMOS、DNSMOS。** 无参考的神经 MOS 预测器，用于排行榜。
- **通过 ASR 计算 CER（字符错误率）。** 把 TTS 输出送入 Whisper，再与输入文本计算 CER，作为可懂度的代理指标。
- **SECS（说话人嵌入余弦相似度）。** 衡量声音克隆质量。

LibriTTS test-clean 上的 2026 年数据：

| 模型 | UTMOS | CER（通过 Whisper） | 大小 |
|-------|-------|-------------------|------|
| 真实语音 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## 动手构建

### 第 1 步：把输入转换为音素

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。不要把原始文本送入质量低于 VITS 水平的模型。

### 第 2 步：运行 Kokoro（2026 年 CPU 默认方案）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单个文件，8200 万参数。

### 第 3 步：使用 F5-TTS 进行声音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

提供一段 5 秒参考音频及其转写，F5 就会克隆韵律与音色。

### 第 4 步：从零实现 HiFi-GAN 声码器

它太大，无法放进教程脚本，但整体形态如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练目标包括：对抗损失（判别器观察短窗口）+ 梅尔频谱图重建损失 + 特征匹配损失。声码器已经商品化——直接使用 `hifi-gan` 代码库或 NVIDIA NeMo 提供的预训练检查点。

### 第 5 步：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 使用 5 秒参考音频克隆声音 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声读物旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5～20 小时目标语言数据上训练 VITS |
| 富表现力/情绪标签 | ElevenLabs v2.5 或微调 StyleTTS 2 |

截至 2026 年的开源领先者：**F5-TTS 擅长质量，Kokoro 擅长效率**。除非你是历史研究者，否则不要选择 Tacotron。

## 陷阱

- **没有文本规范化器。** “Dr. Smith”应该读作“Doctor”还是“Drive”？“2026”应该读作“twenty twenty six”还是“two zero two six”？必须在音素转换器*之前*规范化。
- **词表外专有名词。** “Ghumare”→“ghyu-mair”？为未知词元提供字素到音素模型作为后备。
- **削波。** 声码器输出很少发生削波，但推理时梅尔缩放不匹配可能让幅度超出 ±1.0。始终执行 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz，下游流水线却期望 16 kHz → 必须重采样，否则会发生混叠。

## 交付成果

保存为 `outputs/skill-tts-designer.md`。针对给定声音、延迟与语言目标设计 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会根据玩具词表构建音素字典，估计每个音素的时长，并打印伪造的“梅尔”调度。
2. **中等。** 安装 Kokoro，分别使用 `af_bella` 和 `am_adam` 合成同一个句子，比较音频时长和主观质量。
3. **困难。** 录制一段自己的 5 秒参考音频，使用 F5-TTS 克隆声音，并报告参考音频与克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 音素 | 声音单元 | 抽象的语音类别；英语 ARPABet 中有 39 个。 |
| 时长预测器 | 每个音素持续多久 | 非自回归模型输出；每个音素对应的整数帧数。 |
| 声码器 | 梅尔特征 → 波形 | 将梅尔频谱图映射为原始采样的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；在 2020～2024 年占据主导。 |
| MOS | 主观质量 | 人类评分者给出的 1～5 平均意见分。 |
| SECS | 声音克隆指标 | 目标语音与输出语音的说话人嵌入余弦相似度。 |
| F5-TTS | 2024 年开源顶尖模型 | 流匹配扩散；支持零样本克隆。 |
| Kokoro | CPU 英语领先模型 | 8200 万参数，Apache 2.0。 |

## 延伸阅读

- [Shen 等（2017），Tacotron 2](https://arxiv.org/abs/1712.05884)——序列到序列基线。
- [Kim、Kong、Son（2021），VITS](https://arxiv.org/abs/2106.06103)——端到端流模型。
- [Chen 等（2024），F5-TTS](https://arxiv.org/abs/2410.06885)——当前开源顶尖方案。
- [Kong、Kim、Bae（2020），HiFi-GAN](https://arxiv.org/abs/2010.05646)——2026 年仍在生产中使用的声码器。
- [Hugging Face 上的 Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)——2024 年发布、适合 CPU 的英语 TTS。
