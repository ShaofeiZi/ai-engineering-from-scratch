# 音频 Transformer——Whisper 架构

> 音频是一幅频率随时间变化的图像。Whisper 就是一个以梅尔频谱图为输入、再用文本作答的 ViT。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 7 · 05（完整 Transformer）、阶段 7 · 08（编码器—解码器）、阶段 7 · 09（ViT）
**Time:** 约 45 分钟

## 问题

在 Whisper（OpenAI，Radford 等，2022）出现之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 与 HuBERT——自监督特征提取器加微调输出头。它们质量很高，却需要昂贵的数据流水线，而且对领域变化十分脆弱。多语言语音识别还需要为不同语系分别准备模型。

Whisper 作出了三项选择：

1. **在所有数据上训练。** 从互联网抓取 68 万小时、覆盖 97 种语言的弱标注音频—文本对。不使用干净的学术语料库，也不需要音素标签。
2. **一个多任务模型。** 通过任务词元，让一个解码器同时学习转写、翻译、语音活动检测、语言识别和时间戳预测。
3. **标准 Transformer 编码器—解码器。** 编码器接收对数梅尔频谱图，解码器自回归地生成文本词元。没有声码器，没有 CTC，也没有 HMM。

结果是：Whisper large-v3 能够稳健处理口音、噪声，以及完全没有干净标注数据的语言。到 2026 年，它已经成为所有开源语音助手和大多数商业语音助手的默认语音前端。

## 概念

![Whisper 流水线：音频 → 梅尔特征 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 第 1 步——重采样 + 分窗

音频采用 16 kHz。裁剪或填充到 30 秒，计算对数梅尔频谱图：80 个梅尔分箱，10 毫秒步幅 → 约 3000 帧 × 80 维特征。这就是 Whisper 看到的“输入图像”。

### 第 2 步——卷积主干

两个卷积核为 3、步幅为 2 的 Conv1D 层，把 3000 帧减少到 1500 帧，在不增加大量参数的情况下将序列长度减半。

### 第 3 步——编码器

一个在 1500 个时间步上运行的 24 层 Transformer 编码器（large 版本）。使用正弦位置编码、自注意力和 GELU FFN，生成 1500 × 1280 个隐藏状态。

### 第 4 步——解码器

一个 24 层 Transformer 解码器。它从一个包含 GPT-2 词表并额外加入少量音频专用特殊词元的 BPE 词表中，自回归地生成词元。

### 第 5 步——任务词元

解码器提示以控制词元开头，用来告诉模型应该执行什么任务：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或者：

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按照这套约定训练的。你可以通过前缀控制任务。这相当于 2026 年的指令微调，只不过应用于语音。

### 第 6 步——输出

使用束宽为 5、带对数概率阈值的束搜索。如果提示中没有 `<|notimestamps|>` 词元，模型会以音频每 0.02 秒的间隔预测时间戳。

### Whisper 各种规模

| 模型 | 参数量 | 层数 | d_model | 头数 | 显存（fp16） |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | 约 1 GB |
| Base | 74M | 6 | 512 | 8 | 约 1 GB |
| Small | 244M | 12 | 768 | 12 | 约 2 GB |
| Medium | 769M | 24 | 1024 | 16 | 约 5 GB |
| Large | 1550M | 32 | 1280 | 20 | 约 10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | 约 10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | 约 6 GB（4 层解码器） |

Large-v3-turbo（2024）把解码器从 32 层缩减到 4 层，以不到 1 个 WER 点的退化换来 8 倍解码速度。这项解码提速让 Whisper-turbo 成为 2026 年实时语音智能体的默认选择。

### Whisper 不会做什么

- 不进行说话人分离（判断谁在说话），需要搭配 pyannote。
- 原生不支持实时流式处理——30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过 VAD + 重叠拼接出流式能力。
- 如果不在外部进行分块，就无法利用超过 30 秒的长程上下文。实践中仍然表现良好，因为语音转写很少需要长距离上下文。

### 2026 年的技术版图

| 任务 | 模型 | 说明 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘端快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可以达到 150 毫秒延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器—解码器模式，但形态类似 Whisper |
| 音频 + 语言 | AudioLM、SeamlessM4T | 在一个 Transformer 中同时处理文本与音频词元 |

```figure
n5-mel-decode
```

## 动手构建

见 `code/main.py`。我们不会训练 Whisper，而是构建对数梅尔频谱图流水线与任务词元提示格式器。它们才是生产环境中真正需要接触的部分。

### 第 1 步：合成音频

以 16 kHz 采样率生成一秒钟的 440 Hz 正弦波，共 16000 个样本。

### 第 2 步：对数梅尔频谱图（简化版）

完整的梅尔频谱图需要 FFT。这里使用简化的分帧 + 逐帧能量方案来展示流水线，无须依赖 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长为 25 毫秒，步幅为 10 毫秒，与 Whisper 的分窗方式一致。教学中使用逐帧能量代替梅尔分箱。

### 第 3 步：填充至 30 秒

Whisper 始终处理 30 秒音频块。将频谱图填充（或裁剪）到 3000 帧。

### 第 4 步：构建提示词元

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整的任务控制面：一个包含 4 个词元的前缀。

## 学以致用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

速度更快且与 OpenAI 兼容的方案：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用一个模型完成多语言 ASR。
- 稳健转写嘈杂且多样的音频。
- ASR 研究/原型开发——最快起点。

**何时选择其他方案：**

- 边缘端超低延迟流式处理——在质量相当时，Moonshine 比 Whisper 更快。
- 需要低于 200 毫秒延迟的实时对话式 AI——使用专用流式 ASR。
- 说话人分离——Whisper 不提供这项能力，需要外接 pyannote。

## 交付成果

见 `outputs/skill-asr-configurator.md`。该技能会为新的语音应用选择 ASR 模型、解码参数与预处理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz、步幅 10 毫秒的一秒信号约有 100 帧；30 秒约有 3000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的对数梅尔频谱图。验证 80 个梅尔分箱与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：把音频切成带 2 秒重叠的 10 秒窗口，逐块运行 Whisper，再合并转写结果。在一段 5 分钟播客样本上测量相对于单次处理的词错误率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 梅尔频谱图 | “音频图像” | 二维表示：一个轴为频率分箱，另一个轴为时间帧，每个单元是对数缩放后的能量。 |
| 对数梅尔 | “Whisper 看到的内容” | 经过对数变换的梅尔频谱图，近似人类对响度的感知。 |
| 帧 | “一个时间切片” | 25 毫秒采样窗口，以 10 毫秒步幅重叠。 |
| 任务词元 | “语音的提示前缀” | 解码器提示中的 `<\|transcribe\|>` / `<\|translate\|>` 等特殊词元。 |
| 语音活动检测（VAD） | “找出语音” | 在 ASR 前移除静音的门控，可大幅降低成本。 |
| CTC | “连接时序分类” | 用于无对齐训练的经典 ASR 损失；Whisper **不**使用它。 |
| Whisper-turbo | “小解码器、完整编码器” | large-v3 编码器 + 4 层解码器；解码速度快 8 倍。 |
| Faster-whisper | “生产封装” | CTranslate2 重实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford 等（2022），通过大规模弱监督实现稳健语音识别](https://arxiv.org/abs/2212.04356)——Whisper 论文。
- [OpenAI Whisper 代码库](https://github.com/openai/whisper)——参考代码与模型权重。阅读 `whisper/model.py`，可用约 400 行代码自顶向下理解 Conv1D 主干、编码器与解码器。
- [OpenAI Whisper——`whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py)——第 5～6 步所述的束搜索与任务词元逻辑；约 500 行，完全可以通读。
- [Baevski 等（2020），wav2vec 2.0：语音表示自监督学习框架](https://arxiv.org/abs/2006.11477)——前身；在某些设置中仍能提供顶尖特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)——生产封装，比参考实现快 4 倍。
- [Jia 等（2024），Moonshine：用于实时转写与语音命令的语音识别](https://arxiv.org/abs/2410.15608)——2024 年适合边缘端的 ASR，架构类似 Whisper，但更小。
- [Hugging Face 博客——“使用 Transformers 微调 Whisper 进行多语言 ASR”](https://huggingface.co/blog/fine-tune-whisper)——包含梅尔频谱图预处理器与词元时间戳处理的标准微调方案。
- [Hugging Face `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)——与课程架构图对应的完整实现（编码器、解码器、交叉注意力、生成）。
