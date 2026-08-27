# 语音活动检测与轮次交接——Silero、Cobra 与 Flush 技巧

> 每个语音智能体的成败都取决于两个判断：用户现在是否正在说话，以及用户是否已经说完？VAD 回答第一个问题。轮次检测（VAD + 静音拖尾 + 语义端点模型）回答第二个。任何一个判断出错，助手要么打断用户，要么永远说个不停。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 11（实时音频）、阶段 6 · 12（语音助手）
**Time:** 约 45 分钟

## 问题

语音智能体每收到一个 20 毫秒音频块，就要作出三个不同的判断：

1. **这一帧是语音吗？**——VAD，逐帧二分类。
2. **用户开始说一段新话了吗？**——起点检测。
3. **用户说完了吗？**——端点检测（轮次结束）。

朴素答案（能量阈值）遇到任何噪声都会失效——车流声、键盘声、人群嘈杂声。2026 年的答案是：Silero VAD（开放、深度学习）+ 轮次检测模型（语义端点检测）+ 根据 VAD 校准的静音拖尾。

## 概念

![VAD 级联：能量 → Silero → 轮次检测器 → Flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第 1 层：能量门。** 成本最低。以 -40 dBFS 为 RMS 阈值。它能过滤明显的静音，但任何超过阈值的噪声都会触发它。

**第 2 层：Silero VAD**（2020～2026，MIT）。100 万参数，在 6000 多种语言上训练。单个 CPU 线程处理每个 30 毫秒音频块约需 1 毫秒。在 5% FPR 下达到 87.7% TPR，是开源默认方案。

**第 3 层：语义轮次检测器。** LiveKit 的轮次检测模型（2024～2026），或你自己的小型分类器。它区分“句中停顿”和“说完了”，使用语言上下文（语调 + 最近的词），而不只依赖静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；大于 0.5（默认）或大于 0.3（敏感模式）时判定为语音。阈值越低，首词截断越少，误报越多。
- **最短语音时长。** 拒绝短于 250 毫秒的语音——通常是咳嗽声或椅子噪声。
- **静音拖尾（端点检测）。** VAD 返回 0 后，等待 500～800 毫秒再宣布轮次结束。太短会打断用户，太长会显得迟钝。
- **预滚动缓冲。** 保留 VAD 触发前 300～500 毫秒的音频，防止“hey”被截掉。

### Flush 技巧（Kyutai，2025）

流式 STT 模型存在前瞻延迟（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常，你需要在语音结束后再等待这么久才能拿到转写。Flush 技巧是：VAD 发出语音结束信号时，**向 STT 发送 flush 信号**，强制其立即输出。STT 以约 4 倍实时速度处理，所以 500 毫秒缓冲只需约 125 毫秒即可完成。

端到端效果：125 毫秒 VAD + flush STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD | 5% FPR 下的 TPR | 延迟 | 许可证 |
|-----|--------------|---------|---------|
| WebRTC VAD（Google，2013） | 50.0% | 30 毫秒 | BSD |
| Silero VAD（2020～2026） | 87.7% | 约 1 毫秒 | MIT |
| Cobra VAD（Picovoice） | 98.9% | 约 1 毫秒 | 商业许可 |
| pyannote 分割 | 95% | 约 10 毫秒 | 类 MIT 许可 |

Silero 是正确的默认选择，Cobra 是面向合规与更高准确率的升级方案。2026 年的生产系统不应只使用能量 VAD。

```figure
sp-vad-cascade
```

## 动手构建

### 第 1 步：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 第 2 步：在 Python 中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 第 3 步：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 第 4 步：Flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

要使用这种方法，STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush。Whisper 流式方案不支持——它基于分块，始终需要等待音频块。

## 学以致用

| 场景 | VAD 选择 |
|-----------|-----------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/说话人分离 | pyannote 分割 |
| 零依赖后备方案 | WebRTC VAD（旧式） |
| 需要高质量轮次结束判断 | 分层使用 Silero + LiveKit 轮次检测器 |

经验法则：除非真的别无选择，否则绝不要只使用能量 VAD 投入生产。

## 陷阱

- **固定阈值。** 安静环境中有效，嘈杂环境中失效。应在设备上校准，或改用 Silero。
- **静音拖尾太短。** 智能体会在用户句中停顿时打断他。500～800 毫秒是对话语音的最佳平衡点。
- **拖尾太长。** 会让系统显得迟钝。应在目标用户中进行 A/B 测试。
- **没有预滚动缓冲。** 用户语音开头的 200～300 毫秒会丢失。始终维护滚动预缓冲。
- **忽略语义端点判断。** “Hmm, let me think...”中包含较长停顿。用户讨厌在思考途中被打断。应使用 LiveKit 的轮次检测器或类似模型。

## 交付成果

保存为 `outputs/skill-vad-tuner.md`。针对具体工作负载选择 VAD 模型、阈值、拖尾、预滚动与轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它会模拟语音 + 静音 + 语音 + 咳嗽组成的序列，并测试三层 VAD。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟录音，调节阈值以同时减少首词截断和误触发，并报告精确率与召回率。
3. **困难。** 构建微型轮次检测器：Silero VAD + 三层 MLP，输入最近 10 个词的嵌入（使用 sentence-transformers）。在手工标注的轮次结束数据集上训练，使 F1 比只用 Silero 高 10%。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| VAD | 语音检测器 | 逐帧二分类：这是语音吗？ |
| 轮次检测 | 端点检测 | VAD + 静音拖尾 + 语义端点。 |
| 静音拖尾 | 语音结束后等待 | 宣布轮次结束前等待的时间；通常为 500～800 毫秒。 |
| 预滚动 | 语音前缓冲 | 保留 VAD 触发前 300～500 毫秒的音频。 |
| Flush 技巧 | Kyutai 技巧 | VAD → 刷新 STT → 把 500 毫秒延迟缩短到 125 毫秒。 |
| 语义端点 | “用户真的说完了吗？” | 根据词语而不只根据静音判断的机器学习分类器。 |
| 5% FPR 下的 TPR | ROC 工作点 | 标准 VAD 基准；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad)——开放 VAD 参考方案。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/)——商业准确率领先者。
- [Kyutai——Unmute 与 Flush 技巧](https://kyutai.org/stt)——低于 200 毫秒的工程技巧。
- [LiveKit——轮次检测](https://docs.livekit.io/agents/logic/turns/)——生产环境中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/)——旧式基线。
- [pyannote 分割](https://github.com/pyannote/pyannote-audio)——说话人分离级分割。
