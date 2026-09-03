# 实时音频处理

> 批处理流水线处理完整文件，实时流水线则必须在下一个 20 毫秒到达前处理完当前的 20 毫秒。每个对话式 AI、广播演播室和电话机器人都成败于这份延迟预算。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图）、阶段 6 · 04（ASR）、阶段 6 · 07（TTS）
**Time:** 约 75 分钟

## 问题

你希望语音助手听起来像活人。人类对话的轮次交接延迟约为 230 毫秒（从沉默到响应）。超过 500 毫秒就显得机械，超过 1500 毫秒则让人觉得系统已经坏了。2026 年，完整的**听取 → 理解 → 回应 → 说话**循环预算如下：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 毫秒 |
| VAD | 10 毫秒 |
| ASR（流式） | 150 毫秒 |
| 大语言模型（首词元） | 100 毫秒 |
| TTS（首个音频块） | 100 毫秒 |
| 渲染 → 扬声器 | 20 毫秒 |
| **合计** | **约 400 毫秒** |

Moshi（Kyutai，2024）的全双工延迟达到 200 毫秒，GPT-4o-realtime（2024）约为 320 毫秒，而 2022 年的级联流水线交付时延迟达 2500 毫秒。这 10 倍提升来自三项技术：（1）所有环节流式化；（2）使用部分结果进行异步流水线处理；（3）可中断生成。

## 概念

![带环形缓冲区、VAD 门控与打断机制的流式音频流水线](../../../../../../phases/06-speech-and-audio/11-real-time-audio-processing/assets/real-time.svg)

**帧/块/窗口。** 实时音频以固定大小的数据块流动。常见选择是 20 毫秒（16 kHz 下为 320 个样本）。所有下游环节都必须跟上这个节奏。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取，在热点路径中避免内存分配。容量约等于最大延迟 × 采样率；16 kHz 下两秒的环形缓冲区包含 32000 个样本。

**VAD（语音活动检测）。** 没有人说话时，阻止下游继续工作。Silero VAD 4.0（2024）在 CPU 上处理每个 30 毫秒帧用时不到 1 毫秒。`webrtcvad` 是较早的替代方案。

**流式 ASR。** 随着音频到达，持续输出部分转写。流式模式下的 Parakeet-CTC-0.6B（NeMo，2024）能以 320 毫秒延迟达到 2%～5% WER。Whisper-Streaming（Macháček 等，2023）通过对 Whisper 分块，以约 2 秒延迟实现近流式处理。

**打断。** 当助手正在说话而用户插话时，必须：（a）检测插话；（b）停止 TTS；（c）丢弃大语言模型尚未输出的内容。整个过程必须在 100 毫秒内完成，否则用户会觉得助手听不见自己。

**WebRTC Opus 传输。** 20 毫秒帧、48 kHz、8～128 kbps 自适应比特率，是浏览器和移动端的标准。LiveKit、Daily.co 与 Pion 是 2026 年构建语音应用的常用技术栈。

**抖动缓冲区。** 网络数据包会乱序或延迟到达。抖动缓冲区负责重新排序和平滑；太小会出现声音缺口，太大则增加延迟。典型值为 60～80 毫秒。

### 常见陷阱

- **线程争用。** Python 的 GIL 加上大型模型可能饿死音频线程。应使用基于 C 回调的音频库（sounddevice、PortAudio），并让 Python 离开热点路径。
- **采样率转换延迟。** 流水线内部重采样会增加 5～20 毫秒。应提前重采样，或使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使 Kokoro 这样快速的 TTS，第一次请求也需要 100～200 毫秒预热。应缓存模型，并在首次真实对话前用虚拟输入预热。
- **回声消除。** 如果没有 AEC，TTS 输出会重新进入麦克风，触发 ASR 识别机器人自己的声音。WebRTC AEC3 是开源默认方案。

```figure
nyquist-aliasing
```

## 动手构建

### 第 1 步：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下的 32000 个样本等于 2 秒。

### 第 2 步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境应替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

这一机制依赖异步 I/O 与可以取消的流式 TTS。在音轨上调用 WebRTC peerconnection.stop() 是标准做法。

## 学以致用

2026 年的技术栈：

| 层 | 选择 |
|-------|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| 大语言模型首词元 | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 原生端到端 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **为了稳妥而缓冲 500 毫秒。** 缓冲区*本身*就是延迟下限，应缩小它。
- **没有固定线程。** 音频回调所在的线程优先级低于 UI 线程，系统在负载下就会出现杂音。
- **TTS 块太小。** 小于 200 毫秒的块会让声码器瑕疵变得可闻。320 毫秒是最佳平衡点。
- **没有抖动缓冲区。** 真实网络必然存在抖动，不做平滑就会产生爆音。
- **一次性错误处理。** 音频流水线必须具备崩溃防护；一次异常就可能终止整个会话。

## 交付成果

保存为 `outputs/skill-realtime-designer.md`。设计实时音频流水线，并为每个阶段给出具体延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。它会模拟环形缓冲区与能量 VAD，并打印一段虚拟 10 秒音频流各阶段的延迟。
2. **中等。** 使用 `sounddevice` 构建直通循环，以 20 毫秒为一帧处理麦克风输入，并逐帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲区 | 循环队列 | 用于音频帧的定长无锁（或 SPSC 加锁）FIFO。 |
| VAD | 静音门 | 标记语音与非语音的模型或启发式规则。 |
| 流式 ASR | 实时 STT | 随音频到达发出部分文本，前瞻范围有界。 |
| 抖动缓冲区 | 网络平滑器 | 重新排列乱序数据包的队列；典型值为 60～80 毫秒。 |
| AEC | 回声消除 | 去除扬声器到麦克风的反馈路径。 |
| 打断 | 用户插话 | 系统在 TTS 期间检测到用户说话，必须取消播放。 |
| 全双工 | 双向同时进行 | 用户与机器人可以同时说话；Moshi 是全双工系统。 |

## 延伸阅读

- [Macháček 等（2023），Whisper-Streaming](https://arxiv.org/abs/2307.14743)——通过分块实现近流式 Whisper。
- [Kyutai（2024），Moshi](https://kyutai.org/Moshi.pdf)——全双工、200 毫秒延迟。
- [LiveKit Agents 框架（2024）](https://docs.livekit.io/agents/)——生产级音频智能体编排。
- [Silero VAD 代码库](https://github.com/snakers4/silero-vad)——低于 1 毫秒的 VAD，Apache 2.0。
- [WebRTC AEC3 论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/)——开源回声消除。
