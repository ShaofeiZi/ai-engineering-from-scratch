# 构建语音助手流水线——阶段 6 综合项目

> 把第 01～11 课的所有内容连接起来。构建一个能够聆听、推理并开口回答的语音助手。到 2026 年，这已经是工程问题而非研究问题——但集成细节会决定它能否真正交付。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 04、05、06、07、11；阶段 11 · 09（函数调用）；阶段 14 · 01（智能体循环）
**Time:** 约 120 分钟

## 问题

构建一个端到端助手：

1. 采集麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始与结束。
3. 流式转写。
4. 把转写文本交给能够调用工具（计时器、天气、日历）的大语言模型。
5. 将大语言模型文本流式送入 TTS。
6. 通过扬声器向用户播放音频。
7. 如果用户在回答途中插话，则停止播放。

延迟目标：在笔记本电脑 CPU 上，从用户结束话语到首个 TTS 音频字节不超过 800 毫秒。质量目标：不漏词、不在静音上生成虚假字幕、不泄漏克隆声音、不让提示注入成功。

## 概念

![语音助手流水线：麦克风 → VAD → STT → 大语言模型 + 工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 毫秒数据块。在 Python 中通常使用 `sounddevice`，生产环境则使用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD，阈值 0.5，最短语音 250 毫秒，静音拖尾 500 毫秒，发出“开始”和“结束”信号。
3. **流式 STT（第 4～5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API），输出部分转写与最终转写。
4. **带工具调用的大语言模型。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。用 JSON Schema 定义工具，并流式输出词元。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快的开放方案）或 Cartesia Sonic（商业方案）。大语言模型输出 20 个词元后就启动 TTS。
6. **播放。** 输出至扬声器；低带宽网络使用 Opus 编码。
7. **打断处理器。** 如果 TTS 播放期间 VAD 被触发，就停止播放、取消大语言模型生成，并重新启动 STT。

### 你一定会遇到的三种失败模式

1. **首词截断。** VAD 启动慢了一拍，用户说的“hey”丢失。起始阈值应设为 0.3，而不是 0.5。
2. **回答途中打断混乱。** 用户插话后，大语言模型仍在生成；助手与用户抢话。必须连接 VAD → 取消大语言模型。
3. **静音幻觉。** Whisper 在静音预热帧上输出“Thanks for watching”。必须始终用 VAD 把关。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 许可证 | 说明 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350～500 毫秒 | 商业 API | 2026 年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500～800 毫秒 | 大多开放 | 适合自行构建 |
| Moshi（全双工） | 200～300 毫秒 | CC-BY 4.0 | 单模型；架构不同，见第 15 课 |
| Vapi / Retell（托管） | 300～500 毫秒 | 商业 | 上线最快；定制能力有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开放 | 隐私/边缘端 |

```figure
v4-voice-latency
```

## 动手构建

### 第 1 步：通过分块采集麦克风（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第 2 步：由 VAD 控制的轮次采集

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第 3 步：流式 STT → 大语言模型 → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第 4 步：在大语言模型循环中调用工具

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第 5 步：处理打断

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 学以致用

可运行的模拟见 `code/main.py`。它使用桩模型连接所有七个组件，因此即使没有硬件，你也能看清流水线的结构。要实现真实系统，可将桩替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- 用于 I/O 的 `sounddevice`

## 陷阱

- **永久记录个人身份信息。** 完整轮次音频在大多数司法辖区都属于个人身份信息。保留期应为 30 天，并进行静态加密。
- **没有打断机制。** 用户一定会插话，助手必须停止说话。
- **TTS 阻塞。** 同步 TTS 会阻塞事件循环。应使用异步调用或独立线程。
- **工具调用没有错误处理。** 工具会失败。大语言模型必须收到错误并重试一次，随后平稳降级。
- **幻觉过滤器过度激进。** 过滤太多，助手只会重复“I can't help with that.”；过滤不足，它又会什么都说。应在留出集上校准。
- **没有唤醒词选项。** 始终监听会带来隐私风险。应增加唤醒词门控（Porcupine 或 openWakeWord）。

## 交付成果

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言和合规约束，生成完整技术栈规格。

## 练习

1. **简单。** 运行 `code/main.py`。它会用桩模块模拟一个完整的端到端轮次，并打印各阶段延迟。
2. **中等。** 把 STT 桩替换为真实 Whisper 模型，对预录制的 `.wav` 文件进行转写，测量 WER 与端到端延迟。
3. **困难。** 增加工具调用：实现 `get_weather`（使用任意 API）和 `set_timer`。让大语言模型通过这些工具完成任务，并验证用户说“set a 5 minute timer”时调用了正确函数，且语音回复进行了确认。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 轮次 | 一次用户与助手往返 | 一段由 VAD 划定边界的用户语音 + 一次大语言模型—TTS 响应。 |
| 打断 | 插话 | 用户在助手说话时开口，助手停止输出。 |
| 唤醒词 | “Hey assistant” | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测 | 轮次结束 | VAD + 最短静音时长，用于判断用户已经说完。 |
| 预滚动 | 语音前缓冲 | 保留 VAD 触发前 200～400 毫秒的音频，避免截掉首词。 |
| 工具调用 | 函数调用 | 大语言模型输出 JSON；运行时分派执行；结果反馈回循环。 |

## 延伸阅读

- [LiveKit——语音智能体快速入门](https://docs.livekit.io/agents/)——生产级参考方案。
- [Pipecat——语音智能体示例](https://github.com/pipecat-ai/pipecat)——适合自行构建的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)——托管的原生语音路线。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)——全双工参考实现（第 15 课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/)——唤醒词门控。
- [Anthropic——工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)——大语言模型函数调用。
