# 综合项目 03——实时语音助手（从 ASR 到 LLM 再到 TTS）

> 一个用起来顺畅的语音智能体，端到端延迟要低于 800 毫秒，能判断用户何时说完，支持插话打断，并且调用工具时不会让音频停顿。到 2026 年，Retell、Vapi、LiveKit Agents 和 Pipecat 都达到了这一标准。它们的架构很相似：流式 ASR、轮次检测器、流式 LLM 和流式 TTS 通过 WebRTC 串联，每个环节都有严格的延迟预算。请亲手搭建一套这样的系统，测量字错率（WER）、平均意见分（MOS）和误截断率，再到丢包环境中检验它。

**Type:** 综合项目
**Languages:** Python（智能体 + 流水线），TypeScript（Web 客户端）
**Prerequisites:** 第 6 阶段（语音与音频）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 17 阶段（基础设施）
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 小时

## 问题

语音是 2025 至 2026 年变化最快的 AI 用户体验类别，可实现的延迟上限几乎每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70，使首包音频输出低于 800 毫秒成为可能。但门槛不只是延迟，交互是否自然同样重要：既不能抢用户的话，也不能一被用户打断就失去状态；需要从半句话的中断中恢复，在对话中途调用工具时保持音频连贯，还要经受住抖动明显的移动网络。

把三个 REST 调用拼在一起，做不到这种体验。整套架构必须从输入到输出都采用流式管线。系统搭起来后，故障模式会立刻暴露：针对电话音频调好的 VAD 可能被电视背景声误触发；轮次检测器可能一直等待根本不会出现的标点；TTS 可能先缓冲 400 毫秒才开始输出。本综合项目要求你在负载下逐一解决这些问题，并发布一份延迟与质量报告。

## 核心概念

整条管线分为五个流式阶段：**音频输入**（浏览器或 PSTN 通过 WebRTC 传入音频）、**ASR**（Deepgram Nova-3 或 faster-whisper 持续返回部分转写）、**轮次检测**（VAD 配合读取部分转写的小型轮次检测模型，判断用户是否已经说完）、**LLM**（一旦确认本轮发言结束，立即开始流式输出令牌）、**TTS**（在 LLM 生成第一个令牌后的约 200 毫秒内开始输出音频流）。

整条管线还要同时处理三个问题。**插话打断（Barge-in）**：智能体说话时，如果用户重新开口，必须立刻取消 TTS，并让 ASR 马上恢复接收。**工具使用**：对话中的函数调用（天气、日历等）必须通过旁路执行，不能阻塞音频；如果延迟超过 300 毫秒，智能体应先说一句“请稍等，我查一下”作为回应。**背压（Backpressure）**：出现丢包时，需要暂存部分转写，提高 VAD 的语音门限，并避免智能体对一条尚未确认送达的消息作答。

验收标准全部量化：在 15 dB 信噪比下，Hamming VAD 基准的 WER 低于 8%；100 通实测通话的首包音频输出 p50 低于 800 毫秒；误截断率低于 3%；TTS 的 MOS 高于 4.2；单台 g5.xlarge 支持 50 路并发通话。交付时要拿这些数字说话。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）配合 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式传输，首个部分转写延迟低于 300 毫秒）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 配合 LiveKit 轮次检测器（读取部分转写的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime（便于紧密集成）、Gemini 2.5 Flash Live，或级联式 Claude Haiku 4.5（流式补全，音频路径分离）
- TTS：Cartesia Sonic-2（首字节延迟最低）、ElevenLabs Flash v3，或用于自托管的开源 Orpheus
- 工具：天气、日历和预订工具通过 FastMCP 旁路调用；若工具耗时超过 300 毫秒，智能体会先说一句简短回应
- 可观测性：使用 OpenTelemetry 语音跨度，并在 Langfuse 中记录可回放音频的语音追踪
- 部署：单台 g5.xlarge（24 GB 显存）运行自托管的 Whisper 与 Orpheus；追求最低延迟时使用托管 API

```figure
ce-voice-latency
```

## 动手构建

1. **WebRTC 会话。** 创建一个 LiveKit 房间和一个能流式传输麦克风音频的 Web 客户端。在服务端挂接智能体工作进程，并让它加入房间。

2. **ASR 流式处理。** 将每帧 20 毫秒的 PCM 音频送入 Deepgram Nova-3（或运行在 GPU 上的 faster-whisper）。订阅部分转写和最终转写，并记录每次部分转写的延迟。

3. **VAD 与轮次检测器。** 在音频帧流上运行 Silero VAD v5。语音结束事件发生时，用最新的部分转写调用 LiveKit 轮次检测器。只有 VAD 判断静音已持续 500 毫秒，且轮次检测器给出的完成评分 > 0.6，才确认“本轮发言结束”。

4. **LLM 流。** 确认本轮发言结束后，将当前对话历史和最终转写一并送入 LLM，并流式输出令牌。第一个令牌生成后，立即交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 以流式方式返回音频块。第一个音频块必须在 LLM 生成首个令牌后的 200 毫秒内离开服务端。将音频块发送到 LiveKit 房间，由客户端通过 WebRTC 抖动缓冲区播放。

6. **插话打断。** TTS 播放期间，如果 VAD 检测到用户重新说话，立即取消 TTS 流、丢弃剩余的 LLM 输出，并让 ASR 重新进入监听状态。记录一个名为 `tts_canceled` 的追踪跨度。

7. **工具旁路。** 将天气和日历注册为函数调用工具。调用工具时并发执行；如果 300 毫秒内仍未返回，就让 LLM 先说一句“请稍等，我查一下”，工具返回后再继续回答。

8. **评测框架。** 录制 100 通电话。计算 WER（与留出的转写文本对照）、误截断率（用户还没说完时 TTS 就被取消）、首包音频输出 p50、TTS MOS（人工评分或 NISQA），并执行抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 使用模拟呼叫方，在单台 g5.xlarge 上发起 50 路并发通话，测量持续负载下的首包音频输出 p95。

## 实际使用

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付成果

交付物是 `outputs/skill-voice-agent.md`。输入一个应用领域（客户支持、日程安排或自助终端）后，它会启动 LiveKit 智能体，并调优 ASR/VAD/LLM/TTS 管线，使各项指标达到验收标准。评分标准如下：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 通录音通话的首包音频输出 p50 低于 800 毫秒 |
| 20 | 轮次交接质量 | Hamming VAD 基准上的误截断率低于 3% |
| 20 | 工具调用正确性 | 对话中途调用工具能返回正确信息且不阻塞音频 |
| 20 | 丢包条件下的可靠性 | 注入 3% 丢包后的 WER 与轮次交接稳定性 |
| 15 | 评测框架完整度 | 使用公开配置即可复现测量结果 |
| **100** | | |

## 练习

1. 把 Deepgram Nova-3 换成在 g5.xlarge 上运行的 faster-whisper v3 turbo。测量延迟与 WER 差距，并指出哪些环节最需要权衡 CPU 与 GPU。

2. 增加一项中断仲裁策略：用户在工具调用期间插话时，智能体该如何处理？比较三种方案：立即取消、等待工具完成后停止，以及将用户输入排入下一轮。

3. 对轮次检测器进行对抗测试：让用户在句子中间长时间停顿。调节 VAD 静音阈值和轮次检测器评分阈值，在延迟不超过 900 毫秒的前提下，将误截断率降到最低。

4. 通过 Twilio 将同一套智能体部署到 PSTN。比较 PSTN 与 WebRTC 的首包音频输出延迟，并解释抖动缓冲区和编解码器带来的差异。

5. 为非英语语种（日语、西班牙语）增加语音活动检测。测量 Silero VAD v5 的误触发率，并与针对特定语言微调的版本比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 轮次检测（Turn detection） | “话语结束” | 结合 VAD 检测到的静音和部分转写，判断用户是否已经说完的分类器 |
| 插话打断（Barge-in） | “中断处理” | VAD 检测到用户重新说话时，取消正在播放的 TTS |
| 首包音频输出（First-audio-out） | “延迟” | 从用户停止说话到第一个音频包离开服务端所经过的时间 |
| VAD | “语音门控” | 将音频帧分为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲区（Jitter buffer） | “音频平滑” | 客户端短暂缓存数据包，以吸收网络延迟波动的缓冲区 |
| 填充语（Filler） | “确认令牌” | 工具响应较慢时，智能体为避免沉默而先说出的简短语句 |
| MOS | “平均意见分” | 对感知语音质量的评分；NISQA 是它的自动化近似指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — WebRTC 智能体参考框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 另一套以 Python 为主的流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考接口
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI architecture](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai production stack](https://docs.vapi.ai) — 另一套生产级技术栈参考
