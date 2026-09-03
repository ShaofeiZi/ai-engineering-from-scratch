---
name: voice-agent
description: 构建实时语音智能体，实现首音频输出低于 800ms、打断处理及对话中工具调用。
version: 1.0.0
phase: 19
lesson: 03
tags: [capstone, voice, webrtc, livekit, pipecat, asr, tts, streaming]
---

给定一个领域（客户支持、日程安排、零售助手），部署一个 WebRTC 语音智能体，在处理打断、工具调用和丢包的同时，将端到端首音频输出保持在 800ms 以内。

构建计划：

1. 搭建一个 LiveKit Agents 1.0 房间，配备流式传输麦克风音频的 Web 客户端。添加 Twilio PSTN 网关以支持电话接入。
2. 运行流式 ASR（Deepgram Nova-3 托管版或 g5.xlarge 上的 faster-whisper Whisper-v3-turbo）。订阅部分转录和最终转录。
3. 在 20ms 帧上运行 Silero VAD v5。检测到语音结束时，用 LiveKit 轮次检测器对最新部分转录评分；仅当 VAD 静默 >= 500ms 且完成评分 >= 0.6 时才提交为轮次完成。
4. 流式传输 LLM（GPT-4o-realtime、Gemini 2.5 Flash Live 或级联 Claude Haiku 4.5）。在 200ms 内将首个 token 交给 TTS。
5. 流式传输 TTS（Cartesia Sonic-2 或 ElevenLabs Flash v3）。首个音频分块必须在首个 LLM token 后 200ms 内离开服务器。
6. 打断：当 VAD 在 SPEAKING 或 THINKING 期间检测到新的用户语音时，取消 TTS，丢弃剩余 LLM 输出，重新武装 ASR。发布一个 `tts_canceled` span。
7. 工具侧信道：并发运行函数调用；如果延迟 > 300ms，发送一个确认填充词，使音频流永不停滞。
8. 录制 100 通通话。针对留存转录文本测量 WER，在 Hamming VAD 基准上测量误截断率，首音频输出 p50，NISQA MOS，以及 3% 丢包下的行为表现。
9. 在单台 g5.xlarge 上用合成呼叫方负载测试 50 个并发通话；报告持续首音频输出 p95。

评估标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 通录制通话中首音频输出 p50 低于 800ms |
| 20 | 轮次交替质量 | Hamming VAD 基准上误截断率低于 3% |
| 20 | 工具调用正确性 | 对话中工具调用返回正确数据且音频不停滞 |
| 20 | 丢包下可靠性 | 注入 3% 丢包下的 WER 和轮次交替稳定性 |
| 15 | 评估框架完整性 | 可复现的测量及公开配置 |

硬性否决项：

- 非流式流水线（批量 ASR、批量 TTS）无法达到延迟目标。
- 任何不立即取消 TTS 缓冲区的打断策略。延迟取消会产生最差的用户体验回归。
- 同步阻塞 LLM 流的工具调用。它们必须在侧信道上运行。

拒绝规则：

- 拒绝在没有 VAD 或轮次检测器的情况下部署。固定超时轮次交替会产生不可接受的截断率。
- 拒绝在未记录是人工评分还是 NISQA 代理评分的情况下报告 MOS。
- 拒绝在未至少录制 100 通通话并公布通话轨迹的情况下报告"p50 延迟低于 X"。

输出：一个仓库，包含 LiveKit 智能体 worker、PSTN 网关配置、100 通通话评估框架、公开的 Langfuse 语音仪表板、与一个托管竞争对手（Retell、Vapi 或直接使用 OpenAI Realtime API）的并排对比，以及关于你观察到的三个最大轮次交替失败及各自修复检测器调优的书面报告。
