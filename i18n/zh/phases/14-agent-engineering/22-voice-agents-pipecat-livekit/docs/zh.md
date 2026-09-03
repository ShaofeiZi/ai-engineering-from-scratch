# 语音代理：Pipecat 与 LiveKit

> 到了 2026 年，语音代理已经是正式的一线生产形态。Pipecat 提供基于 Python 的 frame 管线（VAD → STT → LLM → TTS → transport），LiveKit Agents 则通过 WebRTC 把 AI 模型接到真实用户面前。高端语音栈的端到端延迟目标，通常落在 450–600ms。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 01（Agent Loop），第 14 阶段 · 12（工作流模式）
**Time:** 约 60 分钟

## 学习目标

- 解释 Pipecat 的 frame 管线模型：DOWNSTREAM（source→sink）与 UPSTREAM（control）。
- 说出标准语音管线的主要阶段，以及 Pipecat 支持哪些 transport。
- 解释 LiveKit Agents 的两类语音代理：MultimodalAgent 与 VoicePipelineAgent，并说明各自适用场景。
- 概括 2026 年生产环境常见的延迟预期，以及这些延迟预算如何反向约束架构设计。

## 问题

语音代理不是在文本循环外面简单套一层 TTS。它的延迟预算非常苛刻，大致只有 ~600ms；部分音频输出是默认能力；turn detection 本身也需要模型参与；传输层从电话 SIP 到 WebRTC 都得支持。你通常只有两条路线：要么自己搭一条基于 frame 的实时语音管线，例如 Pipecat；要么基于现成平台，如 LiveKit。

## 概念

### Pipecat（仓库：pipecat-ai/pipecat）

- 一个基于 Python 的 frame 化管线框架。
- 以 `Frame` → `FrameProcessor` 的链式结构组织处理流程。
- 数据流有两个方向：
  - **DOWNSTREAM**：source → sink，也就是音频进入、语音输出的正向路径。
  - **UPSTREAM**：反馈与控制路径，例如取消、指标回传、barge-in。
- `PipelineTask` 负责整个生命周期，带有事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）以及用于指标、tracing、RTVI 的 observers。

典型管线如下：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

支持的 transport 包括：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 在此基础上增加了结构化会话能力，本质上是状态机；Pipecat Cloud 则是它的托管运行时。

### LiveKit Agents（仓库：livekit/agents）

- 负责通过 WebRTC 把 AI 模型连接到最终用户。
- 核心概念包括：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类主要语音代理：
  - **MultimodalAgent**：直接走音频输入和音频输出，例如 OpenAI Realtime 或同类实时模型。
  - **VoicePipelineAgent**：采用 STT → LLM → TTS 级联，因此可以在文本层面做更细粒度控制。
- 支持基于 transformer 的语义 turn detection。
- 原生集成 MCP。
- 支持通过 SIP 接入电话系统。
- 可通过 LiveKit Inference 使用 50+ 无需 API key 的模型，也可通过插件接入额外 200+ 模型。

### 商业平台

Vapi 与 Retell 这类商业产品通常就是在上述基础设施之上再做托管封装。Vapi 在优化过的高端栈上可做到约 450–600ms；Retell 在 180 次测试通话中的端到端延迟大约是 600ms。如果你不想自己养一支 WebRTC 团队，直接选托管平台往往更现实。

### 这种模式常见的失败点

- **没有处理 barge-in。** 用户已经打断，代理还在继续说。Pipecat 里需要依赖 UPSTREAM cancel frame，LiveKit 里也需要对应机制。
- **忽略 STT confidence。** 低置信度转写被当成确定事实直接送给 LLM，应该基于置信度做拦截或请求用户确认。
- **TTS 在句中被截断。** 如果管线中途取消了一段正在播报的 utterance，TTS 侧必须明确感知到取消或能主动截音。
- **完全没算延迟预算。** 每个组件都可能额外增加 50–200ms，上线前必须把整条链路加总一遍。

### 2026 年的典型延迟

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首个词元：150–400ms
- TTS 首段音频：100–200ms
- 传输往返时间（RTT）：30–80ms

450–600ms 的端到端体验算高端。800–1200ms 很常见。任何超过 1500ms 的系统，用户都会明显感觉它坏了。

```figure
voice-pipeline
```

## 动手构建

`code/main.py` 实现了一个基于 frame 的玩具管线，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，定义 `process(frame)`。
- 一个五阶段管线（VAD → STT → LLM → TTS → transport），由脚本化 processor 构成。
- 一个用于演示 barge-in 的 UPSTREAM cancel frame。

运行方式：

```
python3 code/main.py
```

输出 trace 会展示正常流动，以及一次 barge-in cancel 如何在播报中途打断 TTS。

## 如何使用

- **Pipecat**：适合需要完全控制权的场景，比如自定义 processor、Python 优先开发、自由切换 provider。
- **LiveKit Agents**：适合 WebRTC 优先部署，以及电话语音接入。
- **Vapi / Retell**：适合没有 WebRTC 工程团队、但又要尽快落地托管语音代理的团队。
- **OpenAI Realtime / Gemini Live**：适合直接做 audio-in/audio-out 的实时语音代理，也就是 MultimodalAgent 一类方案。

## 交付成果

`outputs/skill-voice-pipeline.md` 提供一个 Pipecat 风格的语音管线脚手架，包含 VAD + STT + LLM + TTS + transport，以及 barge-in 处理。

## 练习

1. 给这个玩具管线加一个 metrics observer：统计每秒每个阶段处理了多少 frame。延迟主要积累在哪一段？
2. 实现带置信度门槛的 STT：低于阈值时，不进入 LLM，而是请求用户“could you repeat that?”
3. 增加语义 turn detection：先用一个最简单的规则，如果 transcript 以 ? 结尾，就视为轮次结束。
4. 阅读 Pipecat 的 transport 文档。把当前 stdlib transport 替换成 SmallWebRTCTransport 的配置桩（stub）。
5. 对同一个问题，分别测量 OpenAI Realtime 与 STT+LLM+TTS 级联系统的延迟。文本层控制到底带来了多少延迟成本？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Frame | "Event" | 管线中的一个带类型数据单元，例如 audio、transcript、text、control |
| Processor | "Pipeline stage" | 一个实现了 process(frame) 的处理节点 |
| DOWNSTREAM | "Forward flow" | 从 source 到 sink 的正向路径：音频输入，语音输出 |
| UPSTREAM | "Feedback flow" | 控制路径：cancel、metrics、barge-in |
| VAD | "语音活动检测" | 检测用户是否正在说话 |
| 语义轮次检测 | "智能轮次结束判断" | 通过模型判断用户这一轮是否已经说完 |
| MultimodalAgent | "直接音频智能体" | 中间不经过文本，直接 audio in / audio out |
| VoicePipelineAgent | "Cascade agent" | 通过 STT + LLM + TTS 级联，保留文本层控制能力 |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline、processors、transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + 语音基础能力
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音平台，并公开过延迟基准
