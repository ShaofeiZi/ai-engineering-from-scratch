---
name: realtime-voice-pipeline
description: 针对目标端到端延迟，选择传输层、VAD、流式 STT、LLM、流式 TTS 及编排框架。
version: 1.0.0
phase: 6
lesson: 11
tags: [voice-agent, livekit, pipecat, silero, streaming, latency]
---

给定目标（延迟 P50/P95、语言、通道、离线 vs 云端、通话量），输出：

1. 传输层。WebRTC（LiveKit / Daily）· WebSocket · SIP 中继（Twilio / Telnyx）。理由需关联抖动容忍度与使用场景。
2. VAD 与轮次切换。Silero VAD（开源，99.5% TPR）· Cobra（商用）· LiveKit turn-detector。阈值、最小语音时长、静音悬挂时间。
3. 流式 STT。Parakeet TDT（最快的开源方案）· Kyutai STT（配合 flush 技巧）· Deepgram Nova-3（API，约 150 ms）· Whisper-streaming。给出理由。
4. LLM 与流式。在 TTS 启动前锁定前 20 个 token。模型 + 流式配置 + 针对提示注入的防护栏。
5. 流式 TTS。Kokoro-82M（约 100 ms TTFA）· Orpheus · Cartesia Sonic · ElevenLabs Turbo。音色包或克隆防护（第 8 课）。
6. 编排。LiveKit Agents · Pipecat · Vapi · Retell · 自研 Rust。理由需关联团队技能与规模。
7. 可观测性。按阶段的 P50/P95/P99 直方图；误打断率；掉话率；通话样本 WER。

拒绝在 STT 前缓存整段话语的部署方案。拒绝不流式的 TTS。拒绝以平均延迟评估——必须使用 P95。对于 > 100k 分钟/月的场景，拒绝在未与自建方案做成本对比的情况下使用托管平台（Vapi / Retell）。

示例输入："车险报价语音助手。< 500 ms P95。美式英语。5 万分钟/周。合规要求：HIPAA 相邻（日志中不含 PII）。"

示例输出：
- 传输层：LiveKit Agents + Twilio SIP。已在呼叫中心规模验证，支持 HIPAA 模式开启。
- VAD：Silero VAD，阈值 0.45，最小语音时长 220 ms，静音悬挂 400 ms。叠加 LiveKit turn-detector。
- STT：Deepgram Nova-3 英语（约 150 ms P95）；若需本地审计则回退至 Parakeet-TDT。
- LLM：通过 OpenAI realtime API 流式调用 GPT-4o；以后置过滤器防护提示注入；锁定前 20 个 token 给 TTS。
- TTS：Cartesia Sonic 2（约 150 ms TTFA，不使用语音克隆——使用预设音色）。
- 编排：LiveKit Agents。生产环境通过 Hamming AI 进行可观测性监控。
- 日志：在持久化前通过正则 + NER 过程脱敏 CVV / SSN / DOB。保留 30 天。
