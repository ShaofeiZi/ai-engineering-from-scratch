---
name: voice-pipeline
description: 搭建一个 Pipecat 风格的语音管线（VAD + STT + LLM + TTS + transport），支持抢话、置信度门控以及延迟预算强制执行。
version: 1.0.0
phase: 14
lesson: 22
tags: [voice, pipecat, livekit, webrtc, latency]
---

给定一个语音产品规格说明（语言、transport、provider），搭建一个基于帧的管线。

产出：

1. `Frame` 类型，包含 `kind`、`payload`、`direction`（downstream / upstream）。
2. 处理器：`VAD`、`STT`、`LLM`、`TTS`、`Transport`。每个都实现 `process(frame)`。
3. `link()` 辅助函数，将处理器向前和向后串联。
4. 取消帧处理：UPSTREAM 路径从 transport 到 TTS 到 LLM 到 STT，在每个阶段丢弃待处理的工作。
5. 观察者：逐阶段延迟指标；每当一个帧穿过处理器时发射一个 OTel span（第 23 课）。
6. STT 上的置信度门控：低于阈值时，发射一个 "please repeat" 文本帧而非转写结果。

硬性拒绝：

- 缺少 UPSTREAM 处理的管线。抢话对于语音来说不是可选项。
- 不使用流式的 LLM 调用。首 token 延迟占主导地位；必须流式传输。
- 不考虑置信度的 STT。将错误的转写结果喂给 LLM 会产生错误的回复。

拒绝规则：

- 如果端到端延迟在冷启动时超过 1500ms，拒绝交付。优化链路或使用 MultimodalAgent（LiveKit direct-audio）。
- 如果产品以电话为先且管线没有 SIP 适配器，拒绝。通过 LiveKit SIP 或平台（Vapi/Retell）路由。
- 如果产品在传输中携带 PII 音频但未加密，拒绝。

输出：`frames.py`、`processors.py`、`pipeline.py`、`observers.py`、`README.md`，后者需解释延迟预算、抢话设计和 transport 选择。以"下一步阅读"结尾，指向第 23 课（OTel）、第 24 课（可观测性后端），或 LiveKit 文档以了解 WebRTC 细节。
