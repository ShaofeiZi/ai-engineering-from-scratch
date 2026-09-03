---
name: duplex-pipeline
description: 针对语音智能体工作负载，在全双工（Moshi）与流水线（VAD + STT + LLM + TTS）架构之间做出选型。
version: 1.0.0
phase: 6
lesson: 15
tags: [moshi, hibiki, full-duplex, voice-agent, streaming]
---

根据工作负载（延迟目标、工具调用需求、语言覆盖范围、硬件预算、云端还是边缘），输出：

1. 架构。全双工（Moshi / GPT-4o Realtime / Gemini Live）对比流水线（LiveKit + STT + LLM + TTS，第 12 课）。用一句话说明理由。
2. 模型。Moshi · Hibiki · Hibiki-Zero · Sesame CSM · GPT-4o Realtime · Gemini 2.5 Live · 传统流水线。给出理由。
3. 规模。单会话 GPU 成本（Moshi 会持续占用一个槽位）、最大并发会话数、冷启动的影响。
4. 工具调用路径。若有需要——混合流水线（全双工 + 外部 LLM 处理工具调用）或纯流水线。说明权衡。
5. 语言覆盖。全双工模型支持的语言较窄；流水线则继承 LLM 的多语言能力。

对于需要工具调用 / 检索的企业级智能体，拒绝纯全双工架构——Moshi 是对话模型，不是智能体框架。对于低于 250 ms 的对话式智能体，拒绝纯流水线——各阶段延迟会叠加。对于单 GPU 上超过 4 个并发会话的场景，拒绝使用 Moshi——会遭遇资源争用。

示例输入："语言学习的语音伴侣——用于对话流利度练习。英语 + 法语。响应延迟 < 250 ms。日活 1 万。"

示例输出：
- 架构：全双工（Moshi）。低于 250 ms 的延迟要求 + 对话流利度契合 Moshi 的优势。
- 模型：Moshi。英语和法语均良好支持。CC-BY 4.0 许可证。
- 规模：每 4-6 个并发会话使用一块 L4 GPU → 在 10% 并发率下，1 万日活峰值约需 1500 块 GPU。规划使用 Kyutai Pocket TTS + 本地 Whisper 作为静音路径的设备端轻量模式。
- 工具调用：极少——"显示语法提示"和"翻译这句话"可通过一个轻量 LLM 旁路处理；大部分交互是开放式对话，正是 Moshi 的强项。
- 语言覆盖：英语 + 法语（原生）；西班牙语 / 德语 / 日语通过 Hibiki-Zero 适配实现（每种新语言需要约 1000 小时音频）。
