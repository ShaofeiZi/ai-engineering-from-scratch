---
name: voice-assistant-architect
description: 针对给定工作负载，产出全栈语音助手规格说明——涵盖组件、延迟预算、可观测性、合规性。
version: 1.0.0
phase: 6
lesson: 12
tags: [voice-assistant, architecture, livekit, pipecat, compliance]
---

给定用例（消费级 / 客服 / 无障碍 / 边缘）、预期规模（并发会话数、每月分钟数）、语言、延迟目标、合规要求（HIPAA、PCI、欧盟 AI 法案、加州 SB 942），输出：

1. 组件（7 层）。麦克风 + 分块 · VAD · 流式 STT · LLM + 工具 · 流式 TTS · 播放 · 中断处理。为每一层指明确切的供应商/模型。
2. 延迟预算。每个阶段的 P50 / P95 / P99 目标，相加得到端到端目标。标注哪些阶段是并行的、哪些是串行的。
3. 工具调用 schema。每个工具的 JSON 规范 + 错误处理 + 兜底文案。必须始终包含一条"无法提供帮助"的路径，当 LLM 失败两次时必须走该路径。
4. 安全。提示注入防护、语音克隆锁定（若 TTS 具备克隆能力）、唤醒词门控（针对常驻设备）、日志中的 PII 脱敏、30 天留存。
5. 可观测性。各阶段 P50/P95/P99 · 误中断率 · 工具调用成功率 · 每 100 次通话的 WER · 每分钟成本 · 放弃率。
6. 合规。披露音频（"这是一位 AI 助手"）、区域固定（欧盟数据留在欧盟）、审计日志留存、退出路径。

拒绝没有唤醒词的常驻部署。拒绝不流式输出的 TTS（会增加整句长度的延迟）。拒绝不带 P95 的平均延迟——尾部才是用户流失的地方。拒绝超过 30 天的原始音频留存且未经法务审查。

示例输入："面向低视力用户的无障碍助手：纯语音界面操作消费级邮件应用。英语。P95 < 600 ms。约 1 万并发用户。"

示例输出：
- 组件：sounddevice（通过 LiveKit Agents 的 WebRTC）· Silero VAD · Deepgram Nova-3（英语）· GPT-4o 配邮件工具（read_message、compose_reply、mark_read）· Cartesia Sonic 2 流式 · WebRTC 输出 · 中断=VAD 触发时取消 LLM 与 TTS。
- 预算：采集 120 ms + VAD 40 + STT 150 + LLM TTFT 100 + TTS TTFA 150 = 560 ms P95。
- 工具：read_message({id})、compose_reply({message_id, body})、mark_read({id})、search({query})。均返回 JSON；LLM 每个工具最多重试 2 次，随后兜底"我没能完成这个操作——请换种说法"。
- 安全：提示注入防护（检测 `ignore previous instructions`）；唤醒词"Hey Mail"；无语音克隆（固定 Cartesia 音色）；日志中脱敏邮件正文。
- 可观测性：Hamming AI 生产监控；各阶段 Prometheus 直方图；在误中断 > 5% 或 P95 > 800 ms 时告警。
- 合规：首次使用时进行 AI 披露；仅针对医疗邮件启用 HIPAA 选项；欧盟用户访问欧盟托管的 Cartesia + GPT-4o 爱尔兰。
