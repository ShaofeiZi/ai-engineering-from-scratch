---
name: any-to-any-pipeline-auditor
description: 审计对话式 any-to-any 设计，并为 MIO / AnyGPT / Moshi 系列技术栈计算延迟预算。
version: 1.0.0
phase: 12
lesson: 16
tags: [mio, anygpt, moshi, any-to-any, streaming, ttfab]
---

给定一个对话式产品（语音输入 / 语音输出，可选视觉，可选音乐）、一个模型规模和一个目标延迟，审计 any-to-any 设计并产出可行的配置。

产出：

1. 模态组合。哪些模态输入，哪些输出。选择系列：MIO / AnyGPT（离散 token，4 模态）、Moshi（语音+文本为主，内心独白）、Unified-IO 2（视觉丰富）。
2. 共享词表规划。文本 + 图像 + 语音 + 音乐 + 分隔符的 ID 范围。总规模通常为 40-50k。
3. 分词器栈。BPE + SEED + SpeechTokenizer-RVQ + Encodec。标出哪些仍是瓶颈（通常是语音质量）。
4. 训练课程。四阶段 MIO 方案，或面向以语音为核心的 Moshi 的两阶段方案。
5. TTFAB 延迟预算。麦克风编码器 + 预填充 + 首个 token + 残差解码 + 语音解码器。与约 500ms 的对话门槛比较。
6. 质量与延迟的帕累托权衡。低延迟用更小模型，更高质量用更大模型；给出 A100/H100 上的粗略数值。

硬性拒绝：
- 当需求是对话流畅性时，提出每个模态使用独立模型。流水线延迟会叠加，体验更差。
- 使用仅 1 层 codebook 的语音分词器。任何生产级语音都会听起来很机械。
- 声称 MIO 的 TTFAB 可媲美 GPT-4o。目前还不能；Moshi 的 160ms 是最接近的开放数据。

拒绝规则：
- 如果目标 TTFAB <200ms，拒绝 MIO 级（8B+）并推荐 Moshi 级（7B，针对语音调优）或更小的语音专用模型。
- 如果用户需要录音棚级语音输出，拒绝开放残差 VQ 并推荐 ElevenLabs / 链式 TTS，直到开放质量跟上（Qwen3-Omni / Moshi2）。
- 如果用户希望在语音通话期间进行图像生成，拒绝以流式语音优先的方案，并提出带模式切换的拆分流水线。

输出：一页审计报告，包含模态组合、词表规划、分词器栈、训练课程、TTFAB 延迟、质量-延迟帕累托。末尾附 arXiv 2409.17692（MIO）、2410.00037（Moshi）、2402.12226（AnyGPT）。
