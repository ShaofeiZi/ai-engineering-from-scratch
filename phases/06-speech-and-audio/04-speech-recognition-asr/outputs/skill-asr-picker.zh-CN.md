---
name: asr-picker
description: 针对给定部署目标，挑选 ASR 模型、解码策略、分块方式以及语言模型融合方案。
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech-recognition]
---

给定一个部署目标（语言列表、领域、延迟预算、硬件、离线/流式、片段时长），输出：

1. 模型。Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine。用一句话说明理由。
2. 解码。贪心解码 / 束宽 / 温度回退 / 语言模型融合权重。理由需与质量预算挂钩。
3. 分块与 VAD。块长度、步长、是否使用 Silero-VAD 或 Whisper 自带 VAD 进行门控。
4. 语言策略。强制指定语言还是自动语种识别（auto-LID）；如何处理跨语种帧。
5. 评测方案。在领域测试集上的 WER、分说话人覆盖率、静音片段上的幻觉率。

对于任何未启用 VAD 门控的长格式 Whisper 部署方案，应予以拒绝（在静音上容易出现幻觉）。若未进行文本归一化（小写化、去除标点），不得报告 WER。若未使用语言模型却设置束宽 > 16，应予以标记；在空白上的原始束搜并无帮助。
