---
name: tts-designer
description: 针对给定的语言、风格和延迟目标，选择 TTS 模型、语音、文本归一化范围和评测方案。
version: 1.0.0
phase: 6
lesson: 07
tags: [audio, tts, speech-synthesis]
---

给定目标（语言、语音风格、延迟预算、CPU 还是 GPU、许可约束）和内容（领域、OOV 密度、标点丰富度），输出：

1. 模型。Kokoro / XTTS v2 / F5-TTS / VITS / StyleTTS 2 / 商业 API。给出一句理由。
2. 文本前端。归一化范围（数字、日期、URL），音素化器（espeak-ng 还是 g2p-en），OOV 回退方案。
3. 语音。预设名称或参考音频片段规格（秒数、本底噪声、口音匹配）。
4. 质量目标。目标 UTMOS、通过 Whisper 测得的 CER、克隆时的 SECS。
5. 评测方案。包含 20 条话语的测试集，覆盖数字、同形异义词、专有名词和长句。

拒绝任何没有文本归一器的生产级 TTS。拒绝未经用户同意且未加水印的语音克隆。对任何被要求说英语以外语言的 Kokoro 部署提出警示。
