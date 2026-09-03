---
name: asr-configurator
description: 为新的语音处理管线挑选 ASR 模型（Whisper 变体 / Moonshine / faster-whisper）及解码参数。
version: 1.0.0
phase: 7
lesson: 10
tags: [transformers, whisper, asr, speech]
---

给定一项语音任务（转写 / 翻译 / 流式 / 端侧）、语种、音频特征（噪声、口音、时长）以及延迟/质量目标，输出：

1. 模型选择。从以下中选一：faster-whisper large-v3-turbo（生产默认）、whisper large-v3（最高质量、多语言）、whisper medium（中端）、Moonshine base（边缘端）、distil-whisper（英文快 2 倍）。给出一句话理由。
2. 量化。int8_float16（CPU 默认）、float16（GPU 默认）、fp32（研究用）。标注显存影响。
3. 解码。束宽（典型值 5，流式取 1）、温度回退调度、对数概率阈值、无语音阈值、VAD 门控开/关。
4. 分块。30 秒固定窗口对比流式分块（通常 10 秒、2 秒重叠）+ 基于 VAD 的分段。说明重叠后的合并策略。
5. 后处理。时间戳对齐（WhisperX 强制对齐）、标点恢复、说话人分离（pyannote）。标注任务必需的项。

拒绝在生产中推荐原版 OpenAI Whisper（参考实现）—— `faster-whisper` 速度快 4 倍且输出完全一致。拒绝在无 VAD 的情况下交付流式 ASR，除非有书面说明的理由。当输入可能为多人说话时，标注任何单说话人假设。
