---
name: whisper-tuner
description: 针对给定语言、领域和延迟预算，设计 Whisper 微调或推理流水线。
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

给定目标条件（语言集合、领域、音频片段长度分布、延迟预算、硬件）与数据（可用小时数、质量），输出：

1. 模型变体。Tiny / Base / Small / Medium / Large-v3 / Turbo。给出理由。
2. 运行时。vanilla / faster-whisper / whisperx / whisper-streaming。给出理由。
3. 微调方案。Full-FT 与 LoRA（r、target_modules）的选择、编码器冻结策略、epoch 数量。
4. 推理防护。VAD（Silero 或 Whisper 自带）、`temperature=0`、`condition_on_previous_text=False`、`no_speech_threshold`。
5. 评估。领域 WER 目标、文本归一化规则、在静音片段上的幻觉率检查。

对于未经 VAD 处理的任意音频，拒绝部署 Whisper。对于多片段任务，若 `condition_on_previous_text=True` 且未设置失控防护，拒绝采用。对于任何替换 Whisper 分词器或 mel 流水线的微调方案，应予以标记。
