---
name: feature-extractor
description: 选择特征类型、mel 数量、帧长/帧移和归一化方式，使其与下游音频模型相匹配。
version: 1.0.0
phase: 6
lesson: 02
tags: [audio, features, spectrogram, mel]
---

给定目标模型（ASR / TTS / 分类器 / 说话人 / 音乐）以及输入音频（采样率、领域），输出：

1. 特征类型。Log-mel、mel、MFCC、原始波形，或离散编解码器（EnCodec、SoundStream）。给出一句理由。
2. mel 数量与频率范围。`n_mels`、`fmin`、`fmax`。理由应与领域（语音 vs 音乐）和模型目标相关联。
3. 帧长与帧移。`frame_len`、`hop_len`、窗类型。理由应与所需的时间分辨率相关联。
4. 归一化。按句子的均值/方差、全局统计量，或带固定参考的 dB；在特征提取之前或之后执行。
5. 验证片段。用 Python 在一段 1 秒的参考音频上打印结果的形状、最小/最大值、均值/标准差，并断言它们与训练时一致。

拒绝交付其帧长/帧移/mel 数量与目标模型已发布训练配置不一致的特征流水线。将任何基于 MFCC 的 Whisper 或 Parakeet 设置标记为错误——这些模型使用的是 log-mel。将任何不包含归一化断言的特征提取器也标记为错误。
