---
name: audio-brief
description: 将音频需求简报转化为涵盖 TTS、音乐和 SFX 的模型 + 提示词 + 评测方案。
version: 1.0.0
phase: 8
lesson: 11
tags: [audio, tts, music, sfx, codec]
---

给定一个音频简报(任务:TTS / 音乐 / SFX / 语音克隆,时长,风格,嗓音或流派,许可约束,实时还是离线,质量门槛),输出:

1. 模型 + 托管。ElevenLabs V3、OpenAI TTS、XTTS v2、Suno v4、Udio、Stable Audio 2.5、MusicGen 3.3B、AudioCraft 2 或 GPT-4o realtime。一句理由。
2. 提示词格式。TTS:文本 + 语音提示(3-10 秒样本或 voice ID)+ 情绪 / 节奏标签。音乐:流派 + 配器 + 情绪 + BPM + 结构标记。SFX:拟声词 + 声源 + 时长提示。
3. 编解码器 + 生成器 + 声码器链。指明具体编解码器(Encodec 32 kHz、DAC 44 kHz、自定义)和生成器选择(token-AR 还是 flow-matching)。
4. 种子 + 可复现性。种子锁定、版本锁定、提示词哈希。
5. 评估。TTS 用 MOS(平均主观评分)或 A/B,音乐用 CLAP 得分,TTS 转录用 CER,SFX 用用户试听测试。
6. 护栏。语音克隆需同意 + 水印(PerTh / SynthID-audio)、音乐输出的版权扫描、训练数据策略检查。

拒绝在没有所有者可验证同意的情况下克隆任何语音(磁带时代的 "3 秒提示" 不算同意)。拒绝交付含未授权参考素材的音乐。将任何 &lt; 200 ms 的实时目标且未使用流式 token-AR 模型的情况标记出来——基于扩散的音频在 2026 年无法满足低于 300 ms 的 TTFB。
