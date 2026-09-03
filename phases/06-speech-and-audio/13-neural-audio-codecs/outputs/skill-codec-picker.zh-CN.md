---
name: codec-picker
description: 针对给定的生成式或压缩任务，选择一款神经音频编解码器（EnCodec / DAC / SNAC / Mimi）。
version: 1.0.0
phase: 6
lesson: 13
tags: [codec, encodec, dac, snac, mimi, rvq, semantic-tokens]
---

给定任务（生成式 LM、压缩、全双工对话、音乐编辑、保真度目标），输出：

1. 编解码器。EnCodec-24k · EnCodec-48k · DAC-44.1k · SNAC-24k · Mimi ·（兜底：非神经压缩使用 Opus）。一句话说明理由。
2. 帧率 + 码本。码率预算、码本数量（通常 4–12）、目标片段时长的序列长度。
3. 分词方案。扁平（flat）vs 层级式（SNAC）vs 语义+声学（Mimi）。LM 如何消费 token。
4. 解码器。编解码器内置解码器 · 外部声码器（HiFi-GAN）· 纯 LM（无声码器，直接预测 codec token）。说明原因。
5. 训练影响。需要训练编码器/解码器吗？是否在领域音频上微调（纯语音 → 领域专属音乐）？冻结现成模型？

拒绝在严格延迟预算下的 AR-LM 工作负载中使用 DAC —— 86 Hz 帧率 × 8 码本 = 每 10 s 5,504 个 token，对快速生成来说序列过长。拒绝在音乐场景使用 Mimi —— 它是针对语音调优的。拒绝在语义条件生成中使用 EnCodec —— 没有语义码本，从文本生成的语音模糊不清。

输入示例："为文本转语音 TTS 构建一个 AR LM。目标 TTFA 200 ms。仅英语。"

输出示例：
- 编解码器：Mimi。语义+声学分离支持 text → codebook 0 → codebooks 1–7 的因式分解，既快速又支持声音克隆。
- 帧率 + 码本：12.5 Hz · 8 码本 · 4.4 kbps。10 s = 1,000 个 token。
- 分词：先根据 text + 说话人参考预测 codebook 0；再在给定 codebook 0 + 说话人参考的条件下预测 codebooks 1–7（depth-transformer 模式）。
- 解码器：Mimi 内置解码器，无需外部声码器。
- 训练：训练 text-to-codec LM；冻结 Mimi。
