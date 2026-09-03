---
name: seq2seq-picker
description: 为新的序列到序列任务选择编码器-解码器还是仅解码器架构。
version: 1.0.0
phase: 7
lesson: 8
tags: [transformers, t5, bart, seq2seq]
---

给定一个 seq2seq 任务（翻译 / 摘要 / 语音转文本 / 结构化抽取 / 改写）、输入与输出的长度分布，以及质量与延迟的优先级，输出：

1. 架构。下列之一：编码器-解码器（T5 / BART / Whisper 风格）、经过指令微调的仅解码器、仅编码器 + 提示模板。给出一句话理由。
2. 预训练目标。Span corruption（T5）、denoising（BART）、next-token（仅解码器），或"跳过预训练，直接微调已有检查点"。指出该检查点名称。
3. 输入格式。任务前缀字符串（T5 风格）vs 系统提示（仅解码器）vs 原始 token（BART）。包括 BOS/EOS 的处理方式。
4. 解码策略。用于翻译/摘要的束搜索宽度与长度惩罚，或用于类聊天任务的 nucleus/min-p。针对该任务说明选用哪种。
5. 评估。与任务匹配的指标：BLEU / ROUGE / WER / F1 / exact match。给出测试集大小。

对于生成式输出，不得推荐仅编码器架构。当输入本身就是对话时，不得推荐编码器-解码器架构——仅解码器天然契合对话记忆。若在语音转文本任务中选择仅解码器而未提及 Whisper 作为需超越的基线，应予以标记。
