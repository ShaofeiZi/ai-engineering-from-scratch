---
name: alm-picker
description: 为音频理解任务挑选音频语言模型、基准测试子集、输出模态（文本还是语音）以及护栏。
version: 1.0.0
phase: 6
lesson: 10
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

给定任务（语音 / 声音 / 音乐 / 多音频 / 长音频，输出模态，延迟，许可证），输出：

1. 模型。Qwen2.5-Omni-7B · Qwen3-Omni · SALMONN · Audio Flamingo 3 · AF-Next · LTU · GAMA · Gemini 2.5 Pro (API) · GPT-4o Audio (API)。给出一句理由。
2. 验证用的基准子集。MMAU-Pro 语音 / 声音 / 音乐 / 多音频 · LongAudioBench · AudioCaps · ClothoAQA。选择与用户任务匹配的维度。
3. 输出模态。仅文本 · 文本 + 语音（Qwen-Omni、GPT-4o Audio）。若需要，为额外的语音解码器预留预算。
4. 护栏。当你的模型多音频得分 &lt; 30%（接近随机）时，拒绝需要多音频比较的提示。对 &gt; 10 分钟的输入，在送入 LALM 之前先做说话人分离。
5. 升级。何时应将任务回退到专用模型——Whisper 用于转写、BEATs 用于分类、pyannote 用于说话人分离。LALM 并非在每个单项上都是最优。

在未验证你的模型在 MMAU-Pro 多音频子集上得分 &gt; 40% 的情况下，拒绝发布多音频比较任务。拒绝在无上游说话人分离的情况下处理长音频（&gt; 10 分钟）。对任何使用厂商报告数值而未经独立复验的部署，予以标记。

示例输入："合规审计：转写 10 分钟银行通话录音 + 检测坐席是否朗读了强制性披露内容。"

示例输出：
- 模型：Whisper-large-v3-turbo 用于转写 + Gemini 2.5 Pro（通过 API）对转写文本做披露检查问答。直接在原始音频上用 LALM 很诱人，但长音频 LALM 在超过 10 分钟后准确率会下降。
- 基准子集：MMAU-Pro 语音子集（Gemini 2.5 Pro = 73.4%）——覆盖语音推理维度。同时在自有的 50 通通话黄金集上做抽查。
- 输出模态：仅文本。审计报告不需要语音输出。
- 护栏：先用 pyannote 3.1 做说话人分离；按说话人分别发送片段；逐通记录置信度分数。
- 升级：若某通通话未通过披露检查，转人工复核，而非自动标记。
