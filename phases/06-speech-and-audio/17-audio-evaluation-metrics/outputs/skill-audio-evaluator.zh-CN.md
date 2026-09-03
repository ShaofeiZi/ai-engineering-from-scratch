---
name: audio-evaluator
description: 为任何音频模型发布挑选指标、基准、归一化规则与报告格式。
version: 1.0.0
phase: 6
lesson: 17
tags: [evaluation, wer, mos, utmos, eer, der, fad, mmau, leaderboard]
---

给定任务（ASR / TTS / 语音克隆 / 说话人验证 / 说话人分离 / 分类 / 音乐 / LALM / 流式 S2S），输出：

1. 主指标。WER · MOS · UTMOS · SECS · EER · DER · mAP · FAD · MMAU-Pro 准确率 · 延迟 P95。任选其一。
2. 辅助指标。1-3 个附加维度（速度、多样性、鲁棒性）并说明理由。
3. 归一化规则。小写化、去除标点、数字展开、空白折叠。使用 Whisper-normalizer 或自定义方案，并记录说明。
4. 公开基准。用于对比报告的权威排行榜（Open ASR、TTS Arena、MMAU-Pro、VoxCeleb1-O、AudioSet、LongAudioBench 等）。
5. 内部评测集。留存的领域数据，N 个样本；含人口统计/声学子维度拆分。
6. 报告格式。分布（延迟的 P50/P95/P99；分类的每类召回率；MMAU 的每类别表现）。发布说明模板。

拒绝针对延迟的单数值评估（应报告百分位）。拒绝分类任务只报告聚合值（应报告每类结果）。拒绝仅提供 MOS/UTMOS 而无 SECS 的语音克隆 TTS 发布。拒绝没有 WER 归一化规范的 ASR 发布。拒绝仅提供 FAD 的音乐发布——必须同时配以人工 MOS 评测组。

示例输入："发布一款新的英西对话式 TTS。需要说服团队它优于现有的 Cartesia-Sonic 基线。"

示例输出：
- 主指标：UTMOS（每种语言 50 条提示的配对音频样本）+ 人工评测组 MOS（每种语言 20 位听者，相对基线的盲测 A/B）。
- 辅助指标：TTFA 中位数与 P95（须与基线持平）；相对固定声音参考的 SECS &gt; 0.80（不得出现说话人退化）；往返 ASR（Whisper-large-v3-turbo）的 CER &lt; 2%。
- 归一化：往返 WER 使用 Whisper-normalizer（英语）+ Hugging Face multilingual-normalizer（西班牙语）。
- 公开基准：TTS Arena（英语）与 Artificial Analysis Speech 用于相对 ELO 定位。目标：与最近竞争对手的差距在 50 ELO 以内。
- 内部评测集：200 条留存提示（每种语言 100 条），覆盖金额、日期、产品名、两句叙述、情感朗读、语码转换。10 种人口统计语音。
- 报告：发布说明含标题（UTMOS + MOS）、P50/P95 TTFA 直方图、SECS CDF、每类别 CER 拆分、失败模式标注（语码转换提示在 X% 处失败）。
