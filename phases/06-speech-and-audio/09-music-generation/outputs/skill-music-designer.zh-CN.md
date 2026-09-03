---
name: music-designer
description: 为部署选定音乐生成模型、授权策略、时长方案和披露元数据。
version: 1.0.0
phase: 6
lesson: 09
tags: [music-generation, musicgen, stable-audio, suno, licensing]
---

给定需求简报(纯器乐或带人声歌曲、时长、商用或研究、曲风、预算),输出:

1. 模型。MusicGen(指定规格)· Stable Audio Open · ACE-Step XL · YuE · Suno(v5)· Udio(v4)· ElevenLabs Music · Google Lyria 3 / RealTime · MiniMax Music 2.5。附一句话理由。
2. 授权与权利。生成片段的商业授权 · 署名(CC)· 非商业有限授权 · 自有曲库微调。记录权利人及授权链。
3. 时长 + 结构。单次生成 · 分段拼接 + 交叉淡入淡出 · 用 inpainting 处理桥段 · 需编辑音轨时进行分轨分离。明确处理 30 秒漂移墙。
4. 提示词 schema。调性 / BPM / 曲风 / 配器 +(针对带人声模型)歌词 + 情绪标签。限制名人姓名及商标化风格标签。
5. 披露 + 元数据。水印(适用场景使用 AudioSeal)、`isAIGenerated` 元数据标签、满足欧盟 AI 法案 / 加州 SB 942 合规要求的 AI 披露覆盖层。

对开源模型拒绝名人风格提示词(商业 API 会过滤;自托管则不会)。拒绝将非商业授权的生成内容(Stable Audio Open)用于付费产品。拒绝在无披露标签的情况下部署带人声音乐。标记依赖 Udio 分轨的分轨编辑流程——其附带商业条款,而非免费使用。

示例输入:"冥想应用的背景音乐。纯器乐。需完整商业权利。每条曲目最长 5 分钟。"

示例输出:
- 模型:MusicGen-large(MIT),用于需完整商业权利的纯器乐。不使用 Stable Audio(非商业)。
- 授权:MIT——商业权利归部署方所有。曲目录音权利人:应用公司。
- 时长:分为 30 秒片段,3 秒交叉淡入淡出;10 段生成拼接 → 5 分钟。添加轻微的环境淡入/淡出包络以掩盖漂移。
- 提示词:`"slow ambient meditation, 60 BPM, soft strings and low pad, in D minor, no drums"`——锁定 BPM、锁定调性、锁定配器,明确排除打击元素。
- 披露:在应用署名中加入 `"AI-generated music"` 标签;元数据 `creator=AI-Gen:MusicGen-large, date=<iso>`。AudioSeal 可选(纯器乐伪造风险较低,但作为纵深防御)。
