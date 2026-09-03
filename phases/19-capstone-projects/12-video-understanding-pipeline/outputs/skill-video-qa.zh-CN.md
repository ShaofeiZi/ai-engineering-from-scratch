---
name: video-qa
description: 构建一个视频理解管线，包含场景分割、多向量索引、时间定位和时间戳引用。
version: 1.0.0
phase: 19
lesson: 12
tags: [capstone, video, multimodal, gemini, qwen-vl, molmo, transnet, qdrant]
---

给定 100 小时的视频，构建一个摄入管线和查询系统，用 (start, end) 时间戳加帧预览回答自然语言问题。

构建计划：

1. 摄入视频（YouTube URL 或 MP4）；如需则降采样至 720p。
2. 使用 TransNetV2 或 PySceneDetect 进行场景分割；发射 `[{scene_id, start_ms, end_ms, keyframe_path}]`。
3. 使用 Whisper-v3-turbo（faster-whisper）进行 ASR，生成词级时间戳；按场景切片。
4. 使用 Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2 进行 VLM 字幕生成；发射字幕 + 帧嵌入。
5. Qdrant 多向量索引，每个场景三个命名向量（caption_emb、frame_emb、transcript_emb），载荷为 {video_id, scene_id, start_ms, end_ms, keyframe_url}。
6. 查询：三个并行稠密查询；倒数排名融合合并；top-k=5 个场景。
7. 时间定位（TimeLens 适配器或 VideoITG）在 top 场景内精炼 (start, end)。
8. VLM 合成（Gemini 2.5 Pro），输入为查询 + top-3 场景片段 + 转录文本；要求 `(video_id, start_ms, end_ms)` 引用。
9. 在 ActivityNet-QA、NeXT-GQA 以及一个 100 题人工标注的自定义集合上评估。报告整体准确率和按问题类别（描述性、计数、动作类型）的准确率。

评估标准：

| 权重 | 标准 | 度量 |
|:-:|---|---|
| 25 | 时间定位 IoU | 留出定位集上的 IoU |
| 20 | 问答准确率 | NeXT-GQA 和 100 题自定义集合 |
| 20 | 摄入吞吐 | 每美元索引的视频小时数 |
| 20 | UI 和引用用户体验 | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类型准确率分别报告 |

硬性拒绝：

- 管线在每个场景上仅池化为单一向量。多向量是让类别差异显现的必要条件。
- 回答没有 (start, end) 引用。
- 报告一个整体准确率而没有计数/动作类型子集分解。
- VLM 合成不直接接收场景帧（纯文本输入会丢失视觉定位）。

拒绝规则：

- 拒绝在许可证来源不清晰的情况下提供视频；要求每个 video_id 带有许可证标签。
- 拒绝在摄入速率高于已测吞吐的情况下声称"实时"响应。
- 拒绝将计数/动作类型幻觉数值隐藏在整体准确率数字中。

产出：一个包含场景分割 + ASR + 字幕生成管线、多向量 Qdrant 集合、时间定位适配器、带时间戳深链的 Next.js 15 查看器、三项基准评估结果（ActivityNet-QA、NeXT-GQA、自定义）的仓库，以及一份报告，指出你观察到的三个计数或动作类型失败类别，以及减少每种失败的检索或合成变更。
