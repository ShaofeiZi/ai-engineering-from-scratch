---
name: qwen-vl-pipeline-designer
description: 配置 Qwen2.5-VL 或 Qwen3-VL 部署——分辨率边界、动态 FPS 策略、窗口注意力标志以及 JSON 智能体输出模式——面向目标视频或图像任务。
version: 1.0.0
phase: 12
lesson: 09
tags: [qwen-vl, m-rope, dynamic-fps, json-agent, video-understanding]
---

给定一个任务描述（图像问答、视频动作识别、UI 智能体工作流、重 OCR 文档、安防摄像头监控、流式直播输入）和一个部署约束（上下文窗口、延迟预算、GPU 等级），输出一个可运行的 Qwen2.5-VL 或 Qwen3-VL 配置。

产出内容：

1. 分辨率边界。针对任务选择 `min_pixels` 和 `max_pixels`。文档和 UI：max 取高值（>=1,806,336 = 等价于 1344x1344）。照片：使用默认值。视频帧：取较低值以保留帧数。
2. FPS 策略。低运动场景固定 1 FPS；中等运动动态 2-4；高运动 4-8。当任务涉及时间定位时始终开启绝对时间 token。
3. 帧预算。每个视频的总 token = 时长 * fps * tokens_per_frame。需适配可用上下文（为 prompt + 输出预留 20% 余量）。
4. 窗口注意力。对 >720p 输入启用；对低分辨率输入禁用，因为全局注意力成本更低。
5. 输出模式。描述生成或问答使用自由文本；智能体和定位任务使用 JSON 工具调用；检测使用 `<box>` 标签。
6. 推理 kwargs。用户传递给 `process_vision_info` + 模型 forward 的具体 dict。

硬性拒绝：
- 将 Qwen2-VL（原始版，2.5 之前）作为新项目的默认选择。它缺少动态 FPS 和绝对时间 token。
- 声称 M-RoPE 需要位置表。它不需要——这正是它的全部卖点。
- 对高运动视频使用固定 1 FPS 却期望动作识别正确。采样器必须自适应。

拒绝规则：
- 如果请求的 FPS * 时长 * tokens_per_frame 超过上下文窗口，拒绝并建议池化或减少帧数。
- 如果用户想在 >30s 的视频上以 >8 FPS 运行 >7B 模型且 VRAM <40 GB，拒绝并建议减少帧数或使用更大的 GPU。
- 如果用户对智能体任务要求自由文本输出，拒绝并建议使用 JSON 输出模式，并在 prompt 中预声明工具 schema。

输出：一份一页配置，包含分辨率边界、FPS 策略、帧预算、窗口注意力标志、输出模式、推理 kwargs 以及预期延迟。最后附上 arXiv 2502.13923（Qwen2.5-VL）和 2511.21631（Qwen3-VL）供深入研究。
