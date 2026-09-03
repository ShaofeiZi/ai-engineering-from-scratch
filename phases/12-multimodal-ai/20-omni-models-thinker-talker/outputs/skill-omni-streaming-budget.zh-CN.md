---
name: omni-streaming-budget
description: 根据目标 TTFAB 和功能集，为 Thinker-Talker 流式语音流水线（Qwen-Omni / Moshi / Mini-Omni）进行容量规划。
version: 1.0.0
phase: 12
lesson: 20
tags: [qwen-omni, moshi, mini-omni, streaming, ttfab, thinker-talker]
---

给定一个语音优先产品规格（目标 TTFAB、麦克风采样率、是否有视觉输入、是否双语、是否全双工）和算力约束（GPU 级别、预算），为 Thinker-Talker 流水线进行容量规划。

产出内容：

1. 模型族选择。Moshi（延迟最优）、Qwen2.5-Omni（开源功能最优）、Qwen3-Omni（前沿质量）、Mini-Omni（最简单）。
2. Thinker 与 Talker 规模。<400ms TTFAB 用 7B Thinker + 200-300M Talker。质量优先用 70B+ Thinker，接受更高 TTFAB。
3. TTFAB 分解。逐组件的延迟估算。
4. 双工模式。默认半双工 + VAD 轮次切换；若产品需要反馈通道则用全双工。
5. 视觉集成。用带绝对时间戳的 TMRoPE 处理交错视频帧。
6. 部署形态。根据吞吐需求选择单 GPU 或拆分（Thinker 在 A，Talker 在 B）。

硬性拒绝：
- 提议 70B Talker。Talker 必须足够小才能跟上语音 token 速率。
- 使用非流式语音解码器。TTFAB 会暴涨。
- 声称全双工即插即用。它需要专门的训练数据。

拒绝规则：
- 若目标 TTFAB <200ms，拒绝任何大于 Moshi 级别（7B 融合）的方案在单张 A100 上运行。
- 若产品需要在流中生成音乐，拒绝本架构并推荐独立的音乐流水线。
- 若麦克风采样率为 48kHz 且质量要求严格，标记需要更强的语音编码器；不要盲目降采样。

输出：一页流式方案，包含模型选择、规模、TTFAB 分解、双工模式、视觉策略、部署形态。最后附上 arXiv 2503.20215（Qwen2.5-Omni）、2410.00037（Moshi）。
