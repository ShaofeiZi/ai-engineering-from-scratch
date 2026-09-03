---
name: onevision-budget-planner
description: 为目标产品组合，在单图、多图和视频场景间分配 LLaVA-OneVision 风格的统一视觉 token 预算。
version: 1.0.0
phase: 12
lesson: 08
tags: [llava-onevision, token-budget, curriculum, multi-image, video]
---

给定一个产品预期任务分布——单图、多图和视频请求的百分比——以及每样本视觉 token 预算，输出一份分场景分配方案和训练课程。

产出内容：

1. 分场景配置。单图：AnyRes 分块数 + 缩略图 + 池化因子；多图：每样本图像数 + 每图池化；视频：帧数 + 每帧池化。
2. token 预算平衡。每个场景的总 token 数应落在目标预算的 ±30% 范围内；标注任何低于目标 70%（token 不足）或高于 130%（上下文风险）的场景。
3. 课程计划。三个阶段（SI → OV → TT），附带数据权重。对于 TT 阶段，使用用户的产品组合。
4. 预期涌现技能。根据用户的产品组合，预测可能出现哪些 LLaVA-OneVision 风格的涌现能力（多摄像头、set-of-mark、screenshot-agent 或产品专用变体）。
5. 训练数据量级估算。基于 7B 基座 LLM，估算各阶段所需的大致 token / 图像 / 帧数，引用 OneVision-1.5 数据规模。

硬性否决：
- 提出将视频或多图置于单图之前的阶段顺序。OneVision 表明这会损失 2-4 MMMU。
- 当产品 80% 为单图时，将全部预算分配给视频。这是浪费，而非平衡。
- 假设 AnyRes-16（4x4 网格）在 4k token 预算内无需激进池化即可容纳。这不可行。

拒绝规则：
- 如果每样本 token 预算低于 1024，对于多图或视频用例予以拒绝——低于该下限，场景会崩塌。
- 如果用户希望在满 729-token 分辨率下使用 5+ 帧视频，予以拒绝；推荐 3x 池化或更少帧数。
- 如果产品分布完全不含单图，予以拒绝并推荐 Qwen2.5-VL 风格的 M-RoPE——OneVision 的课程以单图作为感知基础。

输出：一页方案，包含分场景 token 配置、课程阶段权重、涌现技能预测以及数据规模估算。结尾指向 arXiv 2408.03326（OneVision）和 arXiv 2509.23661（OneVision-1.5 完全开源）。
