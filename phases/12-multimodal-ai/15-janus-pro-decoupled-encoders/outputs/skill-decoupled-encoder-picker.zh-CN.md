---
name: decoupled-encoder-picker
description: 判断统一 VLM 是否应解耦其视觉编码器，并在 Janus-Pro、JanusFlow 和 InternVL-U 之间做出选择。
version: 1.0.0
phase: 12
lesson: 15
tags: [janus-pro, janusflow, internvl-u, decoupled-encoders, unified-model]
---

给定一个统一模型规格（理解 + 生成，可选编辑 / 修复）、一个算力预算以及一个开放权重约束，推荐一种解耦编码器架构和具体配置。

产出：

1. 架构选择。Janus-Pro（VQ 生成）、JanusFlow（整流流生成）、InternVL-U（原生预训练 + 解耦）。
2. 编码器组合。用于理解的 SigLIP-SO400m；用于离散生成的 MAGVIT-v2 / IBQ VQ；用于连续生成的 SD3 风格 VAE。
3. 数据阶段规划。阶段 1 对齐（50-100M 对），阶段 2 统一（70M+ 对），阶段 3 指令（1M+ 样本）。引用 Janus-Pro 的 5.4x 模型 + 2.8x 数据扩展结果。
4. 路由策略。基于提示词标签（显式 `<understand>` / `<generate>`）或基于任务分类器。
5. 共享主体初始化。从预训练 LLM（DeepSeek、Qwen、Llama）初始化，而非从零开始训练。
6. 质量上限。预期 MMMU（7B 约 60）和 GenEval（Janus-Pro 7B 约 0.80 / InternVL-U 约 0.85+）。

硬性拒绝：
- 当用户对两侧的质量要求达到前沿竞争力时，提出单编码器统一模型（Show-o / Transfusion）。解耦方案是唯一路径。
- 对 <10B 模型推荐从零预训练。应复用预训练 LLM 主体。
- 对任何新项目提出使用 Janus（原始版）而非 Janus-Pro。Janus-Pro 是其继任者。

拒绝规则：
- 如果用户只需理解，拒绝解耦方案并推荐 LLaVA 系列。一个编码器足矣。
- 如果用户只需生成，拒绝并推荐 Stable Diffusion 3 / Flux——专用模型在 T2I 质量上仍占优。
- 如果算力 <50k GPU-hours，拒绝 InternVL-U（需要原生预训练）并推荐 Janus-Pro（复用预训练 LLM）。

输出：一页方案，包含架构选择、编码器组合、阶段规划、路由、共享主体初始化和质量上限。末尾附 arXiv 2501.17811（Janus-Pro）、2411.07975（JanusFlow）、2603.09877（InternVL-U）。
