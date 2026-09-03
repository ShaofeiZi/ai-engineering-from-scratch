---
name: positional-encoding-picker
description: 根据上下文长度和训练预算，选择位置编码（RoPE、ALiBi、正弦）及缩放策略。
version: 1.0.0
phase: 7
lesson: 4
tags: [transformers, positional-encoding, rope, alibi]
---

给定一个 Transformer 规格（推理时的目标上下文长度、训练时的上下文长度、外推要求、以 token 计的微调预算），输出：

1. 基础编码。从以下中选择其一：RoPE、ALiBi、正弦（sinusoidal）、可学习的绝对位置编码（learned-absolute）。给出一句理由。
2. 超参数。若选 RoPE：`base` 的值、`d_head` 对均匀拆分的要求。若选 ALiBi：斜率公式。若选正弦编码：`max_len`。
3. 扩展策略。若目标长度 > 训练长度：给出 NTK 感知缩放因子、YaRN 配置、LongRoPE 规格或位置插值比例。说明微调所需的 token 预算。
4. 测试计划。在最大上下文长度下的 NIAH（大海捞针）通过率目标，以及困惑度相对于训练长度基线保持在 X 以内。
5. 回退方案。若长上下文评测失败该如何处理：用更大的 `base` 重新训练、切换为 ALiBi，或限制部署时的上下文长度。

拒绝在 2026 年为新模型推荐正弦编码或可学习的绝对位置编码——它们无法外推，且所有现代技术栈都默认使用 RoPE 或 ALiBi。拒绝在没有微调阶段的情况下将 RoPE 缩放超过训练长度的 8 倍。拒绝在未对完整部署长度进行 NIAH 测试的情况下发布长上下文配置。
