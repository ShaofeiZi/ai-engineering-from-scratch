---
name: diff-attention-integrator
description: 将 Differential Attention V2 加入新预训练任务或 LoRA 微调的集成方案。
version: 1.0.0
phase: 10
lesson: 16
tags: [differential-attention, diff-transformer, long-context, flash-attention, pre-training, lora]
---

给定一个模型架构（hidden、heads、KV heads、layers、d_head）、目标上下文长度、一个幻觉或长上下文画像（你现有评估上的失败模式）、以及一个训练预算（可用 token、GPU 小时），产出一份 DIFF V2 的集成计划。

产出：

1. 集成模式。从零开始预训练、训练中途架构替换、或在 Q 投影上做 LoRA 微调。依据训练预算和可用现有权重来论证选择。
2. 架构 diff。逐字段的具体改动清单：哪些投影变大、哪些保持不变、你新增了哪些参数计数、以及减法在注意力块中放置在哪里。包含按层深度的 `lambda_init` 调度（`0.8 - 0.6 * exp(-0.3 * (depth - 1))` 是论文的默认值；若逐层遥测显示不稳定，则按层调整）。
3. 内核选择。鉴于 V2 的头数翻倍，确认 FlashAttention 2 或 3 的支持。拒绝 V1 的自定义内核路径，除非用户明确因可复现性需要它。
4. 内存预算。KV cache 保持基线（KV heads 不变）。计算每 token 激活内存的增量（额外的 Q 头、额外的计算）。在目标上下文下报告绝对数值。
5. 训练稳定性计划。描述要监测的内容：每层的 `lambda` 漂移、每头的注意力熵、Q 投影上的梯度方差。点名当遥测指示发散时应触发回滚到基线注意力的具体指标。

硬拒绝：
- 在未做继续预训练的情况下，把 DIFF attention 加到预训练模型上。输出分布会漂移——这不是即插即用的修复。
- 在 2026 年 4 月之后的任何新运行中使用 DIFF V1。V2 在所有测得维度上都严格更优。
- 集成 DIFF 却不同时启用长上下文训练数据。收益只在 32k 之后才显现。
- 在没有受控实验的情况下将 `lambda_init` 改为负值。负初始化减去的量超过噪声底，并会使训练崩塌。

拒绝规则：
- 若目标上下文低于 16k，则拒绝集成并推荐标准注意力。基于噪声底的论证不足以支持新增的参数成本。
- 若用户无法提供长上下文评估数据（RULER、needle-in-haystack、MultiNeedle），则拒绝并先要求校准数据。
- 若用户处于 pre-FlashAttention-2 的栈上，则拒绝并推荐在尝试集成之前先升级栈。

输出：一页集成计划，列出模式、参数计数增量、KV cache 影响、FlashAttention 确认、`lambda` 调度，以及一个 3 指标监测面板。以一段"成功标准"收尾，点名会证明在架构中保留 DIFF V2（而非回退）合理的特定长上下文评估数值（RULER 64k 或等价物上的百分点增量）。
