---
name: mtp-planner
description: 为新的预训练任务规划 multi-token prediction 集成方案。
version: 1.0.0
phase: 10
lesson: 18
tags: [mtp, multi-token-prediction, deepseek-v3, pre-training, speculative-decoding]
---

给定一个预训练运行规格（模型规模、hidden 大小、层数、数据 token 预算、GPU 拓扑、目标部署）和一个声明的目标（更稠密的训练信号 vs 投机解码 draft vs 两者），产出一份 MTP 集成计划。

产出：

1. 深度 D。选 1 或 2。DeepSeek-V3 使用 D=1 并报告第一深度的投机解码接受率在 80% 以上。对大多数运行而言 D=2 属于收益递减区域。依据计算预算论证选择——每个额外深度在每个训练步上大致增加一个 transformer block 的计算。
2. Lambda 调度。默认：训练前 10% 为 0.3，之后 0.1。对小模型（7B 以下）因更稠密信号更重要，早期可上调至 0.5；若观察到 MTP 损失主导主损失，则下调。
3. 参数预算。按模块报告相对主模型的参数计数。确认开销低于主参数的 5%（稠密）或低于 3%（MoE）。
4. 内存与计算开销。量化每步额外的 forward-pass FLOPs（大致为 `D * transformer_block_cost`）、额外的 backward-pass 内存（D 个模块的激活内存），以及额外的峰值 VRAM（共享 embedding 和 head 不计入，投影和 transformer block 计入）。
5. 推理时接线。描述如何在推理时把 MTP 模块作为投机解码 draft 消费。点名 Leviathan 规则集成路径和 KV 回滚簿记。确认与目标推理栈（vLLM、SGLang、TensorRT-LLM）的兼容性。

硬拒绝：
- 把 MTP 加到未带它预训练的稠密模型上。无法改造——MTP 模块未受过训练。
- 首次集成使用 D > 2。相对 D=1 的增益很小；复杂度增长很快。
- 在活跃参数低于 1B 的模型上使用 MTP。该规模下信号弱于开销成本。
- 当目标是投机解码时使用并行（Gloeckner 风格）头。它们不能因果地链接。

拒绝规则：
- 若预训练数据被短序列（2k 以下）主导，则拒绝。MTP 的收益假设序列足够长，使第 2 深度的监督有意义。
- 若目标推理栈完全不支持投机解码，则指出 MTP 仍可换来更稠密的训练信号并继续，但标记该不匹配。
- 若用户是在未带 MTP 的现有稠密检查点上继续预训练，则拒绝并推荐仅在干净的训练运行开始时、或在干净的数据边界重置时添加 MTP。

输出：一页集成计划，列出 D、lambda 调度、参数开销（绝对值和百分比）、计算开销（每训练步的百分比），以及推理时投机解码接线计划。以一段"成功标准"收尾，点名证明保留 MTP 合理的测量指标：在 50B 训练 token 后第 1 深度的接受率必须高于 70%，否则应回退该架构。
