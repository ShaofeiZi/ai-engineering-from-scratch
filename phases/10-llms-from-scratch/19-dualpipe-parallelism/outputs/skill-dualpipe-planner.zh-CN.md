---
name: dualpipe-planner
description: 为训练集群规划流水线并行策略（1F1B、Zero Bubble、DualPipe、DualPipeV）。
version: 1.0.0
phase: 10
lesson: 19
tags: [pipeline-parallelism, dualpipe, dualpipev, zero-bubble, expert-parallelism, distributed-training]
---

给定一个训练集群规格（总 GPU 数、互联拓扑、加速器型号、每 GPU 内存）、一个模型形状（总参数、活跃参数、MoE 或稠密、期望层数），以及一个目标训练数据量，推荐一种流水线并行策略并确认期望的气泡占比。

产出：

1. 流水线深度 P。依据 GPU 内存预算（每 rank 必须容纳一个流水线阶段）、MoE 还是稠密、以及互联带宽来选择。范围：小集群 4，前沿 MoE 训练 16-32。
2. Micro-batch 数 M。对 DualPipe 和 DualPipeV 必须能被 2 整除。典型比率 M/P 在 8 到 16 之间。依据梯度累积目标和目标序列长度下的激活内存来论证。
3. 调度选择。从 1F1B、Zero Bubble、DualPipe、DualPipeV 中选择。决策表：500 GPU 以下的稠密训练 -> Zero Bubble。带 expert parallelism 的 MoE -> DualPipe。500 GPU 以上、没有重型 all-to-all 的稠密训练 -> DualPipeV。100 GPU 以下的小运行 -> 1F1B 即可。
4. 期望气泡占比。针对所选调度在目标 P 和 M 下计算。报告为百分比，以及相对 1F1B 在总训练预算下节省的绝对 GPU 小时。
5. 参数复制计划（仅 DualPipe）。确认 2x 参数复制可放入可用 VRAM。报告在所选 P 下每 GPU 的有效参数密度。

硬拒绝：
- 没有 Expert Parallelism 的 DualPipe。没有需要隐藏的 EP 重量级通信，2x 复制不合理。
- 在任何训练运行上 P > 64。无论何种调度，气泡占比随 P 线性增长。
- 对 DualPipe/DualPipeV micro-batch 数不能被 2 整除。调度无法闭合。
- 当模型可放入单 GPU 内存时使用流水线并行。应仅使用数据并行。

拒绝规则：
- 若互联为每 GPU 200Gbps 或更慢，则拒绝 DualPipe 并推荐 DualPipeV。all-to-all 重叠窗口太窄，不足以证明复制的合理性。
- 若用户无法提供适合其集群拓扑的自定义 all-to-all 内核，则推荐 Zero Bubble 而非 DualPipe。
- 若训练运行低于 1B token，则完全拒绝流水线并行规划并推荐数据并行加张量并行。

输出：一页计划，列出 P、M、调度、期望气泡占比、参数复制成本（若 DualPipe），以及一个 all-to-all 内核推荐。以一段"回滚触发器"收尾，点名会证明在未命中目标数值时切换到更简单调度合理的特定利用率指标（聚合 GPU 利用率百分比，在前 1000 步上测量）。
