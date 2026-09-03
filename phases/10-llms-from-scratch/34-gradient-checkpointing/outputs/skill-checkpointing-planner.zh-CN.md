---
name: checkpointing-planner
description: 根据训练配置和 HBM 预算，为每一层选择激活重计算策略（none / selective / full / offload）。
version: 1.0.0
phase: 10
lesson: 34
tags: [gradient-checkpointing, activation-recomputation, selective-checkpoint, fsdp-offload, training-memory]
---

给定训练配置（层数 L、hidden size d、序列长度 S、microbatch B、每值 dtype 字节数、attention kernel、tensor-parallel 度 TP、pipeline-parallel 度 PP、MoE 时的 expert-parallel 度 EP）以及扣除权重和优化器状态后的每 rank HBM 预算，输出：

1. 逐层策略。对栈中每个层族（embedding、attention、FFN、MoE expert、norm、output head）在 none、selective、full、offload 中选一。默认：S 超过 4_096 时对 attention 采用 selective；对 residual 流和 norm 默认 none；仅当该层激活的实测 PCIe 传输时间小于其实测重算时间时，才对 FFN 默认 offload。
2. 段大小 k。若开启 full checkpointing，对均匀层成本取 k 为 round(sqrt(L))；当激活显存主导预算时取更小的 k。以正向 FLOP 的 (1/k) 报告额外 FLOP 百分比。
3. FlashAttention 交互。确认 attention kernel 是否已重算 softmax。若是，则 selective attention checkpointing 收益甚微；降级为 none。按名称指明 kernel（FlashAttention-2/3、xFormers memory-efficient、vanilla）。
4. TP / PP 方案。对 TP，指出重算时需要 gather 或 rescatter 的激活，以及每步新增的通信字节数。对 PP，确认哪些流水线阶段被端到端 checkpoint，以便反向 microbatch 在回流前释放激活显存。
5. 预算算术。预测策略前后激活显存（每 rank MB）。预测 FLOP 开销占 fwd+bwd 的百分比。拒绝任何在 HBM 预算内不留 10% 余量的方案。

当仅对 attention 做 selective 即可平衡预算时，拒绝每层都 full checkpointing；profile 显示其 FLOP 开销在同等显存节省下比 selective 高出数倍，且确切比率因工作负载而异。当某层在目标 PCIe 链路上的实测激活传输时间超过其实测重算时间时，拒绝 offload；重算胜出。当所选框架不快照 amax 历史时，拒绝在 FP8 训练中“处处 checkpoint”；重算会使 scale 漂移并悄无声息地 corrupt 梯度。

示例输入："L=64, d=8192, S=8192, B=1, bf16, FlashAttention-3, TP=8, PP=4, HBM budget per rank 32 GB after weights, MoE with 8 experts and EP=8."

示例输出：
- 逐层策略：attention selective、FFN none、MoE expert full、embedding none、output head offload。
- 段大小：full 仅作用于 MoE，k=8；FLOP 开销在 expert 路径上 12%，其余为 0。
- FlashAttention 交互：FA-3 已重算 softmax；selective 作用于层 wrapper，而非 kernel 内部。
- TP / PP 方案：TP 在重算时 gather attention 输入，每步额外通信 0.3 GB；PP 各阶段对其完整正向做 checkpoint；PP 阶段 3 为最终反向保留其激活。
- 预算算术：无策略时激活 38 GB，有策略时 11 GB。FLOP 总开销占 fwd+bwd 的 7.5%。
