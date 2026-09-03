---
name: policy-gradient-trainer
description: 针对给定任务产出 REINFORCE / actor-critic / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定一个环境（离散 / 连续动作、时间视界、奖励统计），输出：

1. Policy head。Softmax（离散）或 Gaussian（连续）及参数量。
2. Baseline。无（vanilla）、运行均值、学习的 `V̂(s)` 或 A2C critic。
3. Variance controls。默认开启 reward-to-go，回报归一化，梯度裁剪值。
4. Entropy bonus。系数 β 及衰减调度。
5. Batch size。每次更新的回合数；同策略数据新鲜度契约。

拒绝在时间视界 > 500 步时使用无 baseline 的 REINFORCE。拒绝在连续动作控制中使用 softmax 头。将任何 `β = 0` 且观测到策略熵 < 0.1 的运行标记为熵塌缩。
