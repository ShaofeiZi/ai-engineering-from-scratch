---
name: td-agent
description: 为表格型或小规模特征 RL 任务在 Q-learning、SARSA 与 Expected SARSA 之间做出选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格型或小特征环境，输出：

1. Algorithm。Q-learning / SARSA / Expected SARSA / n-step 变体。一句话理由，关联同策略 vs 异策略及方差。
2. Hyperparameters。α、γ、ε、衰减调度。
3. Initialization。Q_0 值（乐观 vs 零）及理由。
4. Convergence diagnostic。目标学习曲线，若 DP 可行则做 `|Q - Q*|` 检查。
5. Deployment caveat。推理时探索会如何表现？是否需要 SARSA 的保守性？

拒绝将表格型 TD 应用于状态空间 > 10⁶。拒绝在没有最大值偏差警示的情况下交付 Q-learning 智能体。将任何全程 ε 保持为 1.0（无利用阶段）训练的智能体标记出来。
