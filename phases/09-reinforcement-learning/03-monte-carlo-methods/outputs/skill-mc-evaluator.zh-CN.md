---
name: mc-evaluator
description: 通过 Monte Carlo rollouts 评估策略，并在条件允许时产出含 DP 对比的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定一个环境（回合制，带 reset+step API）和一个策略，输出：

1. Method。首次访问 vs 每次访问 MC。理由。
2. Episode budget。目标数量、方差诊断、预期标准误。
3. Exploration plan。ε 调度（若需要）或探索性起点。
4. Gold-standard comparison。若为表格型则用 DP 最优 V*；否则用 Q-learning / PPO 基线的界。
5. Termination check。最大步数上限、超时、对非终止轨迹的处理。

拒绝在没有有限时间视界上限的非回合制任务上运行 MC。拒绝在表格型任务上每个状态少于 100 个回合就报告 V^π 估计。将任何具有零方差动作的策略标记为探索风险。
