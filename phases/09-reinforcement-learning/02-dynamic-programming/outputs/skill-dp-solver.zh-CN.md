---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解小型表格型 MDP，并报告收敛表现。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定一个已知模型的 MDP，输出：

1. Choice。策略迭代 vs 值迭代。基于 |S|、|A|、γ 的理由。
2. Initialization。V_0，起始策略。收敛敏感性。
3. Stopping。上确界范数容差 ε。预期扫描次数。
4. Verification。精确计算的 V*(s_0)。提取贪心策略。
5. Use。此基线将如何用于调试/评估基于采样的方法。

拒绝在状态空间 > 10⁷ 上运行 DP。拒绝在未做上确界范数检查的情况下声称收敛。对无限时间视界任务标记任何 γ ≥ 1 为保证违反。
