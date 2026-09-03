---
name: mdp-modeler
description: 根据任务描述产出 Markov Decision Process 规格，并在训练前标记问题建模风险。
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

给定一个任务（控制 / 游戏 / 推荐 / LLM 微调），输出：

1. State。精确的特征向量或张量规格。论证马尔可夫性。
2. Action。离散集合或连续区间。维度。
3. Transition。确定性、已知模型的随机性，或仅采样。
4. Reward。函数与来源。稀疏 vs 塑造。终止奖励 vs 每步奖励。
5. Discount。取值与时间视界论证。

若状态不满足马尔可夫性且未显式说明帧堆叠（frame-stacking）或循环状态，则拒绝交付该 MDP。若奖励未以目标结果定义，则拒绝。对无限时间视界任务标记任何 `γ ≥ 1.0`。将任何奖励范围超过典型每步奖励 100 倍的情况标记为可能的梯度爆炸源。
