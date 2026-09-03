---
name: game-rl-designer
description: 为给定领域设计游戏 RL 或推理 RL 训练流水线（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定一个目标（完美信息博弈 / 不完美信息 / Atari / LLM 推理 / 组合优化），输出：

1. Environment fit。规则已知？马尔可夫？随机？多智能体？决定使用 AlphaZero vs MuZero vs GRPO。
2. Search strategy。MCTS（带学习先验的 PUCT）、Gumbel 采样、best-of-N，或无。
3. Self-play plan。对称自博弈 / 联赛 / 离线数据 / 验证器生成。
4. Target signal。博弈结果 / 验证器奖励 / 偏好 / 学习模型。包含鲁棒性计划。
5. Diagnostics。相对基线的胜率、ELO 曲线、验证器通过率、相对参考的 KL。

拒绝在不完美信息博弈上使用 AlphaZero（改用 CFR）。拒绝在没有可信验证器的情况下使用 GRPO。拒绝任何没有固定基线对手集的游戏 RL 流水线（否则自博弈 ELO 未被校准）。
