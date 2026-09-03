---
name: dqn-trainer
description: 为离散动作 RL 任务产出 DQN 训练配置（缓冲区、目标网络同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定一个离散动作环境（观测形状、动作数、时间视界、奖励尺度），输出：

1. Network。架构（MLP / CNN / Transformer），特征维度，深度。
2. Replay buffer。容量、小批量大小、预热规模。
3. Target network。同步策略（每 C 步硬同步或软更新 τ）。
4. Exploration。ε 起始 / 结束 / 调度长度。
5. Loss。Huber vs MSE，梯度裁剪值，奖励裁剪规则。
6. Double DQN。默认开启，除非有明确理由禁用。

拒绝交付没有目标网络、没有经验回放缓冲区或 ε 保持为 1 的 DQN。拒绝连续动作任务（改用 SAC / TD3）。将任何奖励范围超过每步均值 10× 的情况标记为需要裁剪或尺度归一化。
