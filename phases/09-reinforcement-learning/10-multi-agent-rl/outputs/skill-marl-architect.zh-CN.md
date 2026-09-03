---
name: marl-architect
description: 针对给定任务选择合适的多智能体 RL 范式（IPPO、CTDE、self-play、league）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定一个包含 `n` 个智能体的任务，输出：

1. Regime classification。合作 / 对抗 / 一般和。给出理由。
2. Algorithm。IPPO / MAPPO / QMIX / 自博弈 / 联赛。基于耦合紧密度和奖励结构的理由。
3. Information access。集中式训练（哪些全局信息给 critic）？分布式执行？
4. Credit assignment。反事实基线、值分解，或奖励塑造。
5. Exploration plan。每个智能体的熵、基于种群的训练，或联赛。

拒绝在紧密耦合的合作任务上使用独立 Q-learning。拒绝对存在循环风险的一般和博弈推荐自博弈。将任何没有固定对手评测的 MARL 流水线标记出来（自博弈数据常被挑选，不可信）。
