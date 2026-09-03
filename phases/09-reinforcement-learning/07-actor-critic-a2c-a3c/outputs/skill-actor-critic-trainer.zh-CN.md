---
name: actor-critic-trainer
description: 针对给定环境产出 A2C / A3C / GAE 配置，并明确优势估计方法和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定一个环境和算力预算，输出：

1. Parallelism。A2C（GPU 批处理）vs A3C（CPU 异步）及 worker 数量。
2. Rollout length T。每次更新每个环境的步数。
3. Advantage estimator。n-step 或 GAE(λ)；指定 λ。
4. Loss weights。`c_v`（值）、`c_e`（熵）、梯度裁剪。
5. Learning rates。Actor 和 critic（若使用则分离）。

拒绝在时间视界 > 1000 的环境上使用单 worker A2C（过于同策略、过慢）。拒绝在未做优势归一化的情况下交付。将任何 `c_e = 0` 且观测到熵 < 0.1 的运行标记为熵塌缩。
