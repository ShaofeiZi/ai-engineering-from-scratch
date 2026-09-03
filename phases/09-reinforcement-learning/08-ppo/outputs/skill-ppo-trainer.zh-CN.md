---
name: ppo-trainer
description: 针对给定环境产出 PPO 训练配置和诊断方案。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定一个环境和训练预算，输出：

1. Rollout size。`N` 个环境 × `T` 步。
2. Update schedule。`K` 个 epoch、小批量大小、LR 调度。
3. Surrogate params。`ε`（裁剪）、`c_v`、`c_e`，优势归一化开启。
4. Advantage。GAE(`λ`)，显式指定 `γ` 和 `λ`。
5. Diagnostics plan。KL、裁剪比例、解释方差阈值及告警。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信任域）。拒绝任何没有优势归一化或 KL/裁剪监控的 PPO 运行。将裁剪比例持续高于 0.4 标记为漂移。
