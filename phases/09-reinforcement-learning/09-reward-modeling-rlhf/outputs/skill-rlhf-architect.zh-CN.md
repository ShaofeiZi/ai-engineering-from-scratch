---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐流水线，涵盖 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基础 LM、一个目标行为（对齐 / 推理 / 拒绝 / 智能体）以及一个偏好或验证器预算，输出：

1. Stage。SFT？RM？DPO？GRPO？附理由。
2. Preference or verifier source。人类、AI 反馈、基于规则、单元测试通过，或奖励蒸馏。
3. KL strategy。固定 β、自适应 β，或 DPO（隐式 KL）。
4. Diagnostics。平均 KL、奖励稳定性、过度优化防护（留出人类评测）。
5. Safety gate。红队集合、拒绝率、与有用性 RM 分离的安全 RM。

拒绝在没有 KL 监控的情况下交付 RLHF-PPO。拒绝使用小于目标策略的 RM。拒绝仅基于长度的奖励。将任何未保留盲测人类评测集的流水线标记为缺乏过度优化保护。
