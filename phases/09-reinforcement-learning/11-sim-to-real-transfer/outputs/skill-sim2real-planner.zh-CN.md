---
name: sim2real-planner
description: 为给定机器人 + 任务规划 sim-to-real 迁移流水线，涵盖 DR、SI 和安全性。
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

给定一个机器人平台、一个任务以及对真实硬件时长的访问权限，输出：

1. Reality gap inventory。按预期影响排序的疑似来源（接触、感知、执行延迟、视觉）。
2. DR parameters。确切列表、范围、分布。依据真实测量值为每个范围给出理由。
3. SI steps。需测量哪些参数；测量方法。
4. Teacher/student split。教师使用哪些特权信息；学生使用哪些观测。
5. Safety envelope。底层限制、紧急停止、备用控制器。

拒绝在缺少 (a) 零样本仿真变体测试、(b) 安全护盾、(c) 回滚计划的情况下部署。将任何 DR 范围超过真实测量变异 3× 的情况标记为可能过度随机化。
