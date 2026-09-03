---
name: web-desktop-harness
description: 构建一个 WebArena/OSWorld 风格的测试框架，包含基于执行的评价和轨迹效率指标。
version: 1.0.0
phase: 14
lesson: 20
tags: [webarena, osworld, harness, trajectory-efficiency]
---

给定一个目标应用（web 或桌面）以及一组带有黄金轨迹的任务，构建一个评测测试框架。

需要产出：

1. 任务定义：`(tid, description, gold_steps, success_predicate, state_reset)`。
2. 运行器：运行智能体，捕获每一个动作，记录步数 + 耗时 + 成功状态。
3. 轨迹效率指标：`agent_steps / gold_steps`。报告每个任务的值及汇总值。
4. 任务之间进行状态重置——绝不能在前一个任务弄脏的状态上运行另一个任务。
5. 失败模式分类器：对每次失败，标注其是定位失误（grounding miss，选错元素）还是规划失误（planning miss，选错动作）。

硬性拒绝条件：

- 任务之间不进行状态重置。跨任务污染会使所有分数失效。
- 仅报告成功率。轨迹效率是 2026 年的标准。
- 仅截图、缺少 DOM 对齐的测试框架。某些智能体使用 DOM+视觉；除非专门限制交互界面，否则应同时提供两者。

拒绝规则：

- 如果任务没有黄金轨迹，拒绝。没有黄金轨迹就无法衡量效率。
- 如果应用未固定到特定版本，拒绝。版本漂移会使跨运行比较失效。
- 如果智能体拥有破坏性工具（删除、发布），要求提供应用的沙箱副本。

输出：`tasks.py`、`runner.py`、`failure_classifier.py`、`report.py`、`README.md`（说明重置策略、黄金轨迹来源以及定位与规划的划分）。以"延伸阅读"结尾，指向 Lesson 21（computer use models）或 Lesson 30（eval-driven development）。
