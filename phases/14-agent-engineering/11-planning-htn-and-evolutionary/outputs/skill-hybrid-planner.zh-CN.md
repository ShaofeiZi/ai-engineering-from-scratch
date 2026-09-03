---
name: hybrid-planner
description: 构建一个混合规划器 —— 使用 ChatHTN 生成可证明正确的计划，使用 AlphaEvolve 结合机器可校验的评估器进行代码搜索 —— 并针对问题选择合适的规划器。
version: 1.0.0
phase: 14
lesson: 11
tags: [planning, htn, chathtn, alphaevolve, evolutionary-search]
---

给定一个问题类别（受策略约束的工作流 vs 代码优化 vs 开放式任务），选择一个规划器并生成一个正确的脚手架。

决策：

1. 问题是否存在硬性前置条件 / 策略 / 调度约束？ -> HTN（ChatHTN）。
2. 问题是否存在确定性的、机器可校验的适应度函数？ -> 进化式（AlphaEvolve）。
3. 都不是？ -> 转而使用 ReAct（第 01 课）或 ReWOO（第 02 课）。

对于 HTN，需产出：

1. `Operator` 类型，包含 `preconditions`、`effects_add`、`effects_remove`。
2. `Method` 类型，包含 `task`、`preconditions`、`subtasks`。
3. 一个规划器：优先尝试方法，回退到 LLM 分解，并缓存成功的 LLM 分解结果。
4. 一个校验步骤：拒绝引用了未知算子或方法的 LLM 分解结果。

对于进化式，需产出：

1. 候选程序的初始种群。
2. 一个返回标量适应度的确定性评估器。
3. 一个变异算子（LLM 驱动或基于规则）。
4. 一个选择循环（保留前 k 个，变异，重复），并带有提前停止机制。

硬性拒绝：

- ChatHTN 中 LLM 输出在未经算子模式校验的情况下直接应用。正确性保证将失效。
- AlphaEvolve 中评估器调用 LLM 评判。适应度必须是确定性的；LLM 评判会引入循环无法恢复的随机噪声。
- 对开放式任务（"写一篇博客文章"）使用任一模式。没有评估器，没有前置条件 -> 使用 ReAct。

拒绝规则：

- 如果领域没有明确的算子模式，拒绝 ChatHTN。建议使用 ReWOO 或普通 ReAct。
- 如果领域没有机器可校验的适应度，拒绝 AlphaEvolve。建议使用 Self-Refine（第 05 课）。
- 如果用户想要"规划器 + LLM 做最终决策"，拒绝。符号正确性与 LLM 探索之间的分界是承重设计。

输出：`operators.py`、`methods.py`、`planner.py`（HTN）或 `evaluator.py`、`mutator.py`、`loop.py`（进化式），外加包含决策理由的 `README.md`。最后以"延伸阅读"结尾：如果辩论式校验适合该问题，指向第 25 课；如果任务实际上是 ReWOO 形态，则指向第 02 课。
