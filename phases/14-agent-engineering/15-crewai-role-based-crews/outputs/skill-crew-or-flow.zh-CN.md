---
name: crew-or-flow
description: 针对给定任务选择 CrewAI 的 Crew 或 Flow，并搭建最小实现。
version: 1.0.0
phase: 14
lesson: 15
tags: [crewai, crews, flows, multi-agent, role-based]
---

给定任务描述，选择 Crew（自主型）或 Flow（确定性型），然后搭建脚手架。

决策：

1. 任务是否有 SLA、合规性或确定性重放要求？ -> Flow。
2. 任务是否属于探索性（研究、初稿、头脑风暴）？ -> Crew。
3. 任务是否有 4 个及以上由 LLM 决定执行顺序的专家？ -> 分层式 Crew。
4. 任务是否有不超过 3 个按固定顺序执行的专家？ -> 顺序式 Crew 或 Flow — 优先选择 Flow。

对于 Crew，产出：

1. 智能体定义：角色、目标、背景故事（精炼，不超过 200 字）、工具。
2. 任务定义：描述、expected_output、所属智能体。
3. 使用正确 Process（Sequential | Hierarchical）的 Crew。
4. 一个测试框架，在示例输入上运行该 Crew，并检查是否产出了预期的 expected_output。

对于 Flow，产出：

1. `@start` 入口函数。
2. 通过 `@listen(topic)` 步骤构成的 DAG。
3. 显式事件主题；不使用魔法式广播。
4. 一个重放框架：给定 kickoff 载荷，可确定性地重放。

硬性拒绝：

- 没有背景故事的 Crew。背景故事是承重部分。
- 没有显式主题名称的 Flow。“隐式链式调用”违背了审计目的。
- 只有 2 个专家的分层式 Crew。管理开销得不偿失。

拒绝规则：

- 如果用户要求在生产专属合规任务上使用 Crew，拒绝并迁移到 Flow。
- 如果用户要求在开放式研究任务上使用 Flow，拒绝并迁移到 Crew。
- 如果背景故事超过 200 字，拒绝并要求精简。上下文预算有限。

输出：`agents.py`、`tasks.py`、`crew.py` 或 `flow.py`，外加包含决策依据的 `README.md`。最后以“后续阅读”结尾，指向第 24 课（Langfuse/AgentOps）以了解可观测性，或第 13 课（如果 Flow 需要持久化恢复语义）。
