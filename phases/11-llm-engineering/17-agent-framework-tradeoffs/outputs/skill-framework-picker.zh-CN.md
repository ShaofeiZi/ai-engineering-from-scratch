---
name: framework-picker
description: 根据抽象与问题形态的匹配程度，为智能体任务选择 LangGraph、CrewAI、AutoGen、Agno 或原生 Python。
version: 1.0.0
phase: 11
lesson: 17
tags: [langgraph, crewai, autogen, agno, agent-framework, orchestration, decision-matrix]
---

给定任务描述（问题形态、每次运行的 LLM 调用总数、分支模式、持久化与恢复需求、人在回路检查点、并行扇出、会话记忆、预期每日运行量），输出：

1. 形态匹配。一句话点出适合的抽象：图（类型化状态、命名转移）、组织架构图（专家角色、经理路由交接）、对话（智能体交谈直至完成）、带工具的单智能体。如果无法选出一个，说明该任务尚未具备智能体形态；停下来并先做分解。
2. 分支决策权。谁来选择下一步：开发者（显式边）、经理 LLM（CrewAI 层级式）、对话式涌现（AutoGen GroupChat）、工具调用自路由（Agno）。如适用，引用 LLM 选择路由的每轮 token 成本。
3. 状态预算。确认是否需要重启后恢复、时间旅行或人工中断。若需要，LangGraph 凭借状态优先的抽象胜出；Agno 仅覆盖会话级记忆。
4. 框架选择。输出 langgraph、crewai、autogen、agno、plain_python 之一。附一句话理由，将形态与状态的答案映射到该框架的核心抽象。
5. 逃生通道。如果每日运行量超过 10_000，或任务仅含两次及以下 LLM 调用且无状态，则推荐使用纯 Python 配合提供方 SDK。任务规模小时，不使用框架就是最快的框架。

拒绝为带有已知 DAG 的确定性工作流推荐 AutoGen；GroupChatManager 花费 token 去挑选发言人，而这些本可由开发者静态连接。CrewAI 确实支持通过 `output_pydantic` / `output_json` 实现结构化任务输出（参见 [docs.crewai.com/en/concepts/tasks](https://docs.crewai.com/en/concepts/tasks)），但其 `context` 通道仍流经下一任务的提示词字符串。当工作流依赖原始 `context` 在任务间传递结构化状态、却未接入上述任一输出 schema 时，应驳回 CrewAI。对于两次调用的摘要器应驳回 LangGraph；StateGraph 的开销纯属负担。当任务扇出到 4 个以上并行子工作单元且需要 reducer 语义时应驳回 Agno；Agno 提供了 `Parallel` 块，其输出按步骤名为键汇入一个 dict（参见 [docs-v1.agno.com/workflows_2/overview](https://docs-v1.agno.com/workflows_2/overview) 和 [docs.agno.com/workflows/access-previous-steps](https://docs.agno.com/workflows/access-previous-steps)），但它未暴露可与 LangGraph 的 Send 相提并论的扇出-归约 API。

示例输入："长期运行的研究工作流：规划、扇出到三个检索器、综合、人工审批摘要、撰写报告、引用来源。必须能在崩溃后恢复。面向生产，每日 50 次运行。"

示例输出：
- 形态：图。类型化规划、三个并行检索器、综合与撰写之间的命名转移。
- 分支：由开发者通过条件边决定。无每轮经理 LLM。
- 状态：需要恢复与人工中断。LangGraph 为必选项。
- 框架：langgraph。状态、Send 扇出、interrupt_before 与 PostgresSaver 均为一等公民。
- 逃生通道：不适用。每日 50 次运行远低于纯 Python 阈值，且该工作流状态化程度过高，不应脱离框架。
