---
name: runtime-picker
description: 根据给定的技术栈、延迟预算和运维形态，选择一个生产级智能体运行时（Agno、Mastra、LangGraph、提供商 SDK）。
version: 1.0.0
phase: 14
lesson: 18
tags: [agno, mastra, langgraph, runtime, selection]
---

根据技术栈、延迟预算、所需原语和运维形态，选择一个运行时。

决策：

1. Python + FastAPI + 每秒数千个短生命周期智能体 -> **Agno**。
2. TypeScript + Next.js/Vercel + 统一多提供商 -> **Mastra**。
3. 持久化状态、显式图、故障后恢复 -> **LangGraph**（第 13 课）。
4. 以 Claude 为中心的产品，希望采用 Claude Code 框架形态 -> **Claude Agent SDK**（第 17 课）。
5. 以 OpenAI 为中心的产品，需要交接 + 护栏 + 链路追踪 -> **OpenAI Agents SDK**（第 16 课）。
6. 多智能体团队、actor 模型并发、故障隔离 -> **AutoGen v0.4** / **Microsoft Agent Framework**（第 14 课）。
7. 基于角色的协作或事件驱动的确定性工作流 -> **CrewAI** Crew 或 Flow（第 15 课）。
8. 以上都不符合 -> 直接调用 API + 第 01 课中的标准库循环。

产出：

- 一份简短的决策文档：技术栈、延迟目标、所需原语、观察到的权衡。
- 所选运行时的最小脚手架。
- 如果当前已在使用其他运行时，提供迁移计划。

硬性拒绝：

- 仅凭“性能”就选择 Agno 或 Mastra，而实际负载是每个请求只有一次慢调用。性能很少是瓶颈。
- 在 Python 单体仓库中选择 TypeScript 运行时且没有合理理由。混合语言的智能体代码是一种运维成本。
- 为无状态短任务选择 LangGraph。检查点机制会引入额外开销，而简单工作流（第 12 课）可以避免这一点。

拒绝规则：

- 如果用户想要“五个运行时全部，用来对比”，拒绝。应基于你自己的负载进行基准测试；框架厂商的基准测试仅供参考。
- 如果用户想要自行托管 Mastra 的 `ee/` 功能，拒绝并指向许可条款。
- 如果产品需要长时间运行的异步工作（数小时到数天），拒绝自托管方案，并引导至 Claude Managed Agents 或基于队列的架构（第 29 课）。

输出：决策文档 + 脚手架 + README。最后以“接下来读什么”结尾，指向第 24 课（可观测性）和第 29 课（生产运行时），即框架之上的运维层。
