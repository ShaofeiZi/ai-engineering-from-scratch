---
name: orchestration-picker
description: 针对给定问题，选择一种最小化的编排拓扑（supervisor、swarm、hierarchical、debate 或不使用），并完成最小化实现。
version: 1.0.0
phase: 14
lesson: 28
tags: [orchestration, supervisor, swarm, hierarchical, debate]
---

给定一个产品领域和一类任务，选择最小化的拓扑。

决策：

1. 1 个智能体 + 工作流模式（Lesson 12）即可胜任？-> 完全不使用拓扑。
2. 2-4 个职责各异的专家？-> **supervisor-worker**。
3. 对延迟敏感且专家之间可以干净地交接？-> **swarm**。
4. 10+ 个专家，supervisor 的上下文预算已不够用？-> **hierarchical**。
5. 准确性比成本更重要，多提议者 + 评审有帮助？-> **debate**（Lesson 25）。

产出：

1. 所选拓扑的脚手架。
2. swarm 上的跳数计数器；hierarchical 上的嵌套深度限制；debate 上的轮次上限。
3. 每次交接或每步的可观测性钩子（OTel GenAI spans，Lesson 23）。
4. 一段 "为什么选这个，而不是那个" 的 README 章节。

严格拒绝：

- 将连续 3 次 LLM 调用称为 "多智能体"。那只是提示词链。
- 没有跳数计数器的 swarm。来回弹跳是必然的。
- hierarchical 在每个分支末端只剩 1 个专家。予以扁平化。

拒绝规则：

- 如果用户想为单次 ReAct 循环即可处理的任务使用多智能体，拒绝并建议 Lesson 01。
- 如果用户想为 2 步任务使用 supervisor，拒绝并建议提示词链（Lesson 12）。
- 如果该领域有合规 / 审计要求，拒绝 swarm 并建议 supervisor 或 hierarchical。

输出：拓扑脚手架 + 包含决策依据的 README。以 "接下来读什么" 结尾，指向 Lesson 13（LangGraph）了解 supervisor 实现，Lesson 16（OpenAI Agents SDK）了解工具式交接，或 Lesson 25 了解 debate 的具体细节。
