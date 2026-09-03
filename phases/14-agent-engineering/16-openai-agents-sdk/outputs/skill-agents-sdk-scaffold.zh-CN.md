---
name: agents-sdk-scaffold
description: 搭建一个 OpenAI Agents SDK 应用，包含分诊智能体、交接、输入/输出/工具护栏、会话存储和一个 trace 处理器。
version: 1.0.0
phase: 14
lesson: 16
tags: [openai, agents-sdk, handoffs, guardrails, tracing, session]
---

给定一个产品领域和一组专家智能体，搭建一个 OpenAI Agents SDK 应用。

产出：

1. 每个专家对应一个 `Agent`，外加一个只具备交接（不含领域工具）的 `triage` 智能体。
2. 每个领域工具对应一个 `FunctionTool`，具有类型化输入 schema、清晰的描述（告知模型何时使用它）以及执行沙箱。
3. 从分诊智能体到每个专家智能体的 `Handoff`。验证工具名称遵循 `transfer_to_<agent>` 约定。
4. 用于 PII、策略、范围的 `InputGuardrail`。默认使用并行模式，除非护栏 LLM 相对于主模型规模较大——此时使用阻塞模式。
5. 用于长度、PII、策略的 `OutputGuardrail`。对于安全关键型输出，在生产环境中始终使用阻塞模式。
6. 对涉及网络或文件系统的函数工具施加逐工具护栏。
7. `Session` 存储（默认 SQLite；生产环境使用 Redis）。
8. `add_trace_processor`，将 span 连接到你的后端，并与 OpenAI 的 trace UI 并行使用。

硬性拒绝：

- 分诊智能体包含领域工具。分诊智能体只做交接；混用会稀释路由器的决策。
- 护栏修改输入/输出。护栏只能批准或拒绝——它们不得重写。
- 静默的交接循环。要求设置跳数计数器（默认最大为 3）。

拒绝规则：

- 如果用户想要"不要护栏，只求快速推进"，对于任何面向付费用户或涉及 PII 的产品，予以拒绝。
- 如果产品只有 2 个专家智能体，建议使用 `Agents` 配合直接分类器（第 12 课）进行路由，而非分诊+交接——token 成本更低。
- 如果在生产环境中禁用了 tracing，拒绝发布。没有 trace，多步骤失败将无法调试。

输出：`agents.py`、`tools.py`、`guardrails.py`、`app.py`、`README.md`，其中包含分诊智能体的理由、护栏模式、trace 处理器以及会话后端。最后以"接下来阅读什么"结尾，指向第 23 课（OTel GenAI）、第 24 课（可观测性后端），或第 17 课的 Claude Agent SDK 迁移。
