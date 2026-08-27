# 智能体框架取舍——图、角色与 Actor 编排

> 每个框架展示的都是同一种演示（研究智能体生成报告），隐藏的也是同一种问题（状态 Schema 与编排层相互冲突）。应选择核心抽象与问题形态相匹配的框架；否则，你只会把胶水代码写上两遍。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 11 · 09（函数调用）、阶段 11 · 16（LangGraph）
**Time:** 约 45 分钟

## 问题

你有一项需要多次调用大语言模型的任务。它可能是一条研究工作流（规划、搜索、总结、引用），可能是一条代码审查流水线（解析 diff、批判、修补、验证），也可能是一个多轮助手，负责预订航班、撰写电子邮件和提交费用报表。于是，你选择了一个框架。

三天后，你发现框架的抽象开始泄漏。CrewAI 提供了角色，却会在“研究员”需要把结构化计划交给“作者”时处处掣肘。AutoGen 提供智能体之间的对话，却没有一等状态，因此检查点只能是对话日志的 pickle。LangGraph 提供状态图，却迫使你在还不知道智能体会做什么时，就预先为每条转换命名。Agno 提供单智能体抽象，可一旦尝试扇出到三个并发工作器，就显得格格不入。

解决方案不是“挑选最好的框架”，而是让框架的核心抽象匹配问题的形态。本课会画出这张地图。

## 概念

![智能体框架矩阵：核心抽象与问题形态](../assets/framework-matrix.svg)

四个框架主导着 2026 年的生态，它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`——类型化状态、节点、条件边、检查点保存器。 | 拥有显式状态和人在回路中断的工作流；需要时间回溯调试的生产级智能体。 | 拓扑未知、松散且由角色驱动的头脑风暴。 |
| **CrewAI** | `Crew`——角色（目标、背景故事）、任务、流程（顺序或分层）。 | 具有短小线性/分层计划的角色扮演或角色驱动工作流。 | 超出团队轮次历史的有状态任务；复杂分支。 |
| **AutoGen** | `ConversableAgent` 样本对——两个或更多智能体轮流对话，直至满足退出条件。 | 多智能体*对话*（教师—学生、提议者—批评者、执行者—评审者），思考过程从聊天中涌现。 | 具有已知 DAG 的确定性工作流；需要跨重启保存持久状态的任何任务。 |
| **Agno** | `Agent`——单个大语言模型 + 工具 + 记忆，可组合成团队。 | 快速构建单智能体与轻量团队；多模态能力强，内置存储驱动。 | 带有自定义归约器、分支很深且显式定义的图。 |

### “抽象”的真正含义

框架的核心抽象，就是你讲解架构时会画在白板上的东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转换，每个位置的状态对象都带类型。其心智模型是状态机。
- **CrewAI** → 你画一张组织结构图。每个角色都有岗位说明，由管理者分派任务。其心智模型是一支小型专家团队。
- **AutoGen** → 你画一个 Slack 私聊。两个智能体互发消息；需要主持人时，再让第三个智能体加入。其心智模型是聊天。
- **Agno** → 你画一个挂着工具的方框。把多个方框并排放置，就组成团队。其心智模型是“开箱即用的智能体”。

### 状态问题

状态是大多数框架选型在生产环境中失败的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、逐字段归约器、一等检查点保存器（SQLite/Postgres/Redis）。恢复、中断和时间回溯开箱即用。*（参见阶段 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段以字符串形式在任务间流动，也可以通过 `output_pydantic` 形成结构。它没有开箱即用的持久团队存储；如果团队需要跨重启存活，你必须自行添加。
- **AutoGen。** 状态就是聊天历史和用户定义的 `context`。对话记录可以持久化；任意工作流状态则不行，除非自行编写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）附加到一个 `Agent`，具体通过 `storage=` 配置——对话会话与用户记忆会自动持久化。它是会话存储，而不是完整的图检查点保存器。

### 分支问题

每个非平凡智能体都会产生分支。由谁决定分支，至关重要。

- **LangGraph**——由开发者通过条件边决定。路由是带具名分支的 Python 函数。分支是已编译图中的一等对象，检查点保存器会记录走了哪条分支。
- **CrewAI**——在分层模式下由管理者决定，在顺序模式下则由开发者在构建时决定。路由隐含在任务列表中；除了管理者的提示词外，没有一等的“if”。
- **AutoGen**——由智能体通过聊天决定。分支从下一个发言者中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认由大语言模型驱动。
- **Agno**——智能体通过下一步调用哪个工具来决定。团队提供 coordinator/router/collaborator 模式，更复杂的分支由开发者负责。

### 可观测性问题

- **LangGraph**——通过 LangSmith 或任意 OTel 导出器使用 OpenTelemetry。每次节点转换都是一个追踪 Span；检查点也可作为可重放轨迹。LangSmith 是第一方方案，Langfuse/Phoenix 也有适配器。
- **CrewAI**——自 2025 年末起原生支持 OpenTelemetry；可集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen**——通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 提供连接器。追踪粒度是每条智能体消息，而不是每个节点。
- **Agno**——内置 `monitoring=True` 开关与 OpenTelemetry 导出器；与 Langfuse 紧密集成，可记录会话轨迹。

### 成本与延迟

四种框架都会增加每次调用的开销（框架逻辑、验证、序列化）。开销大致从低到高依次为：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架额外执行了多少大语言模型路由。CrewAI 的分层管理者要消耗词元来决定下一位执行者；AutoGen 的 `GroupChatManager` 也是如此。LangGraph 只在你显式编写 `llm.invoke` 的地方消耗词元；Agno 的单智能体路径则很轻。

当单次运行成本很重要时，应优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`），而不是由大语言模型选择路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器和大语言模型。提供一等 MCP 适配器（将工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具与 MCP 工具都可适配接入。通过 `allow_delegation=True` 实现团队间委派。
- **AutoGen** → `FunctionTool` 可以包装任意 Python 可调用对象；提供 MCP 适配器。其智能体间模式与 AG2 生态紧密耦合。
- **Agno** → 使用 `@tool` 装饰器或 BaseTool 子类；提供 MCP 适配器；工具可由多个智能体与团队共享。

## 技能

> 对于给定的智能体问题，你能用一句话说明为什么某个框架最合适。

构建前检查清单：

1. **画出形态。** 它是一张图（类型化状态、具名转换）？一场角色扮演（专家相互交接工作）？一段聊天（智能体对话直至完成）？还是一个带工具的单智能体？
2. **决定由谁分支。** 开发者决定 → LangGraph；管理者智能体决定 → CrewAI 分层模式；聊天涌现 → AutoGen；工具调用决定 → Agno。
3. **检查状态需求。** 是否需要从检查点恢复、时间回溯或运行中等待人工介入？如果需要，默认选择 LangGraph；Agno 会话可满足对话范围内的状态需求。
4. **检查成本预算。** 大语言模型选择路由会在每轮消耗额外词元。如果智能体每天运行数千次，应优先选择显式路由。
5. **计算框架开销。** 每个框架都会增加一项依赖。如果任务只有两次大语言模型调用和一个工具，就写 30 行纯 Python；没有任何框架比“不用框架”更便宜。

在能够画出图、组织结构图、聊天关系或智能体方框之前，不要贸然选择框架。不要选择会迫使你对抗其状态模型、才能实现真实需求的框架。

## 决策矩阵

| 问题形态 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 带类型化状态、人工审批且长期运行的工作流 DAG | LangGraph | 一等状态、检查点保存器、中断与时间回溯。 |
| 具有不同角色的研究/写作流水线 | CrewAI（顺序模式）或 LangGraph 子图 | 在 CrewAI 中表达“一项任务一个角色”的成本很低；分支变复杂后使用 LangGraph 扩展。 |
| 提议者—批评者或教师—学生对话 | AutoGen | 双智能体聊天正是它的原生形态。 |
| 带工具、会话和记忆的单智能体 | Agno | 配置最精简，内置存储与记忆。 |
| 数千个带归约器的并行扇出 | LangGraph + `Send` | 唯一拥有一等并行分派 API 的框架。 |
| 快速原型，不希望绑定框架 | 纯 Python + 提供商 SDK | 没有框架才是最快的框架。 |

```figure
l5-framework-fit
```

## 练习

1. **简单。** 使用 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor），分别实现同一项任务——“研究 Anthropic 总部，撰写一份 200 词简报并引用来源”。报告每次运行的词元成本与代码行数。
2. **中等。** 使用 AutoGen（researcher ↔ writer 对话，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools`、`write_tools` 与会话存储的单智能体）构建同一任务。根据以下维度为四种实现排序：（a）单次运行成本；（b）崩溃后恢复的能力；（c）在写作步骤前插入人工审批的能力。
3. **困难。** 构建决策树脚本 `pick_framework.py`，接收一段简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回框架建议与一句话理由。使用自行设计的六个案例验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 编排 | “智能体如何协调” | 决定下一个运行哪个节点、角色或智能体的层。 |
| 持久状态 | “重启后恢复” | 进程终止后仍然存在、附着于检查点或会话存储的状态。 |
| 大语言模型选择路由 | “让模型决定” | 规划模型每轮选择下一步；灵活，但每次决策都要消耗词元。 |
| 显式路由 | “开发者决定” | 由 Python 函数或静态边选择下一步；成本低、可审计。 |
| Crew | “CrewAI 团队” | 绑定成一个可运行对象的角色 + 任务 + 流程（顺序或分层）。 |
| GroupChat | “AutoGen 多智能体聊天” | 由发言者选择器管理的 N 个智能体之间的对话。 |
| Team（Agno） | “Agno 多智能体” | 在一组智能体上采用路由、协调或协作模式。 |
| StateGraph | “LangGraph 图” | 类型化状态、节点、条件边与检查点保存器的抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、检查点保存器、中断与时间回溯。
- [CrewAI 文档](https://docs.crewai.com/)——Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/)——ConversableAgent、GroupChat、团队与工具。
- [Agno 文档](https://docs.agno.com/)——Agent、Team、Workflow、存储与记忆。
- [Anthropic——Building effective agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——与框架无关的模式库（提示链、路由、并行化、编排器—工作器、评估器—优化器）。
- [Yao 等，“ReAct: Synergizing Reasoning and Acting”（ICLR 2023）](https://arxiv.org/abs/2210.03629)——每个框架都加以包装的循环。
- [Wu 等，“AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation”（2023）](https://arxiv.org/abs/2308.08155)——AutoGen 的设计论文。
- [Park 等，“Generative Agents: Interactive Simulacra of Human Behavior”（UIST 2023）](https://arxiv.org/abs/2304.03442)——CrewAI 风格角色栈所依据的角色扮演基础。
- 阶段 11 · 16（LangGraph）——本课用于基准比较的框架。
- 阶段 11 · 19（Reflexion）——一种很容易映射到 LangGraph、却难以映射到 CrewAI 的模式。
- 阶段 11 · 22（生产可观测性）——如何为所选框架接入可观测性插桩。
