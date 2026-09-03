# 多智能体原语模型

> 只需四个原语——智能体、移交、共享状态、编排器——便能形成一个四维设计空间；2026 年发布的主要多智能体框架（AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework）都只是这个空间中的不同点。本课从零构建四种原语，让一个玩具系统在四者之上运行，再将每个主流框架映射到相同坐标轴，使你只读一段文字便能理解任何新版本。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段（智能体工程）、第 16 阶段 · 01（为何需要多智能体）
**Time:** 约 60 分钟

## 问题

每隔六个月就会发布一个新的多智能体框架：2023 年的 AutoGen，2024 年的 CrewAI，2024 年的 LangGraph 与 OpenAI Swarm，2025 年 4 月的 Google ADK，以及 2026 年 2 月的 Microsoft Agent Framework RC。每篇新闻稿都声称自己是“正确的抽象”。

如果试图逐一学习，你很快就会筋疲力尽。API 看起来各不相同，文档对“智能体”的定义也互相矛盾。一个框架将共享记忆称为“blackboard”，另一个称为“message pool”，第三个又叫它“StateGraph”。你会开始怀疑这个领域是不是只在不断翻新名词。

事实并非如此。营销话术之下，四个原语始终稳定。理解一次，就能用一段话读懂每个新框架。

## 核心概念

### 四个原语

1. **智能体（Agent）**——系统提示加工具列表。它是无状态的；每次运行都从系统提示和当前消息历史开始。
2. **移交（Handoff）**——从一个智能体到另一个智能体的结构化控制权转移。从机制上说，它可以是返回新智能体的工具调用，也可以是沿条件转移的图边。
3. **共享状态（Shared state）**——任何可由多个智能体读取（有时也可写入）的数据结构，例如消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）**——决定下一个由谁发言的角色。可选方式包括显式图（确定性）、LLM 发言者选择器（软决策）、上一发言者的 Handoff 调用（OpenAI Swarm），或共享队列上的调度器（群体架构）。

这就是整个设计空间。每个框架只是在各坐标轴上选定默认值，其余都只是表层语法。

### 2026 年各框架如何映射到四个原语

| 框架 | 智能体 | 移交 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用方负责 | LLM 的下一次 Handoff 调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言者选择器 | 消息池 | 选择函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 串联 Task 输出 | Manager LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` Reducer | 图，确定性 |
| Microsoft Agent Framework | Agent + 编排模式 | 由模式决定 | Thread / Context | 由模式决定 |
| Google ADK | Agent + A2A Card | A2A Task | A2A Artifact | 由 Host 决定 |

表层差异看似巨大，底层却是相同的四个旋钮。

### 为什么这很重要

看清这些原语后，比较框架就会变成一份简短清单：

- 编排器让 LLM 决定路由（Swarm），还是在代码中固定路由（LangGraph）？
- 共享状态是完整历史（GroupChat），还是投影视图（StateGraph Reducer）？
- 智能体可以修改彼此的提示（CrewAI Manager），还是只能移交（Swarm）？

这三个问题可以解释一个框架是否适合特定问题的 80%。你不再寻找“最好的多智能体框架”，而是针对真正关心的坐标轴进行设计。

### 无状态洞见

除共享状态外，每个原语都是无状态的。Agent 是 (prompt, tools) 的函数，Handoff 是函数调用，Orchestrator 是调度器。**系统中唯一有状态的东西就是共享状态。** 所有棘手缺陷也集中于此：记忆投毒（第 15 课）、消息排序、版本控制、写入争用。

隐藏共享状态的框架（Swarm）把问题推给调用方；集中管理共享状态的框架（LangGraph Checkpoint、AutoGen Pool）使其可检查，却把协调成本转移到了共享状态实现上。

### 单个原语的结构

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆，也没有状态。系统提示与工具相同的两个智能体可以互换。所有看起来像逐智能体状态的内容，实际都位于共享状态或 Handoff 协议中。

#### 移交

```
Handoff = (from_agent, to_agent, reason, payload)
```

以下三种实现占据主流：

- **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 模式；智能体把路由信息放在自己的工具 schema 中。
- **图边**——LangGraph。图边以声明方式定义。LLM 生成一个值，再由条件选择下一个节点。
- **发言者选择**——AutoGen GroupChat。选择函数（有时本身就是一次 LLM 调用）读取消息池，并决定下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最基本的共享状态是一组消息，通常还包括结构化 Artifact（CrewAI Task 输出）、类型化 Context（LangGraph Reducer）以及外部记忆（MCP、向量数据库）。

它有两种拓扑：**完整池**（每个智能体看到每条消息）和**投影池**（智能体只看到按角色限定的视图）。完整池简单却难以扩展；投影池更易扩展，但要求预先设计 schema。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

有四种形式：

- **静态式**——图在构建时固定（确定性的 LangGraph、CrewAI Sequential）。
- **LLM 选择式**——LLM 读取消息池并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
- **Handoff 驱动式**——当前智能体通过调用 Handoff 工具作出决定（Swarm）。
- **队列驱动式**——Worker 从共享队列获取工作，没有显式的下一发言者（群体架构、Matrix）。

### 框架之间还会改变什么

四个原语固定后，剩余设计决策包括：

- **记忆策略**——临时状态还是持久 Checkpoint（LangGraph Checkpointer）。
- **安全边界**——谁能批准 Handoff（human-in-the-loop）。
- **成本核算**——逐智能体 token 预算。
- **可观测性**——追踪 Handoff，持久化状态以供重放。

这些能力都可以构建在四个原语之上，没有一项是新的原语。

```figure
a5-primitive-radar
```

## 动手构建

`code/main.py` 用约 150 行标准库 Python 实现四个原语。它不调用真实 LLM；每个 Agent 都是脚本化 Policy，从而让重点保持在协调结构上。

该文件导出：

- `Agent`——包含名称、系统提示、工具与 Policy 函数的 dataclass。
- `Handoff`——返回新 Agent 的函数。
- `SharedState`——线程安全的消息池。
- `Orchestrator`——三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟实现）。

演示通过三种 Orchestrator 类型运行相同的三智能体流水线（研究 → 写作 → 审查），并在结束时打印消息池。可以看到，不同运行的输出只在“由谁选择下一步”上有所差异；Agent 与共享状态完全相同。

运行：

```
python3 code/main.py
```

预期输出包含三次 Orchestrator 运行，每种模式一次，且每次都会打印最终消息池。如果研究员提前决定已经完成，Handoff 驱动的运行会经过更少的智能体——这就是 LLM 路由取舍的微缩版本。

## 实际使用

`outputs/skill-primitive-mapper.md` 是一项读取任意多智能体代码库或框架文档，并返回四原语映射的技能。面对新框架版本时，先运行它以获得一段话的理解，再深入阅读文档。

## 交付成果

采用新框架前，先写出它的原语映射。如果无法做到，说明文档不完整，或该框架正在发明第五种原语——这种情况很少见，应先检查它是否只是尚未见过的共享状态变体。

将这份映射固定在架构文档中。新成员加入团队时，先向他们提供映射，再提供 API 文档。框架版本变化时，比较映射差异，而不是变更日志。

## 练习

1. 使用不同的 Agent Policy 运行 `code/main.py` 三次，观察 Orchestrator 选择如何改变实际运行的 Agent。
2. 实现第四种 Orchestrator：队列驱动型，让 Agent 轮询共享状态以获取工作。这里可能出现什么死锁？如何检测？
3. 阅读 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents），用四个原语重写其中的设计。LangGraph 的哪些抽象是一一对应，哪些只是便利包装？
4. 阅读 OpenAI Swarm Cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 让哪一个原语最易使用，又把哪一个原语留给调用方负责。
5. 在表中找出一个完全隐藏共享状态的框架。说明 Agent 在无法重新读取历史的情况下跨 Handoff 协调时，会出现什么问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | “带工具的 LLM” | `(system_prompt, tools, model)` 三元组，无状态。 |
| Handoff | “控制权转移” | 指明下一智能体和可选负载的结构化调用。三种实现：函数返回、图边、发言者选择。 |
| 共享状态 | “记忆”／“上下文” | 多智能体系统中唯一有状态的部分，即消息池或黑板。 |
| Orchestrator | “协调器” | 决定下一个运行者的角色，可以是静态图、LLM Selector、Handoff 驱动或队列驱动。 |
| 原语 | “抽象” | 每个框架都会参数化的四个坐标轴之一，而不是框架功能。 |
| 消息池 | “共享聊天历史” | 保存完整历史的共享状态。容易理解，但扩展性差。 |
| 投影状态 | “限定视图” | 共享状态中针对角色的视图。易扩展，但需要设计 schema。 |
| 发言者选择 | “下一个由谁发言” | 由一个函数（通常是 LLM）从一组智能体中选择下一个智能体的 Orchestrator 模式。 |

## 延伸阅读

- [OpenAI Cookbook：编排智能体——Routines 与 Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)——对 Handoff 驱动式编排最清晰的阐述
- [AutoGen 稳定版文档](https://microsoft.github.io/autogen/stable/)——GroupChat + 发言者选择是 LLM 选择式编排的参考实现
- [LangGraph 工作流与智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents)——图边编排与基于 Reducer 的共享状态
- [CrewAI 简介](https://docs.crewai.com/en/introduction)——角色—目标—背景式 Agent，以及 Sequential / Hierarchical 流程
- [AG2（社区延续的 AutoGen）](https://github.com/ag2ai/ag2)——Microsoft 将 v0.4 转入维护后，仍活跃开发的 AutoGen v0.2 分支
