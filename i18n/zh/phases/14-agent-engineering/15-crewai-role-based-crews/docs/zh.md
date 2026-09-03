# 基于角色的 Agent 团队：角色、任务与流程

> 四个基本原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自治、基于角色的协作）和 Flows（事件驱动、确定性的流程）。CrewAI 是 2026 年这一思路的代表实现，而且它的官方文档说得很直白：“任何准备投入生产的应用，都应该先从 Flow 开始。”

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 14（Actor 模型）
**Time:** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process），并解释每个原语各自负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程，并能为不同工作负载选对形态。
- 区分 Crews（自治、角色驱动）和 Flows（事件驱动、确定性），并解释官方为何建议生产场景优先从 Flow 开始。
- 理解如何通过 `@tool` 装饰器和 `BaseTool` 子类接工具，并能判断结构化输出与自由文本的取舍。
- 说出 CrewAI 的四类记忆，以及它们各自在哪些场景真正值得引入。
- 实现一个由 researcher、writer、editor 组成的三 agent crew，生成一份简报。
- 识别 CrewAI 的三类典型失败模式：提示膨胀、manager-LLM 税、脆弱交接。

## 问题

采用多 agent 框架的团队，迟早会撞上同一堵墙。“自主协作”在演示里很吸引人，但一旦线上用户报 bug，你需要的是可重放、可定位、可解释的执行路径。财务会追问一次 crew 运行到底烧了多少 token；值班同学会追问凌晨 3 点到底是哪个 agent 卡住了。

纯 DAG 能把这些问题答得很清楚，但它又缺少 brainstorming 这类探索型任务真正需要的开放形状。

CrewAI 把这笔账讲得很诚实：Crews 适合协作式、基于角色、偏探索的问题；Flows 适合事件驱动、代码掌控、可审计的生产流程。还是同一个框架，只是提供两种形状，你要按场景选，而不是幻想一个抽象同时兼顾一切。

## 概念

### 四个原语

CrewAI 的表面其实很小，把这四个东西记住，剩下的大多只是配置细节。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。这里的 backstory 不是装饰，而是会真实影响语气、判断和停止条件的核心提示。tools 则是该 agent 可以调用的函数或适配器。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。这是可复用的工作单元。`expected_output` 是任务契约，`context` 指明要接哪些上游任务输出，`output_pydantic` 则强制输出符合结构化模型。
- **Crew。** 容器。负责承载 `agents`、`tasks`、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 等设置。
- **Process。** 执行策略，也就是 Sequential、Hierarchical、Consensus（计划中）这些运行形态。

Agent 彼此不会直接看见对方。Task 负责引用 agent。Crew 负责组织任务顺序。Process 决定由谁选择下一个任务。这就是整个心智模型。

> **已按以下版本核对** CrewAI 0.86（2026-05）。更新版本可能会重命名或合并流程类型；在依赖某一种具体形态前，请先查看 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### 顺序式、层级式与共识式

- **Sequential。** 任务按声明顺序执行。任务 N 的输出可以作为 `context` 传给任务 N+1。成本最低，也最容易预测。只要顺序本来就是固定的，就优先用它。
- **Hierarchical。** 引入一个 manager Agent，额外做一轮 LLM 路由。CrewAI 会根据你的 `manager_llm` 配置，或者默认值，生成这个 manager。manager 每一轮决定接下来由谁处理，也可以拒绝、重试或改派。只有当你真的有四个以上专家，而且顺序确实依赖前序输出时，它才值得。
- **Consensus。** 这是文档里预留的概念，计划中，但公开 API 里还没有真正落地。现在不要依赖它。

Hierarchical 会在每轮 specialist 调用之外，再多出一轮 manager 调用。一个五步任务，token 成本很容易从五次调用膨胀到六次甚至更多。只有在“路由本身依赖输出”时，才该为这笔税买单。

### Crew 与 Flow

这正是 CrewAI 在 2026 年官方文档里最强调的划分。

- **Crew。** 由 LLM 驱动的自治协作。运行时到底怎么走，由框架和模型共同决定。它适合研究、脑暴、初稿写作，以及那些“路径本身就是答案的一部分”的场景。优点是原型快，缺点是重放和测试都不容易。
- **Flow。** 事件驱动、图结构明确、由代码掌控的流程。`@start` 标记入口，`@listen(topic)` 标记某一步会在特定 topic 被发出时触发。每一步都是普通 Python（也可以内部调用一个 Crew）。

文档给出的 2026 年生产建议非常明确：先从 Flow 开始。把 `Crew.kickoff()` 作为 Flow 步骤中的一个子动作去组合。Flow 负责审计边界与确定性，Crew 负责在某一步里提供自治探索能力。要做的是组合两者，而不是二选一。

### 工具集成

给 agent 接工具主要有三种方式，遵循“能简单就别复杂”的原则。

1. **`@tool` 装饰器。** 最适合简单的一次性 helper。函数签名就是 schema，docstring 就是 LLM 看到的工具描述。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 适合有状态、需要显式 args schema、异步支持或重试逻辑的工具。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 自带了不少第一方适配器，比如 `SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`，一行 import 就能接起来。

结构化输出使用 Pydantic。把 `output_pydantic=MyModel` 挂到 Task 上，CrewAI 就会按这个模型去校验 LLM 输出，不符合时会尝试纠正或重试。它通常要和一个收紧的 `expected_output` 字符串一起使用。自由文本输出适合草稿与开放写作；而真正需要下游 Flows 消费时，结构化输出更可靠。

### 记忆钩子

CrewAI 开箱即用地提供四类记忆，而且可以同时启用。

> **已按以下版本核对** CrewAI 0.86（2026-05）。最近几个版本把所有能力都路由到统一的 `Memory` 系统中，用它来包装这四类存储。下面的概念模型仍然成立，但在更新版本里，公开类接口可能会收敛成单一 `Memory` 入口；请查看 [CrewAI memory 文档](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **Short-term。** 单次运行内的对话缓冲，运行结束就清空。
- **Long-term。** 跨运行持久化，通常落在向量数据库里，默认是 Chroma，也可替换；通过与当前任务的相似度来检索。
- **Entity。** 针对实体存储事实，例如“客户 X 使用的是企业版套餐”。它是按实体键控，而不是按相似度召回，且能跨运行保留。
- **Contextual。** 按需装配，在 agent 真正需要某段 memory 时再拉取，而不是在运行开始时全部预装。

可以通过 `memory=True` 或按类型分别配置，在 Crew 上启用这些能力。记忆系统是 CrewAI 相比更薄框架少数真正占优的地方之一；如果你用的是纯 LangGraph，这些能力通常都得自己一项项接。

### 何时适合采用基于角色的团队

- 你有三到六个具名角色，而且它们之间确实存在协作流程，比如起草、审阅、规划、头脑风暴。
- 下一步交给谁，本身就依赖 LLM 对当前结果的判断，也就是 Hierarchical 真有价值。
- 团队成员更容易阅读 `role + goal + backstory` 这类角色配置，而不是直接读一张图定义。

### 何时不适合采用

- 任务本质上是严格排序、确定性的 DAG。那就该用 LangGraph（Lesson 13），图结构才是更自然的抽象，CrewAI 的角色层反而成了摩擦。
- 延迟预算极紧，尤其追求亚秒级延迟。Hierarchical 会额外加 manager 往返；即使是 Sequential，也要反复传递 backstories 和 prior outputs。
- 其实只有单 agent 循环。那就别上框架了，Lesson 1 里那种 agent loop 加 tool registry 往往更短。

Lesson 17（Agent Framework Tradeoffs）会系统比较这些框架。简短版结论是：CrewAI 位于“协作式、基于角色”这一象限。

### 依赖形态

它不依赖 LangChain。支持 Python 3.10 到 3.13，使用 `uv`。Star 数可以直接看 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（这里采用 2026-05 的快照理解）。官方文档也记录了 AWS Bedrock 集成。至于供应商给出的“在 QA 工作负载上比 LangGraph 更快”的基准，由于没有公开方法学细节，例如数据集、硬件和评估指标，因此只能当方向性信号，不能直接当决策依据。

### 这种模式容易出错的地方

- **Backstory 引发的提示膨胀。** 如果每个 agent 都塞进 2000 词的 backstory，五个 agent 还没开始调工具就先把上下文预算烧掉了。最好把 backstory 控制在 200 词以内，也别让五个 agent 重复写一遍同样的 house style。
- **Manager-LLM 税。** Hierarchical 会在每轮 specialist 调用前多加一轮 manager LLM 调用。一个五任务 crew，调用数会从五次变成六次，而且 manager 调用还会携带完整任务列表和先前输出。除非路由确实依赖输出，否则应退回 Sequential。
- **脆弱交接。** 任务 N 的 `expected_output` 写成 “an outline”，任务 N+1 却把它当成 `context` 里的固定三段结构去解析，结果上游 LLM 实际写了四段，下游 agent 只能硬猜。解决方式是在任务 N 上加 `output_pydantic`，让任务 N+1 读结构化对象，而不是自由文本。
- **把 Crew 直接当生产系统。** 直接把自由发挥的 Crew 当成生产系统发布，没有 Flow 包裹，结果就是输出波动大、重放困难、值班时也无法比较好坏两次运行。应当用 Flow 包一层。

```figure
ae-crew-vs-flow
```

## 动手构建

`code/main.py` 用标准库模拟了这两种形态，并给出一个三 agent crew。

结构：

- `Agent`、`Task` 这两个数据类对齐 CrewAI 的基础表面。
- `SequentialCrew.kickoff(inputs)` 按声明顺序执行任务，并把上游输出穿成 `context`。
- `HierarchicalCrew.kickoff(topic)` 会引入一个 manager Agent，每轮挑选下一个 specialist，直到返回 "done"。
- `Flow` 带有 `@start` 和 `@listen(topic)` 装饰器，以及一个很小的事件循环和 trace。
- `tool(name)` 这个装饰器用来模拟 CrewAI 的 `@tool` 形态。
- `Memory` 提供 `short_term`、`long_term`、`entity` 存储；示例里的模拟相似度使用 numpy。
- 模拟 LLM 响应是按角色加输入前缀硬编码出来的字符串。不走网络，且结果可复现。

具体示例是一个 researcher、writer、editor 三人团队，为 “agent engineering 2026” 生成简报。researcher 先取回模拟 sources，writer 起草，editor 收紧。随后同一套团队逻辑又被放进一个 Flow 里，用来对比自治协作和确定性流程两种形状。

运行它：

```bash
python3 code/main.py
```

Trace 会展示：Sequential crew 如何把输出一路穿给 `context`；Hierarchical crew 里 manager 如何依次选择 researcher、writer、editor，最后给出 “done”；Flow 又如何通过显式 topics（`researched`、`drafted`、`edited`）来固定住同样的三步；工具调用如何经由 `@tool` 路由；long-term memory 如何在两次 kickoff 之间保留下来。

Crew 的 trace 是流动的；理论上 manager 可以重排顺序。Flow 的 trace 是固定的。到底选哪个，这就是本课真正要你学会的判断。

## 如何使用

- **CrewAI Flow**：生产环境优先，即使这个 Flow 里只有一步调用 `Crew.kickoff()`，它依然提供了审计边界。
- **CrewAI Crew（Sequential）**：适合顺序明确的协作式任务，尤其是一稿生成与多轮审阅。
- **CrewAI Crew（Hierarchical）**：适合顺序真由输出决定，并且你有四个或更多 specialist 的场景。
- **LangGraph**（Lesson 13）：适合显式状态机、可持久恢复、严格顺序。
- **AutoGen v0.4**（Lesson 14）：适合 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（Lesson 16）：适合 OpenAI-first、同时需要 handoff 和 guardrail 的产品。
- **Claude Agent SDK**（Lesson 17）：适合 Claude-first、需要 subagent 和 session store 的产品。

## 交付成果

`outputs/skill-crew-or-flow.md` 会帮你根据任务判断该选 Crew 还是 Flow，并生成最小实现。它会直接拒绝以下情况：没有 backstory 的 Crew、没有显式 topic 的 Flow，以及 specialist 少于三个却硬上 Hierarchical 的设计。

## 常见陷阱

- **把 backstory 当调味料。** 把 backstory 当文案味精是错的，它会真实改变输出。每个 agent 至少测试三种版本，再选一版冻结。
- **跳过 `expected_output`。** 少了每个任务的明确契约，下游任务只能接“LLM 想输出什么就输出什么”的结果。这时流程可能能跑，但审计一定不稳。
- **记忆总是常开。** 每次运行都写 long-term memory，向量数据库会持续膨胀，检索噪声也会越来越大。只有真正跨运行持久存在的事实才值得写进去。
- **Manager 提示漂移。** Hierarchical 的 manager prompt 往往是隐含的；一旦路由变怪，先把 verbose mode 打开，把 manager prompt 摊开读。
- **Crew 中的工具副作用。** Crew 里的工具可能会被 LLM 多次调用。像 POST、DELETE、支付这种带副作用的动作，应该放进 Flow step，而不是 Crew tool。

## 练习

1. 把 Sequential crew 改写成 Flow。数一数有多少接点因此失去变动性，同时观察可读性在哪些地方下降了。
2. 给 crew 加上 entity memory，让关于某个客户的事实跨多次 kickoff 持久化。验证检索是否真的只命中正确实体。
3. 实现一个 Hierarchical 流程：manager 在 writer 输出不足三段时，拒绝把任务交给 editor。观察 trace 里的重试。
4. 写一个模拟 web search 的 `BaseTool` 子类，对比它与 `@tool` 装饰器方案在 trace 形状上的差异。
5. 给 editor task 加上 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer task 故意输出一次格式错误的 JSON，观察 CrewAI 的重试行为。
6. 阅读 CrewAI 官方入门文档，把这个 toy 移植到真实 `crewai` API。标准库版本跳过了哪些保证？
7. 接入 AgentOps 或 Langfuse（Lesson 24）跑一次真实执行。对照看看标准库版本漏掉了哪些 trace。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent | “人设” | role + goal + backstory + tools |
| Task | “工作单元” | description + expected_output + assignee + optional structured output |
| Crew | “Agent 团队” | Agent、Task、Process 的容器 |
| Process | “执行策略” | Sequential / Hierarchical / Consensus（计划中） |
| Flow | “确定性工作流” | 事件驱动、代码掌控、可测试 |
| Backstory | “人设提示词” | 真实塑造语气和判断的角色背景 |
| `@tool` | “函数工具” | 把函数变成 agent 可调用工具的装饰器 |
| `BaseTool` | “类工具” | 带参数 schema、重试和异步能力的类工具 |
| Entity memory | “按实体存储的事实” | 按 customer / account / issue 这类实体维度存储的事实 |
| Long-term memory | “跨运行记忆” | 跨多次 kickoff 持续存在的向量型记忆 |
| Contextual memory | “即时检索记忆” | agent 真正需要时才拉取的记忆 |
| Manager LLM | “路由 agent” | Hierarchical 流程中挑选下一个任务的额外 LLM |
| `expected_output` | “任务契约” | 约束 agent 返回形状的任务契约字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): 基本概念与官方建议的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): 事件驱动形态、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`、`BaseTool` 与 built-in toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): short-term、long-term、entity、contextual memory
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 多 agent 什么时候值得引入，什么时候不值得
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 状态机式替代方案
