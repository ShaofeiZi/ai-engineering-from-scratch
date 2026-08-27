# 编排模式：Supervisor、Swarm 与 Hierarchical

> 到了 2026 年，各类框架里反复出现四种编排模式：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的建议很直接：“关键不是做出最复杂的系统，而是为你的需求搭出正确的系统。” 先从简单方案开始；只有当单代理加五种工作流模式仍然不够时，才引入更复杂的拓扑。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 25（多 Agent 辩论）
**Time:** 约 60 分钟

## 学习目标

- 说出这四种高频编排模式，并理解各自适用场景。
- 解释 2026 年 LangChain 的建议：优先用基于 tool call 的 supervision，而不是直接依赖 supervisor 库。
- 理解 Anthropic 提出的“构建适合自身需求的系统”原则，以及它如何约束拓扑选择。
- 在标准库环境下，基于同一个脚本化 LLM 实现这四种模式。

## 问题

很多团队在真正需要之前，就过早地奔向“multi-agent”。但跨框架真正反复出现的模式其实只有四种。只要你能把它们命名清楚，就能判断该选哪一种，或者干脆判断现在根本不需要任何复杂拓扑。

## 概念

### 监督者—工作者

- 一个中心路由 LLM 负责把任务分派给不同 specialist agent。
- 它需要决定：是回到自己继续思考、转交给某个 specialist，还是直接终止。
- Specialist 之间彼此不直接通信；所有路由都经过 supervisor。

典型框架映射：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 年 LangChain 的建议：** 与其依赖 `create_supervisor` 这类封装，不如直接通过 tool calls 来做 supervision。这样上下文工程控制更细，你可以明确决定每个 specialist 到底看见什么。

### 群体 / 点对点

- 多个 agent 通过共享的工具面直接 handoff。
- 没有中心路由器。
- 相比 supervisor，延迟更低，因为跳转更少。
- 但更难推理和调试，因为系统里没有单一控制点。

典型框架映射：LangGraph 的 swarm topology、OpenAI Agents SDK 的 handoffs（当所有 agent 都能互相移交时）。

### 层级式

- supervisor 管 sub-supervisor，sub-supervisor 再管 worker。
- 在 LangGraph 中通常实现为嵌套 subgraph，在 CrewAI 中表现为嵌套 crew。
- 它可以扩展到更大的 agent 群体，但代价是操作复杂度明显上升。

什么时候真的需要它：当单个 supervisor 的上下文预算已经装不下所有 specialist 描述时。

### 辩论式

- 多个 proposer 并行给出方案，再进行迭代式交叉批评，详见 Lesson 25。
- 严格说这更像一种验证模式，不完全是编排模式。
- 但在很多框架里，它确实会作为一种拓扑选择出现。

### Autonomous crews 与 deterministic flows

CrewAI 把部署方式正式区分为两种：

- **Flow**：面向确定性的事件驱动自动化，也是更推荐的生产起点。
- **Crew**：面向自主的、基于角色的协作。

这和前面的四种模式不是同一维度，但会映射到拓扑上：Flow 往往更像 supervisor 或 hierarchical，Crew 往往更像带有 LLM 路由器的 supervisor 体系。

### Anthropic 的指导原则

“在 LLM 领域，成功不在于做出最复杂的系统，而在于构建适合你需求的正确系统。”

可以按这个决策顺序来选：

1. 单代理 + 工作流模式（Lesson 12）——先从这里开始。
2. Supervisor-worker——当你已经有 2-4 个 specialist。
3. Swarm——当延迟比推理清晰度更重要。
4. Hierarchical——只有当 supervisor 的上下文预算已经失效时。
5. Debate——当准确率比成本更重要。

### 这种模式容易出错的地方

- **先想拓扑，再想问题。** 在还没搞清 multi-agent 到底解决什么之前，就先说“我们要上 multi-agent”。这是典型倒置。
- **Swarm 里来回弹跳 handoff。** A -> B -> A -> B。解决方法通常是加 hop counter。
- **假层级。** 只是因为想做出“企业级”的感觉就堆三层结构，实际上团队规模只有两层，完全可以收缩。

```figure
orchestration-pattern
```

## 动手构建

`code/main.py` 在标准库中基于一个脚本化 LLM 实现了四种模式：

- `Supervisor`：中心路由器。
- `Swarm`：点对点直接 handoff。
- `Hierarchical`：supervisor 的 supervisor。
- `Debate`：并行 proposer + critique。

四种模式都处理同一个三意图任务（refund / bug / sales），但 trace 的形状会明显不同。

运行：

```
python3 code/main.py
```

输出会展示每种模式的 trace 和 op count。Supervisor 最清晰；swarm 最短；hierarchical 最深；debate 最昂贵。

## 如何使用

- **LangGraph**：适合实现 supervisor 和 hierarchical（通过嵌套 subgraph）。
- **OpenAI Agents SDK**：适合 handoffs-as-tools 的方式，本质上偏 supervisor 形态。
- **CrewAI Flow**：适合生产中的确定性流程。
- **Custom**：适合 debate，或者你需要完全精确的控制时。

## 交付成果

`outputs/skill-orchestration-picker.md` 用来为具体任务选择拓扑，并给出实现方式。

## 练习

1. 去掉 router，把一个 supervisor-worker 改成 swarm。看看什么地方坏掉了，什么地方又变好了。
2. 给 swarm 加 hop counter：3 次 handoff 后直接拒绝。它能抓住 A->B->A 这种反复弹跳吗？
3. 为一个有 12 个 specialist 的领域建立两层 hierarchical 系统。没有嵌套时，上下文预算会先在哪一步失效？
4. 在一个更接近生产的负载上剖析这四种模式。哪个在延迟、成本、准确率、可调试性上分别占优？
5. 读 Anthropic 的《Building Effective Agents》。把你现有的生产流程逐一映射到这四种模式之一。有没有哪条流程无法清晰映射？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Supervisor-worker | “路由器 + specialist” | 中心 LLM 把任务分发给 specialist，它们彼此不直接交流 |
| Swarm | “点对点” | 通过共享工具直接 handoff，没有中心路由器 |
| Hierarchical | “supervisor 管 supervisor” | 用嵌套 subgraph 来管理大规模 agent 群体 |
| Debate | “提出方案 + 批判” | 并行 proposer，再做交叉批评（Lesson 25） |
| Tool-call-based supervision | “不用库也能做 supervisor” | 用直接 tool call 实现 supervisor，以获得更细的上下文控制 |
| Crew | “自治团队” | CrewAI 中基于角色的协作模式 |
| Flow | “确定性工作流” | CrewAI 中事件驱动的生产模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式，以及 agent 与 workflow 的区别
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical 的实现背景
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew 与 Flow 的区别
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate 模式的经典论文
