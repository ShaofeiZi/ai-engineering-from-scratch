# 有状态图编排：持久执行与检查点

> Agent 可以被建模为状态机：节点是函数，边是状态转移，状态会在每个节点后落盘成检查点。任何失败都可以从最后一个成功检查点恢复。到 2026 年，LangGraph 已经成为这种低层有状态编排模型的代表性参考实现。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 01（Agent Loop），第 14 阶段 · 12（工作流模式）
**Time:** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带类型的状态机、函数节点、条件边，以及节点后检查点。
- 说出官方文档强调的四项能力：durable execution、streaming、human-in-the-loop、comprehensive memory。
- 解释 LangGraph 支持的三种编排拓扑：supervisor、peer-to-peer（swarm）和 hierarchical（nested subgraphs）。
- 用 stdlib 实现一个带 typed state、conditional edges 和 checkpoint/resume 周期的状态图。

## 问题

Agent 和 workflow 共享同一个现实问题：一条 40 步的执行链，如果在第 38 步失败，你想要的是从第 38 步继续，而不是整条流程从头再来。把状态当成二等公民的框架，最后都会逼着运维在“默认假设每次都是新运行”的库外面硬补重试逻辑。

LangGraph 的设计回答很直接：状态是第一类、带类型的对象；状态变更必须显式表达；并且每个节点结束后都要持久化检查点。恢复不是补丁，而是一次标准的 `load_state(session_id)` 调用。

## 概念

### 图

一个图由以下部分定义：

- **State type。** 一个 typed dict（或 Pydantic model），所有节点都从它读取并修改它。
- **Nodes。** 纯函数，形如 `(state) -> state_update`。函数返回后，更新会被合并回总状态。
- **Edges。** 节点之间的直接转移或条件转移。
- **Entry and exit。** `START` 与 `END` 这两个哨兵节点标记执行边界。

例如，一个包含 `classify`、`refund`、`bug`、`sales`、`done` 这些节点的 agent，本质上就是一个以图表示的 routing workflow。

### 持久执行

每个节点返回后，运行时都会把当前状态序列化并写入 checkpointer，比如 SQLite、Postgres、Redis，或者自定义后端。假设在第 N 步失败，运行时就可以通过 `resume(session_id)` 从第 N+1 步继续，而且使用的是失败前精确保存下来的状态。

LangGraph 文档明确把这点当作核心卖点，并列举了 Klarna、Uber、J.P. Morgan 这类生产用户。它真正有价值的不只是“图”这种结构，而是“图 + 检查点”让恢复成本变得足够低。

### 流式输出

每个节点都可以产生部分输出。图会把每个节点的 delta 事件持续流给调用方，因此 UI 可以随着图的执行过程实时更新，而不是只能等最终结果。

### 人在回路

可以在节点之间查看并修改状态。典型实现方式是：在一个关键节点前暂停，把当前状态展示给人工，允许人工修改后再恢复执行。由于状态本来就已经被序列化保存，这个能力做起来会很自然。

### 记忆

既包括短期记忆，也就是单次运行中的对话历史保存在状态里；也包括长期记忆，也就是跨运行持久化的数据，通常由 checkpointer 和独立的长期存储共同承担。LangGraph 还可以通过工具接入外部记忆系统，例如 Mem0 或自定义 memory backend。

### 三种拓扑

1. **Supervisor。** 一个中心化的 router LLM 负责把任务分发给专业子 agent。API 形态上对应 `create_supervisor()` 以及 `langgraph-supervisor`，不过 LangChain 团队在 2026 年更建议直接通过 tool calls 来实现，以获得更强的上下文控制。
2. **Swarm / peer-to-peer。** Agent 之间通过共享工具面直接 handoff，没有中央路由器。
3. **Hierarchical。** 由 supervisor 管理 sub-supervisor，本质上就是嵌套子图。

### 这种模式会在哪些地方出错

- **检查点过小。** 如果你只保存对话轮次，而没保存工具状态和 memory 写入，那恢复时这些副作用就丢了。必须序列化完整状态。
- **节点不具备确定性。** Resume 默认假设给定节点输入会产生相同的状态更新。随机种子、墙上时钟、外部 API 响应这类信息都要被显式捕获。
- **条件边滥用。** 如果图里几乎每条边都是条件边，那它就会退化成一个根本无法推理的状态机。应优先使用线性链路，只在必要位置分支。

```figure
langgraph-state
```

## 动手构建

`code/main.py` 实现了一个 stdlib 版本的状态图：

- `State`：一个 typed dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node`：接收状态并返回 update dict 的可调用对象。
- `StateGraph`：封装 nodes、edges、conditional edges、run 与 resume。
- `SQLiteCheckpointer`（这里是一个 in-memory fake）：在每个节点后序列化状态；`load(session_id)` 可以恢复状态。
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行方式：

```
python3 code/main.py
```

trace 会展示第一次运行如何在 human gate 处失败、状态如何被持久化，以及恢复之后如何产出最终输出。

## 如何使用

- **LangGraph**：当前最成熟、最适合生产的参考实现。你可以用 `create_react_agent`、`create_supervisor`，也可以自己组图。
- **AutoGen v0.4**（Lesson 14）：适合高并发场景的 actor model 替代方案。
- **Claude Agent SDK**（Lesson 17）：自带 session store 的托管式 harness。
- **Custom**：当你需要完全掌控状态结构或 checkpointer 后端时。

## 交付成果

`outputs/skill-state-graph.md` 会在任意目标 runtime 里生成一个 LangGraph 风格的 state graph，并把 checkpoint 与 resume 一并接好。

## 练习

1. 从 `classify` 加一条条件边到 `end`：当分类置信度低于阈值时直接结束。然后让人工手动设置 `route`，再恢复执行。
2. 把这个 SQLite-like fake 换成真正的 SQLite checkpointer，并测量每一步的序列化开销。
3. 实现并行边：两个节点同时运行，再由一个自定义 reducer 合并。这里 immutable state 带来了什么好处？
4. 阅读 `langgraph-supervisor` 参考文档，把这个 toy 移植到 `create_supervisor`。比较两者 trace 的形状。
5. 加入 streaming：每个节点运行时都能持续产出部分状态，并实时打印这些 delta。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| State graph | "智能体作为状态机" | 带类型状态 + 节点 + 边 + reducer |
| Checkpointer | "Persistence backend" | 每个节点后序列化状态；支持 resume |
| Reducer | "State merger" | 把当前状态与节点更新合并起来的函数 |
| Conditional edge | "Branch" | 由状态函数决定选择哪条边 |
| Subgraph | "Nested graph" | 作为另一个图中节点使用的图 |
| Durable execution | "从故障处恢复" | 从最后一个成功节点，以精确状态继续执行 |
| Supervisor | "Router LLM" | 专业子 agent 的中央调度者 |
| Swarm | "P2P agents" | Agent 通过共享工具 handoff；没有中央路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 官方参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor 模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model 替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store 与 subagents
