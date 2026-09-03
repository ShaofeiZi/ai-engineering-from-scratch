# 智能体状态机——图、节点与检查点

> 手写的 ReAct 循环只是一个 `while True`。把同一个循环写成显式图之后，你便可以为它创建检查点、中断执行、建立分支，并在其中进行时间回溯。智能体没有改变，改变的是它外面的运行框架。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 11 · 09（函数调用）、阶段 11 · 14（模型上下文协议）
**Time:** 约 75 分钟

## 问题

你发布了一个使用函数调用的智能体。它顺利运行了三轮，随后出了问题：模型尝试调用一个返回 500 的工具；用户在任务中途改变主意；或者智能体未经人工批准，便决定为订单退款。`while True:` 循环没有任何挂钩。你无法暂停、无法回退，也无法分出一条“如果模型选择了另一个工具会怎样”的路径。一旦把这样的系统从演示推向生产，智能体就成了黑箱：要么成功，要么失败。

看清问题后，下一步就很自然了。智能体本来就是一个状态机——系统提示词、消息历史、待执行工具调用与下一步动作共同构成其状态。把这个状态机显式表示出来：用节点表示“模型思考”“工具运行”“人工批准”，用边表示它们之间的条件转换。一旦图变得显式，运行框架就会自然获得四种能力：检查点（在步骤之间保存状态）、中断（暂停以等待人工处理）、流式传输（输出词元与中间事件）和时间回溯（退回先前状态并尝试另一条分支）。

LangGraph 是这种抽象的参考实现。它不是 LangChain 意义上的智能体框架（“这是 AgentExecutor，祝你好运”），而是拥有一等状态、一等持久化和一等中断能力的图运行时。智能体循环是你画出来的，而不是手写出来的。

## 概念

![LangGraph StateGraph：节点、边与检查点保存器](../../../../../../phases/11-llm-engineering/16-langgraph-state-machines/assets/langgraph-stategraph.svg)

一个 `StateGraph` 包含三样东西。

1. **状态。** 一个在图中流动的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回局部更新，LangGraph 再使用每个字段对应的*归约器*合并更新——需要累积的列表使用 `operator.add`，默认行为是覆盖。
2. **节点。** 形式为 `state -> partial_state` 的 Python 函数。每个节点都是一个离散步骤，例如“调用模型”“运行工具”“生成摘要”。
3. **边。** 节点之间的转换。静态边固定前往一个位置；条件边则调用 `state -> next_node_name` 路由函数，让图能够根据模型输出选择分支。

随后编译这张图。编译会固定拓扑、附加检查点保存器（可选，但生产环境不可或缺），并返回可运行对象。你使用初始状态和 `thread_id` 调用它。每个执行步骤都会保存一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四种超能力

**检查点。** 每次节点转换都会把新状态写入存储（测试可用内存，生产环境可用 Postgres/Redis/SQLite）。再次使用相同的 `thread_id` 调用图即可恢复执行，图会从暂停的位置继续。

**中断。** 使用 `interrupt_before=["human_review"]` 标记节点，执行就会在该节点运行前停止。状态会被持久化，API 向用户返回“等待批准”。之后，使用相同 `thread_id` 并传入 `Command(resume=...)` 的请求即可恢复执行。

**流式传输。** `graph.stream(state, mode="updates")` 会在状态增量产生时逐一返回。`mode="messages"` 流式输出模型节点中的大语言模型词元，`mode="values"` 则返回完整快照。你可以选择向用户界面展示哪一种。

**时间回溯。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。把任何之前的 `checkpoint_id` 传给 `graph.invoke`，即可从该处派生分支。它非常适合调试（“如果模型当时选择工具 B 会怎样？”），也适合重放生产轨迹的回归测试。

### 归约器才是关键

每个状态字段都有一个归约器。多数默认行为都没问题——新值覆盖旧值。但消息列表需要使用 `operator.add`，让新消息追加到列表，而不是替换整个列表。并行边也会通过归约器合并更新。如果两个节点都更新 `messages`，而你忘了使用 `Annotated[list, add_messages]`，后一个更新会悄无声息地覆盖前一个，让你丢失半轮对话。归约器是这个库唯一微妙之处；正确设置之后，其余部分都能自然组合。

### 四个节点组成的 ReAct 图

一个生产级 ReAct 智能体由四个节点和两条边组成：

1. `agent`——使用当前消息历史调用大语言模型，返回助手消息（其中可能包含 tool_calls）。
2. `tools`——执行上一条助手消息中的所有 tool_calls，并把工具结果作为工具消息追加。
3. 从 `agent` 出发的一条条件边：如果最后一条消息包含 tool_calls，就路由到 `tools`；否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的一条静态边。

就这些。大约 40 行代码便可获得完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → ……），同时具备检查点、中断和流式传输能力。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 允许一个节点分派多个并行子图。例如，智能体决定同时查询三个检索器。每个 `Send` 都会产生一次目标节点的并行执行，各自的输出通过状态归约器合并。这就是 LangGraph 在不直接使用线程原语的情况下表达“编排器—工作器”模式的方式。

### 子图

编译后的图可以成为另一张图中的节点。外层图只看到一个节点，内层图则拥有自己的状态与检查点。团队正是以这种方式构建监督者—工作器智能体：监督者图把用户意图路由给各领域工作器子图。

```figure
l5-state-graph-ledger
```

## 动手构建

### 第 1 步：状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让消息列表累积而不是覆盖的归约器。忘记使用它，是最常见的 LangGraph 错误。

### 第 2 步：在线程中运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个 `{node_name: state_delta}` 字典。前端可以将它们流式传给用户界面，让用户看到“智能体正在思考……调用 search_web……取得结果……正在回答”。

### 第 3 步：添加人在回路中的中断

标记一个节点，使执行在该节点运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点与线程都会跨中断持久保存。除执行期间外，没有任何内容仅存在于内存中。

### 第 4 步：通过时间回溯进行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

将 `None` 作为输入会从给定检查点重放；传入一个值，则会先把它作为更新追加到该检查点状态，再恢复执行。这样无需重新运行整段对话，就能复现一次错误的智能体运行。

### 第 5 步：换用生产级检查点保存器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

官方提供 SQLite、Redis 和 Postgres 实现。`MemorySaver` 只适用于测试；任何需要跨重启持久保存的系统都应使用真正的存储。

## 技能

> 把智能体构建成图，而不是 `while True` 循环。

使用 LangGraph 前，先花 60 秒完成设计：

1. **命名节点。** 每个离散决策或有副作用的操作都是一个节点，例如“智能体思考”“工具运行”“评审者批准”“响应流式传输”。如果连节点都无法列出，这项任务还不适合建模为智能体。
2. **声明状态。** 使用最小化的 TypedDict，并为每个列表字段设置归约器。不要把一切都塞进 `messages`；应把任务专用字段（正在执行的 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **绘制边。** 除非下一步取决于模型输出，否则使用静态边。每条条件边都需要带具名分支的路由函数。
4. **预先选择检查点保存器。** 测试使用 `MemorySaver`，其他情况使用 Postgres/Redis/SQLite。不要发布不带检查点保存器的系统——没有检查点，就无法恢复、中断或时间回溯。
5. **在工具运行之前决定中断，而不是之后。** 把审批放在进入有副作用节点的边上，才能在造成损害前取消；把验证放在离开模型的边上，才能低成本拒绝错误调用。
6. **默认采用流式传输。** 用户界面使用 `mode="updates"`；模型节点内部逐词元流式传输使用 `mode="messages"`；评估期间获取完整快照使用 `mode="values"`。

拒绝发布没有检查点保存器的 LangGraph 智能体。拒绝发布在副作用发生*之后*才中断的智能体。拒绝发布 `messages` 字段未使用 `add_messages` 作为归约器的智能体。

## 练习

1. **简单。** 使用计算器工具和 Web 搜索工具，实现上面的四节点 ReAct 图。验证对于两轮对话，`list(app.get_state_history(config))` 至少返回四个检查点。
2. **中等。** 添加一个 `planner` 节点，使其运行在 `agent` 之前，并把结构化 `plan: list[str]` 写入状态，再让 `agent` 把计划步骤标记为完成。如果 `plan` 在检查点恢复后丢失（归约器错误），则让测试失败。
3. **困难。** 构建一个监督者图，在三个子图（`researcher`、`writer`、`reviewer`）之间使用 `Send` 进行路由。每个子图都拥有自己的状态与检查点保存器。在外层图上添加 `interrupt_before=["writer"]`，让人工人员审批研究简报。确认从先前检查点进行时间回溯时，只重新运行派生出的分支。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | “LangGraph 图” | 编译之前用于添加节点与边的构建器对象。 |
| 归约器 | “字段如何合并” | 节点返回某字段更新时应用的 `(old, new) -> merged` 函数；默认为覆盖，`add_messages` 则执行追加。 |
| 线程 | “对话 ID” | 一个 `thread_id` 字符串，用于限定一次会话的全部检查点。 |
| 检查点 | “暂停的状态” | 节点转换后保存的完整图状态快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断 | “暂停等待人工处理” | `interrupt_before` / `interrupt_after` 在节点边界停止执行；使用 `Command(resume=...)` 恢复。 |
| 时间回溯 | “从先前步骤派生分支” | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | “并行子图分派” | 节点可以返回的一种构造器，用于派生目标节点的 N 次并行执行。 |
| 子图 | “作为节点的已编译图” | 在另一张图中作为节点使用的已编译 StateGraph；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、归约器、检查点保存器与中断的权威参考。
- [LangGraph 概念：状态、归约器与检查点保存器](https://langchain-ai.github.io/langgraph/concepts/low_level/)——本课所用心智模型，直接来自官方资料。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/)——关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细说明。
- [LangGraph 人在回路](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)——`interrupt_before`、`interrupt_after`、`Command(resume=...)` 与编辑状态模式。
- [Yao 等，“ReAct: Synergizing Reasoning and Acting in Language Models”（ICLR 2023）](https://arxiv.org/abs/2210.03629)——每个 LangGraph 智能体都在实现的模式；阅读本文可理解推理轨迹的设计依据。
- [Anthropic——Building effective agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——应在何时选择链式、路由、编排器—工作器、评估器—优化器等图结构。
- 阶段 11 · 09（函数调用）——每个 LangGraph 智能体节点都会复用的工具调用原语。
- 阶段 11 · 14（模型上下文协议）——通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现机制。
- 阶段 11 · 17（智能体框架取舍）——何时应选择 LangGraph，而不是 CrewAI、AutoGen 或 Agno。
