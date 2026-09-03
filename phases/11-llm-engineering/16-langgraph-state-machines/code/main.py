"""带检查点、中断和时间旅行的最小 LangGraph ReAct 代理。

使用 Anthropic API 密钥（`ANTHROPIC_API_KEY`）运行。该代理配有两个
示例工具，并会：

1. 构建一个使用 `add_messages` 作为消息列表 reducer 的四节点状态图
   （agent -> tools -> agent）。
2. 使用内存检查点编译图，并在 `tools` 节点前设置中断，以便在产生
   任何副作用之前暂停。
3. 运行两轮对话并流式输出更新事件。
4. 在第一次工具调用前暂停，检查待执行的工具调用，再使用
   `Command(resume=True)` 恢复。
5. 打印检查点历史，并演示从较早检查点分叉的时间旅行。

安装：
pip install "langgraph>=0.2.50" "langchain-anthropic>=0.3.0"

运行：
python main.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


# 状态


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 工具


@tool
def calculator(expression: str) -> str:
    """评估一个Python 算术表达式, 如“ 2 + 2 * 3” 。 返回
由于字符串。"""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "ERROR: only digits and + - * / ( ) are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"ERROR: {exc!r}"


@tool
def web_lookup(query: str) -> str:
    """假网络搜索. 返回已知查询和“未知”的罐装事实
否则 准备一个真正的检索工具 。"""
    facts = {
        "anthropic headquarters": "Anthropic is headquartered in San Francisco, California.",
        "python release year": "Python was first released in 1991.",
    }
    return facts.get(query.strip().lower(), "unknown")


TOOLS = [calculator, web_lookup]


# 图


def build_app() -> tuple:
    """连接四节点 ReAct 图并返回编译后的 app 和已绑定工具的 LLM。"""
    llm = ChatAnthropic(model=os.environ.get("LLM_MODEL", "claude-sonnet-4-5"), temperature=0).bind_tools(TOOLS)

    def agent_node(state: State) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    app = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )
    return app, llm


# 驱动程序


def pretty(msg: AnyMessage) -> str:
    kind = msg.__class__.__name__
    content = msg.content if isinstance(msg.content, str) else str(msg.content)[:200]
    tool_calls = getattr(msg, "tool_calls", None) or []
    tcs = " | ".join(f"{t['name']}({t['args']})" for t in tool_calls)
    return f"[{kind}] {content} {('-> ' + tcs) if tcs else ''}".strip()


def run() -> None:
    app, _llm = build_app()
    config = {"configurable": {"thread_id": "demo-42"}}

    # 第一轮：提出一个需要调用 web_lookup 的问题。
    user = HumanMessage("Where is Anthropic headquartered?")
    for event in app.stream({"messages": [user]}, config, stream_mode="updates"):
        for node, update in event.items():
            print(f"<<{node}>>")
            for m in update.get("messages", []):
                print("   ", pretty(m))

    # 我们现在暂停在中断前。
    pending = app.get_state(config)
    print("\n已中断，待执行的工具调用：")
    for m in pending.values["messages"][-1:]:
        for tc in getattr(m, "tool_calls", []) or []:
            print(f"  - {tc['name']}({tc['args']})")

    # 批准并恢复执行。
    for event in app.stream(Command(resume=True), config, stream_mode="updates"):
        for node, update in event.items():
            print(f"<<{node}>>")
            for m in update.get("messages", []):
                print("   ", pretty(m))

    # 检查点历史。
    history = list(app.get_state_history(config))
    print(f"\n检查点历史：{len(history)} 个快照")
    for i, snap in enumerate(history):
        last = snap.values["messages"][-1] if snap.values.get("messages") else None
        tag = last.__class__.__name__ if last else "?"
        print(f"  {i:>2}  {tag:<15}  next={snap.next}")

    # 时间旅行：从最早的快照分叉，然后提出另一个问题。
    if len(history) >= 3:
        earliest = history[-1].config
        print("\n时间旅行：从最早的检查点分叉并询问数学问题。")
        fork = {"messages": [HumanMessage("What is 17 * 23?")]}
        for event in app.stream(fork, earliest, stream_mode="updates"):
            for node, update in event.items():
                print(f"<<{node}>>")
                for m in update.get("messages", []):
                    print("   ", pretty(m))
        # 越过数学工具调用前的中断并继续执行。
        for event in app.stream(Command(resume=True), earliest, stream_mode="updates"):
            for node, update in event.items():
                print(f"<<{node}>>")
                for m in update.get("messages", []):
                    print("   ", pretty(m))


if __name__ == "__main__":
    run()
