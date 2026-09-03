"""仿 LangGraph 的标准库有状态图，支持检查点与恢复。

状态是一个带类型的字典。节点返回更新字典。运行时在每个节点之后
序列化状态，因此恢复时会从上次中断的位置精确继续。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable


State = dict[str, Any]
Update = dict[str, Any]
NodeFn = Callable[[State], Update]
Router = Callable[[State], str]


START = "__start__"
END = "__end__"


@dataclass
class Edge:
    src: str
    dst: str
    router: Router | None = None


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, list[Edge]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def set_entry(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, []).append(Edge(src=src, dst=dst))

    def add_conditional_edges(self, src: str, router: Router,
                              targets: dict[str, str]) -> None:
        for value, dst in targets.items():
            self.edges.setdefault(src, []).append(
                Edge(src=src, dst=dst, router=_make_router(router, value))
            )

    def _next(self, current: str, state: State) -> str | None:
        for edge in self.edges.get(current, []):
            if edge.router is None or edge.router(state):
                return edge.dst
        return None


def _make_router(router: Router, expected: str) -> Router:
    def fn(state: State) -> bool:
        return router(state) == expected
    return fn


class InMemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, list[tuple[str, State]]] = {}

    def save(self, session_id: str, step_name: str, state: State) -> None:
        self._store.setdefault(session_id, []).append((step_name, copy.deepcopy(state)))

    def load_latest(self, session_id: str) -> tuple[str, State] | None:
        history = self._store.get(session_id, [])
        if not history:
            return None
        return history[-1]

    def history(self, session_id: str) -> list[tuple[str, State]]:
        return list(self._store.get(session_id, []))


class PausedAtNode(Exception):
    def __init__(self, node: str, state: State) -> None:
        super().__init__(node)
        self.node = node
        self.state = state


class Runner:
    def __init__(self, graph: StateGraph,
                 checkpointer: InMemoryCheckpointer) -> None:
        self.graph = graph
        self.checkpointer = checkpointer

    def run(self, session_id: str, initial_state: State,
            resume_from: str | None = None,
            state_override: State | None = None) -> State:
        if state_override is not None:
            state = copy.deepcopy(state_override)
        else:
            state = copy.deepcopy(initial_state)
        current = resume_from or self.graph.entry
        if current is None:
            raise RuntimeError("no entry node set")
        while current is not None and current != END:
            fn = self.graph.nodes.get(current)
            if fn is None:
                raise RuntimeError(f"unknown node {current!r}")
            update = fn(state)
            if update is None:
                update = {}
            state = {**state, **update}
            self.checkpointer.save(session_id, current, state)
            if state.get("_pause_reason"):
                reason = state.pop("_pause_reason")
                raise PausedAtNode(current, state)
            current = self.graph._next(current, state)
        return state


def _classify(state: State) -> Update:
    text = state["input"].lower()
    if "refund" in text or "money back" in text:
        route = "refund"
    elif "crash" in text or "bug" in text or "error" in text:
        route = "bug"
    elif "pricing" in text or "quote" in text:
        route = "sales"
    else:
        route = "sales"
    return {"route": route, "step": state.get("step", 0) + 1}


def _refund(state: State) -> Update:
    return {"ticket": f"REF-{state.get('input', '')[:12]}",
            "step": state.get("step", 0) + 1}


def _bug(state: State) -> Update:
    return {"ticket": f"BUG-{state.get('input', '')[:12]}",
            "step": state.get("step", 0) + 1}


def _sales(state: State) -> Update:
    return {"ticket": f"SAL-{state.get('input', '')[:12]}",
            "step": state.get("step", 0) + 1}


def _human_gate(state: State) -> Update:
    if not state.get("human_approval"):
        return {"_pause_reason": "awaiting human approval",
                "step": state.get("step", 0) + 1}
    return {"step": state.get("step", 0) + 1}


def _send(state: State) -> Update:
    return {"output": f"sent {state.get('ticket')}",
            "step": state.get("step", 0) + 1}


def build_graph() -> StateGraph:
    graph = StateGraph()
    graph.add_node("classify", _classify)
    graph.add_node("refund", _refund)
    graph.add_node("bug", _bug)
    graph.add_node("sales", _sales)
    graph.add_node("human_gate", _human_gate)
    graph.add_node("send", _send)
    graph.set_entry("classify")

    graph.add_conditional_edges(
        "classify",
        router=lambda s: s["route"],
        targets={"refund": "refund", "bug": "bug", "sales": "sales"},
    )
    graph.add_edge("refund", "human_gate")
    graph.add_edge("bug", "human_gate")
    graph.add_edge("sales", "human_gate")
    graph.add_edge("human_gate", "send")
    graph.add_edge("send", END)
    return graph


def main() -> None:
    print("=" * 70)
    print("LANGGRAPH 状态机 — 第 14 阶段，第 13 课")
    print("=" * 70)

    graph = build_graph()
    ckpt = InMemoryCheckpointer()
    runner = Runner(graph, ckpt)

    session = "s001"
    initial: State = {"input": "the CLI crashes on ctrl-c, please fix",
                      "step": 0, "human_approval": False}

    print("\n首次运行（将在 human_gate 处暂停）")
    try:
        final = runner.run(session, initial)
        print(f"  最终结果: {final}")
    except PausedAtNode as paused:
        print(f"  PAUSED 位于 {paused.node}")
        print(f"  暂停时的状态: {json.dumps(paused.state, default=str)}")

    print("\n检查点历史记录")
    for node, snap in ckpt.history(session):
        print(f"  {node}  route={snap.get('route')}  "
              f"ticket={snap.get('ticket')}  step={snap.get('step')}")

    print("\n人工审批通过；从 human_gate 之后的下一个节点恢复")
    latest = ckpt.load_latest(session)
    assert latest is not None
    last_node, last_state = latest
    approved_state = {**last_state, "human_approval": True}
    approved_state.pop("_pause_reason", None)
    ckpt.save(session, f"{last_node}_reviewed", approved_state)

    final = runner.run(
        session_id=session,
        initial_state=initial,
        resume_from="send",
        state_override=approved_state,
    )
    print(f"  最终结果: {final}")

    print()
    print("特性说明：状态在每个节点之后都会序列化；恢复是精确的。")
    print("第 38 步失败后不会重新执行；从第 39 步继续。")


if __name__ == "__main__":
    main()
