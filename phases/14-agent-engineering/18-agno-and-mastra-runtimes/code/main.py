"""并排玩具示例：仿 Agno（无状态 FastAPI）对比仿 Mastra
（primitive-rich）。仅使用标准库；旨在展示结构上的差异。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgnoAgent:
    name: str
    fn: Callable[[str], str]

    def run(self, prompt: str) -> str:
        return self.fn(prompt)


class AgnoSession:
    def __init__(self) -> None:
        self._turns: dict[str, list[str]] = {}

    def append(self, session_id: str, turn: str) -> None:
        self._turns.setdefault(session_id, []).append(turn)

    def history(self, session_id: str) -> list[str]:
        return list(self._turns.get(session_id, []))


def agno_request_handler(session: AgnoSession,
                         agent: AgnoAgent,
                         session_id: str,
                         prompt: str) -> str:
    start = time.perf_counter_ns()
    session.append(session_id, f"user: {prompt}")
    output = agent.run(prompt)
    session.append(session_id, f"assistant: {output}")
    elapsed_us = (time.perf_counter_ns() - start) / 1000
    return f"{output}  (handler {elapsed_us:.1f} us)"


@dataclass
class MastraTool:
    name: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]


@dataclass
class MastraAgent:
    name: str
    instructions: str
    tools: list[MastraTool] = field(default_factory=list)

    def run(self, prompt: str, tool_calls: list[tuple[str, dict[str, Any]]]
            ) -> tuple[str, list[tuple[str, str]]]:
        trace: list[tuple[str, str]] = []
        for tool_name, args in tool_calls:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool is None:
                trace.append((tool_name, "error: unknown"))
                continue
            result = tool.fn(**args)
            trace.append((tool_name, result))
        output = f"{self.name} processed {len(tool_calls)} tools"
        return output, trace


@dataclass
class MastraWorkflow:
    steps: list[tuple[str, Callable[[Any], Any]]]

    def run(self, payload: Any) -> list[tuple[str, Any]]:
        trace: list[tuple[str, Any]] = []
        current = payload
        for name, fn in self.steps:
            current = fn(current)
            trace.append((name, current))
        return trace


def _agno_agent_fn(prompt: str) -> str:
    return f"[agno reply] {prompt[:40]}"


def _mastra_tool_fn(query: str) -> str:
    return f"[mastra search result for {query!r}]"


def main() -> None:
    print("=" * 70)
    print("AGNO 对比 MASTRA — 第 14 阶段，第 18 课")
    print("=" * 70)

    print("\n1. AGNO-shaped（无状态 session-scoped FastAPI 处理器）")
    session = AgnoSession()
    agent = AgnoAgent(name="agno_a", fn=_agno_agent_fn)
    for i in range(3):
        out = agno_request_handler(session, agent, "s001",
                                   f"query {i}: how do I ship an agent")
        print(f"  轮次 {i}：{out}")
    print(f"  会话历史长度：{len(session.history('s001'))}")
    print("  模式：每个请求使用新的智能体；session 保存状态；"
          "FastAPI 保持无状态。")

    print("\n2. MASTRA-shaped（Agents + Tools + Workflows）")
    search_tool = MastraTool(
        name="search",
        input_schema={"type": "object",
                      "properties": {"query": {"type": "string"}}},
        fn=_mastra_tool_fn,
    )
    mastra_agent = MastraAgent(
        name="mastra_a",
        instructions="search, summarize, cite",
        tools=[search_tool],
    )
    output, trace = mastra_agent.run(
        "research agent engineering",
        [("search", {"query": "agent engineering 2026"}),
         ("search", {"query": "BFCL V4 benchmarks"})],
    )
    print(f"  智能体输出：{output}")
    for tool, result in trace:
        print(f"    工具 {tool}：{result}")

    workflow = MastraWorkflow(steps=[
        ("normalize", lambda p: p.strip().lower()),
        ("search", lambda p: f"found 3 results for {p}"),
        ("summarize", lambda p: f"summary: {p}"),
    ])
    print("\n  工作流运行")
    for name, out in workflow.run("  Agent Engineering 2026  "):
        print(f"    {name}：{out}")

    print("\n按技术栈选择：python+fastapi  Agno；typescript+next/vercel  Mastra。")


if __name__ == "__main__":
    main()
