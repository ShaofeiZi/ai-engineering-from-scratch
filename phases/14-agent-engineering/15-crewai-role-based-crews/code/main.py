"""仿 CrewAI 的标准库 Crew 和 Flow 原语。

Three-agent crew（研究员、撰写者、编辑）生成关于
"agent engineering 2026" 的简报。同一个 crew 分别以 Sequential、Hierarchical
以及通过 Flow 方式运行，展示三种执行形态。

标准库 + numpy。模拟的 LLM 响应为确定性硬编码字符串，
以 agent 角色和输入前缀为键。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def tool(name: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """镜像 CrewAI 的 @tool 装饰器。将一个函数标记为
    Agent 可调用的工具。文档字符串即描述；签名即模式。"""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        fn.tool_name = name  # type: ignore[attr-defined]
        fn.is_tool = True  # type: ignore[attr-defined]
        return fn

    return decorator


@tool("Search the web")
def search(query: str) -> str:
    """返回查询的最高匹配结果。"""
    fixtures = {
        "agent engineering": "src1: agent loop, src2: tool use, src3: memory",
        "crewai": "src1: docs intro, src2: flows guide, src3: tools ref",
    }
    for key, value in fixtures.items():
        if key in query.lower():
            return value
    return "src1: generic, src2: generic, src3: generic"


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    fn: Callable[..., str]
    tools: list[Callable[..., str]] = field(default_factory=list)


@dataclass
class Task:
    description: str
    expected_output: str
    agent: Agent
    context: list["Task"] = field(default_factory=list)


@dataclass
class SequentialCrew:
    agents: list[Agent]
    tasks: list[Task]
    memory: "Memory | None" = None

    def kickoff(self, inputs: dict[str, Any]) -> list[str]:
        outputs: list[str] = []
        prior = inputs.get("topic", "")
        by_task: dict[int, str] = {}
        for task in self.tasks:
            if task.context:
                # CrewAI 行为：将每个已声明的上游任务的输出
                # 注入当前任务。当没有声明上游任务时回退到先前的输出。
                joined = "\n\n".join(
                    by_task[id(t)] for t in task.context if id(t) in by_task
                )
                agent_input = joined or prior
            else:
                agent_input = prior
            out = task.agent.fn(agent_input, task.agent.tools, self.memory)
            outputs.append(f"[{task.agent.role}] {out}")
            by_task[id(task)] = out
            prior = out
            if self.memory is not None:
                self.memory.write_short_term(task.agent.role, out)
                self.memory.write_long_term(task.agent.role, out)
        return outputs


@dataclass
class HierarchicalCrew:
    manager: Agent
    specialists: dict[str, Agent]
    max_steps: int = 5
    memory: "Memory | None" = None

    def kickoff(self, topic: str) -> list[str]:
        outputs: list[str] = []
        current = topic
        done: set[str] = set()
        for _ in range(self.max_steps):
            pick = self.manager.fn(done, [], None)
            if pick == "done":
                outputs.append("[manager] done")
                break
            specialist = self.specialists.get(pick)
            if specialist is None:
                outputs.append(f"[manager] unknown pick {pick!r}")
                break
            out = specialist.fn(current, specialist.tools, self.memory)
            outputs.append(f"[manager -> {specialist.role}] {out}")
            current = out
            done.add(pick)
            if self.memory is not None:
                self.memory.write_short_term(specialist.role, out)
        return outputs


class Flow:
    """确定性事件驱动工作流。@start 在启动时触发；
    @listen(topic) 在另一个步骤发出该主题时触发。
    """

    def __init__(self) -> None:
        self.start_step: Callable[[Any], tuple[str, Any]] | None = None
        self.listeners: dict[str, Callable[[Any], tuple[str, Any] | None]] = {}
        self.trace: list[tuple[str, str, Any]] = []

    def start(self, fn: Callable[[Any], tuple[str, Any]]) -> Callable[..., Any]:
        self.start_step = fn
        return fn

    def listen(self, topic: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[[Any], tuple[str, Any] | None]) -> Callable[..., Any]:
            self.listeners[topic] = fn
            return fn

        return decorator

    def kickoff(self, payload: Any) -> list[tuple[str, str, Any]]:
        if self.start_step is None:
            return []
        self.trace = []
        topic, out = self.start_step(payload)
        self.trace.append(("start", topic, out))
        while topic in self.listeners:
            step = self.listeners[topic]
            result = step(out)
            if result is None:
                break
            topic, out = result
            self.trace.append((step.__name__, topic, out))
        return self.trace


class Memory:
    """四存储记忆体系，对应 CrewAI 的短期记忆、长期记忆、实体记忆、上下文记忆。
    Long-term 检索使用 numpy 余弦相似度对哈希 token 向量进行计算。
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.short_term: list[tuple[str, str]] = []
        self.long_term: list[tuple[str, str, np.ndarray]] = []
        self.entity: dict[str, dict[str, str]] = {}

    def _embed(self, text: str) -> np.ndarray:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def write_short_term(self, role: str, value: str) -> None:
        self.short_term.append((role, value))

    def write_long_term(self, role: str, value: str) -> None:
        self.long_term.append((role, value, self._embed(value)))

    def write_entity(self, entity_id: str, key: str, value: str) -> None:
        self.entity.setdefault(entity_id, {})[key] = value

    def recall_long_term(self, query: str, k: int = 2) -> list[tuple[str, str, float]]:
        if not self.long_term:
            return []
        q = self._embed(query)
        scored = [(r, v, float(np.dot(q, e))) for r, v, e in self.long_term]
        scored.sort(key=lambda row: row[2], reverse=True)
        return scored[:k]

    def reset_short_term(self) -> None:
        self.short_term = []


def _researcher(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    topic = prior if isinstance(prior, str) else ""
    # 依次运行 agent 所连接的 search-ish 工具。
    search_fn = next(
        (t for t in tools if getattr(t, "is_tool", False) and "search" in getattr(t, "tool_name", "").lower()),
        None,
    )
    sources = search_fn(topic) if search_fn else "src1, src2, src3"
    return f"3 sources on {topic}: {sources}"


def _writer(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    text = prior if isinstance(prior, str) else ""
    return f"draft (3 paragraphs) from sources: {text[:60]}"


def _editor(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    text = prior if isinstance(prior, str) else ""
    return f"final brief (tightened, 800 words): {text[:60]}"


def _manager(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    done = prior if isinstance(prior, set) else set()
    if "researcher" not in done:
        return "researcher"
    if "writer" not in done:
        return "writer"
    if "editor" not in done:
        return "editor"
    return "done"


def build_agents() -> tuple[Agent, Agent, Agent]:
    researcher = Agent(
        role="researcher",
        goal="find 3 credible sources",
        backstory="former librarian. terse. cites primaries.",
        fn=_researcher,
        tools=[search],
    )
    writer = Agent(
        role="writer",
        goal="turn sources into a draft",
        backstory="editorial voice. paragraphs of three.",
        fn=_writer,
    )
    editor = Agent(
        role="editor",
        goal="tighten draft to final brief",
        backstory="cuts adjectives. enforces house style.",
        fn=_editor,
    )
    return researcher, writer, editor


def main() -> None:
    print("=" * 70)
    print("CREWAI CREW AND FLOW - 第 14 阶段，第 15 课")
    print("=" * 70)

    researcher, writer, editor = build_agents()
    memory = Memory()

    print("\n1. SequentialCrew（研究员 -> 撰写者 -> 编辑）")
    seq = SequentialCrew(
        agents=[researcher, writer, editor],
        tasks=[
            Task("research the topic", "3 sources", researcher),
            Task("write a draft", "3 paragraphs", writer),
            Task("edit to final brief", "800 words", editor),
        ],
        memory=memory,
    )
    for line in seq.kickoff({"topic": "agent engineering 2026"}):
        print(f"  {line}")

    print("\n2. HierarchicalCrew（管理者路由）")
    manager = Agent(
        role="manager",
        goal="pick next specialist",
        backstory="PM background. routes by missing role.",
        fn=_manager,
    )
    hcrew = HierarchicalCrew(
        manager=manager,
        specialists={"researcher": researcher, "writer": writer, "editor": editor},
        memory=memory,
    )
    for line in hcrew.kickoff("agent engineering 2026"):
        print(f"  {line}")

    print("\n3. Flow（确定性，event-driven）")
    flow = Flow()

    @flow.start
    def kickoff(topic: str) -> tuple[str, str]:
        out = _researcher(topic, [search], memory)
        memory.write_short_term("researcher", out)
        memory.write_long_term("researcher", out)
        return "researched", out

    @flow.listen("researched")
    def on_researched(prior: str) -> tuple[str, str]:
        out = _writer(prior, [], memory)
        memory.write_short_term("writer", out)
        memory.write_long_term("writer", out)
        return "drafted", out

    @flow.listen("drafted")
    def on_drafted(prior: str) -> tuple[str, str]:
        out = _editor(prior, [], memory)
        memory.write_short_term("editor", out)
        memory.write_long_term("editor", out)
        return "edited", out

    @flow.listen("edited")
    def on_edited(prior: str) -> None:
        return None

    for step_name, topic, output in flow.kickoff("agent engineering 2026"):
        print(f"  [{step_name}] topic={topic!r} out={output[:60]}")

    print("\n4. 记忆：recall_long_term('brief')")
    for role, value, score in memory.recall_long_term("brief"):
        print(f"  [{role}] score={score:+.3f} value={value[:50]}")

    print("\n5. 第二次启动（long-term 记忆保留）")
    memory.reset_short_term()
    seq2 = SequentialCrew(
        agents=[researcher, writer, editor],
        tasks=[
            Task("research", "3 sources", researcher),
            Task("draft", "3 paragraphs", writer),
            Task("edit", "800 words", editor),
        ],
        memory=memory,
    )
    seq2.kickoff({"topic": "agent engineering 2026"})
    print(f"  long_term 条目：{len(memory.long_term)}")
    print(f"  short_term 条目（本次运行）：{len(memory.short_term)}")

    print()
    print("Crew：由 LLM 选择执行形态。Flow：由代码掌控执行形态。")
    print("文档（2026）：生产环境从 Flow 开始；将 Crew 作为 sub-steps. 纳入")


if __name__ == "__main__":
    main()
