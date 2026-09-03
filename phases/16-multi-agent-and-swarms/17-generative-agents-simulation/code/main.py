"""生成式 Agent 的微型实现：stdlib 版 Smallville。

五个 Agent 共享一个小世界。Agent 0 被植入聚会目标。随着 tick 推进，邀请通过
双边记忆观察传播，reflection 综合形成信念，计划随之更新。到最后一个 tick，
至少 3 个 Agent 会在没有中央 orchestrator 的情况下汇聚到聚会地点。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


TICK_DURATION_S = 0.01  # 模拟值；输出即时产生


@dataclass
class Memory:
    ts: int
    kind: str
    content: str
    importance: int


@dataclass
class Plan:
    tick: int
    where: str
    note: str


@dataclass
class Agent:
    name: str
    location: str
    stream: list[Memory] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)

    def observe(self, tick: int, content: str, importance: int = 3) -> None:
        self.stream.append(Memory(tick, "observation", content, importance))

    def reflect(self, tick: int) -> None:
        recent_important = [m for m in self.stream if m.importance >= 6 and tick - m.ts <= 5]
        for m in recent_important:
            if "邀请" in m.content and "聚会" in m.content:
                belief = "有人邀请我参加聚会"
                if belief not in self.beliefs:
                    self.beliefs.append(belief)
                    self.stream.append(Memory(tick, "reflection", belief, 8))

    def update_plan(self, tick: int) -> None:
        if "有人邀请我参加聚会" in self.beliefs:
            if not any(p.where == "HobbsCafe" for p in self.plans):
                self.plans.append(Plan(tick=5, where="HobbsCafe", note="参加聚会"))

    def act(self, tick: int) -> str:
        for p in self.plans:
            if p.tick == tick:
                self.location = p.where
                return f"{self.name} 前往 {p.where}（{p.note}）"
        return f"{self.name} 留在 {self.location}"


def retrieve_top_k(stream: list[Memory], query: str, tick: int, k: int = 3) -> list[Memory]:
    def score(m: Memory) -> float:
        recency = math.exp(-0.3 * (tick - m.ts))
        importance = m.importance / 10.0
        relevance = 0.6 if any(w in m.content.lower() for w in query.lower().split()) else 0.1
        return recency + importance + relevance
    return sorted(stream, key=score, reverse=True)[:k]


def run_simulation(n_agents: int = 5, ticks: int = 6) -> None:
    agents = [Agent(f"agent-{i}", location="home") for i in range(n_agents)]

    # 为 Agent 0 植入聚会目标。
    agents[0].stream.append(Memory(0, "goal", "在 tick 5 于 HobbsCafe 举办情人节聚会", 10))
    agents[0].plans.append(Plan(tick=5, where="HobbsCafe", note="举办聚会"))
    agents[0].beliefs.append("有人邀请我参加聚会")

    print("=" * 72)
    print(f"生成式智能体（微型版）——{n_agents} 个 Agent，{ticks} 个 tick")
    print("=" * 72)

    for tick in range(ticks):
        print(f"\n--- tick {tick} ---")
        # 邀请传播：Agent 0 在 tick 0-2 邀请直接邻居；之后每个受邀 Agent
        # 在后续 tick 再邀请一个人。
        if tick == 0:
            for i in (1, 2):
                agents[i].observe(tick, "agent-0 邀请我在 tick 5 前往 HobbsCafe 参加聚会", importance=8)
                print(f"  agent-0 -> agent-{i}：邀请")
        if tick == 1:
            agents[3].observe(tick, "agent-1 邀请我在 tick 5 前往 HobbsCafe 参加聚会", importance=7)
            print(f"  agent-1 -> agent-3：二度邀请")
        if tick == 2:
            agents[4].observe(tick, "agent-2 邀请我在 tick 5 前往 HobbsCafe 参加聚会", importance=7)
            print(f"  agent-2 -> agent-4：二度邀请")

        for a in agents:
            a.reflect(tick)
            a.update_plan(tick)
            action = a.act(tick)
            if action.startswith(a.name + " 前往"):
                print(f"  {action}")

    # 最终状态
    print("\n" + "=" * 72)
    print("最终位置：")
    for a in agents:
        print(f"  {a.name:10s} 位于 {a.location}")

    at_party = sum(1 for a in agents if a.location == "HobbsCafe")
    print(f"\n{at_party}/{n_agents} 个 Agent 汇聚到 HobbsCafe 参加聚会。")
    print("没有 orchestrator，只有一个种子，其余来自记忆 + reflection + 计划。")


def demo_retrieval() -> None:
    print("\n" + "=" * 72)
    print("检索演示 — 按新近性 + 重要性 + 相关性选取 top-k")
    print("=" * 72)
    stream = [
        Memory(0, "observation", "在咖啡馆看到了 Isabella", importance=4),
        Memory(1, "observation", "Isabella 说她正在筹办聚会", importance=7),
        Memory(2, "reflection", "我会喜欢咖啡馆里的聚会", importance=6),
        Memory(3, "observation", "Klaus 提到他正在写论文", importance=3),
    ]
    top = retrieve_top_k(stream, query="聚会 咖啡馆", tick=4, k=3)
    print("  查询：在 tick 4 搜索“聚会 咖啡馆”")
    for m in top:
        print(f"  [t={m.ts}] {m.kind:11s} 重要性={m.importance} :: {m.content}")


def main() -> None:
    run_simulation()
    demo_retrieval()
    print("\n要点：")
    print("  一个种子 + 三个组件 = 无需 orchestrator 的协调到场。")
    print("  reflection 是关键支柱：去掉它，信念就无法形成。")
    print("  检索结合新近性、重要性和相关性，任何单一评分都不够。")


if __name__ == "__main__":
    main()
