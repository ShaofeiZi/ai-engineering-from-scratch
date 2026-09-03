"""Agent 经济：Shapley 归因、次价拍卖与声誉路由。

全部仅使用 stdlib。N<=6 时精确计算 Shapley，否则使用采样。次价拍卖演示诚实出价。
声誉路由在 100 轮中对比按声誉加权和随机分配。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable


# ---------- Shapley ----------

def shapley_exact(value_fn: Callable[[frozenset], float], agents: list[str]) -> dict[str, float]:
    n = len(agents)
    contribs = {a: 0.0 for a in agents}
    for order in permutations(agents):
        visited: set[str] = set()
        prev_value = value_fn(frozenset(visited))
        for a in order:
            visited.add(a)
            new_value = value_fn(frozenset(visited))
            contribs[a] += new_value - prev_value
            prev_value = new_value
    factorial = math.factorial(n)
    return {a: v / factorial for a, v in contribs.items()}


def shapley_sampled(value_fn: Callable[[frozenset], float], agents: list[str],
                    samples: int, rng: random.Random) -> dict[str, float]:
    contribs = {a: 0.0 for a in agents}
    for _ in range(samples):
        order = list(agents)
        rng.shuffle(order)
        visited: set[str] = set()
        prev_value = value_fn(frozenset(visited))
        for a in order:
            visited.add(a)
            new_value = value_fn(frozenset(visited))
            contribs[a] += new_value - prev_value
            prev_value = new_value
    return {a: v / samples for a, v in contribs.items()}


# ---------- 次价拍卖 ----------

@dataclass
class Bid:
    bidder: str
    value: float


def second_price(bids: list[Bid]) -> tuple[str, float] | None:
    if len(bids) < 2:
        return None
    sorted_bids = sorted(bids, key=lambda b: b.value, reverse=True)
    winner = sorted_bids[0].bidder
    payment = sorted_bids[1].value
    return winner, payment


# ---------- 声誉加权路由 ----------

class Reputation:
    def __init__(self, alpha: float = 0.95, floor: float = 0.1) -> None:
        self.alpha = alpha
        self.floor = floor
        self.scores: dict[str, float] = {}

    def init(self, agents: list[str]) -> None:
        for a in agents:
            self.scores[a] = 1.0

    def update(self, agent: str, quality: float) -> None:
        current = self.scores.get(agent, 1.0)
        self.scores[agent] = max(self.floor, self.alpha * current + (1 - self.alpha) * quality)

    def weights(self, agents: list[str]) -> list[float]:
        return [self.scores.get(a, 1.0) for a in agents]


def weighted_choice(agents: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for a, w in zip(agents, weights):
        upto += w
        if r <= upto:
            return a
    return agents[-1]


# ---------- 演示 ----------

def demo_shapley() -> None:
    print("=" * 72)
    print("SHAPLEY 归因 — 3 个 Agent 协作完成任务")
    print("=" * 72)

    # 价值函数：coder 单独工作 = 0.5，researcher 单独工作 = 0.3，reviewer 单独工作 = 0.1；
    # 两两组合和三者组合具有超可加收益。
    base = {
        frozenset(): 0.0,
        frozenset(["coder"]): 0.5,
        frozenset(["researcher"]): 0.3,
        frozenset(["reviewer"]): 0.1,
        frozenset(["coder", "researcher"]): 0.85,
        frozenset(["coder", "reviewer"]): 0.70,
        frozenset(["researcher", "reviewer"]): 0.55,
        frozenset(["coder", "researcher", "reviewer"]): 1.00,
    }
    value_fn = lambda s: base[s]
    agents = ["coder", "researcher", "reviewer"]

    exact = shapley_exact(value_fn, agents)
    print("  精确 Shapley 值：")
    for a, v in exact.items():
        print(f"    {a:11s} {v:.4f}")
    print(f"    总和 = {sum(exact.values()):.4f}（应等于大联盟价值 1.0000）")

    rng = random.Random(0)
    sampled = shapley_sampled(value_fn, agents, samples=200, rng=rng)
    print("\n  采样 Shapley 值（N=200）：")
    for a, v in sampled.items():
        print(f"    {a:11s} {v:.4f}")


def demo_auction() -> None:
    print("\n" + "=" * 72)
    print("次价拍卖 — 5 个竞标者争夺一个任务名额")
    print("=" * 72)
    bids = [
        Bid("agent-a", 0.82),
        Bid("agent-b", 0.60),
        Bid("agent-c", 0.95),
        Bid("agent-d", 0.45),
        Bid("agent-e", 0.77),
    ]
    for b in bids:
        print(f"  {b.bidder:10s} 出价 {b.value:.2f}")
    result = second_price(bids)
    if result:
        winner, payment = result
        print(f"\n  胜者：{winner}  支付：{payment:.2f}")
        print("  （胜者支付第二高出价；这会激励诚实出价）")


def demo_reputation_routing() -> None:
    print("\n" + "=" * 72)
    print("声誉加权路由 — 100 个任务、4 个 Agent、50 轮预热")
    print("=" * 72)
    agents = ["alpha", "beta", "gamma", "delta"]
    true_quality = {"alpha": 0.9, "beta": 0.5, "gamma": 0.75, "delta": 0.3}

    rng = random.Random(0)

    # 随机基线
    random_quality = 0.0
    for _ in range(100):
        a = rng.choice(agents)
        q = max(0.0, min(1.0, true_quality[a] + rng.uniform(-0.1, 0.1)))
        random_quality += q

    # 经过 50 轮预热的声誉加权方案
    rng = random.Random(0)
    rep = Reputation()
    rep.init(agents)
    rep_quality = 0.0
    for i in range(100):
        if i < 50:
            a = rng.choice(agents)  # 预热：了解每个 Agent
        else:
            a = weighted_choice(agents, rep.weights(agents), rng)
        q = max(0.0, min(1.0, true_quality[a] + rng.uniform(-0.1, 0.1)))
        rep.update(a, q)
        rep_quality += q

    print(f"  随机路由平均质量：{random_quality / 100:.3f}")
    print(f"  声誉加权路由：    {rep_quality / 100:.3f}")
    print(f"  提升：{(rep_quality - random_quality) / random_quality * 100:+.1f}%")
    print("\n  最终声誉分数：")
    for a in agents:
        print(f"    {a:8s} 声誉={rep.scores[a]:.3f}  真实质量={true_quality[a]:.2f}")


def main() -> None:
    demo_shapley()
    demo_auction()
    demo_reputation_routing()
    print("\n要点：")
    print("  Shapley 公平但计算昂贵。N > 6 时应使用采样。")
    print("  在单调聚合下，次价拍卖能激励诚实出价（Google Research）。")
    print("  声誉资本形成闭环：良好路由 + 衰减 + 惩罚。")


if __name__ == "__main__":
    main()
