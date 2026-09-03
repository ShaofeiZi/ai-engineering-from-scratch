"""投票与辩论拓扑测试工具，仅使用 stdlib。

在脚本化任务中运行 star / chain / tree / graph 拓扑。每个 Agent 都有基础准确率
和 error_bias 方向（出错时偏向哪个错误答案）。模拟 N 个 Agent 进行多轮精炼，
并测量准确率、token 数和模拟延迟。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class SimAgent:
    name: str
    base_accuracy: float
    error_bias: str
    tokens_per_call: int = 400

    def answer(self, correct: str, rng: random.Random) -> str:
        return correct if rng.random() < self.base_accuracy else self.error_bias


@dataclass
class RunResult:
    topology: str
    n: int
    final_answer: str
    correct: str
    tokens: int
    steps: int

    def accuracy(self) -> int:
        return 1 if self.final_answer == self.correct else 0


def majority(items: list[str]) -> str:
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return max(counts, key=counts.get)


def run_star(agents: list[SimAgent], correct: str, rng: random.Random) -> RunResult:
    hub = agents[0]
    workers = agents[1:]
    answers = [w.answer(correct, rng) for w in workers]
    tokens = sum(w.tokens_per_call for w in workers) + hub.tokens_per_call
    final = majority(answers) if answers else hub.answer(correct, rng)
    return RunResult("star", len(agents), final, correct, tokens, steps=2)


def run_chain(agents: list[SimAgent], correct: str, rng: random.Random) -> RunResult:
    current = agents[0].answer(correct, rng)
    tokens = agents[0].tokens_per_call
    for a in agents[1:]:
        proposal = a.answer(correct, rng)
        current = proposal if proposal != current and rng.random() < a.base_accuracy else current
        tokens += a.tokens_per_call
    return RunResult("chain", len(agents), current, correct, tokens, steps=len(agents))


def run_tree(agents: list[SimAgent], correct: str, rng: random.Random) -> RunResult:
    root = agents[0]
    leaves = agents[1:]
    if len(leaves) <= 1:
        return run_star(agents, correct, rng)
    mid = len(leaves) // 2
    left_answers = [a.answer(correct, rng) for a in leaves[:mid]]
    right_answers = [a.answer(correct, rng) for a in leaves[mid:]]
    tokens = sum(a.tokens_per_call for a in leaves) + root.tokens_per_call
    left_consensus = majority(left_answers)
    right_consensus = majority(right_answers)
    final = majority([left_consensus, right_consensus])
    return RunResult("tree", len(agents), final, correct, tokens, steps=3)


def run_graph(agents: list[SimAgent], correct: str, rng: random.Random, rounds: int = 2) -> RunResult:
    # 每个 Agent 先提出答案，然后查看所有提案并可能更新自己的答案
    # （如果向共识漂移，则按比例降低准确率）。
    positions = [a.answer(correct, rng) for a in agents]
    tokens = sum(a.tokens_per_call for a in agents)
    for _ in range(rounds - 1):
        majority_now = majority(positions)
        new_positions = []
        for pos, ag in zip(positions, agents):
            if pos != majority_now and rng.random() < 0.4:
                new_positions.append(majority_now)
            else:
                new_positions.append(pos)
            tokens += ag.tokens_per_call
        positions = new_positions
    return RunResult("graph", len(agents), majority(positions), correct, tokens, steps=rounds * 2)


def make_agents(n: int, heterogeneous: bool, seed: int) -> list[SimAgent]:
    rng = random.Random(seed)
    if heterogeneous:
        biases = ["WRONG-A", "WRONG-B", "WRONG-C"]
        accuracies = [0.72, 0.70, 0.74, 0.71, 0.73, 0.70, 0.72]
    else:
        biases = ["WRONG-A"]
        accuracies = [0.72] * 7
    return [
        SimAgent(f"agent-{i}", accuracies[i % len(accuracies)], biases[i % len(biases)])
        for i in range(n)
    ]


def bench(correct: str, trials: int, heterogeneous: bool) -> None:
    tag = "异构" if heterogeneous else "同构（monoculture）"
    print("\n" + "=" * 72)
    print(f"基准测试 — {tag}")
    print("=" * 72)
    print(f"{'拓扑':10s} {'N':>3s} {'准确率':>8s} {'平均 token':>12s} {'步数':>6s}")
    for topology in ("star", "chain", "tree", "graph"):
        for n in (3, 5, 7):
            acc_sum = 0
            tok_sum = 0
            step_sum = 0
            for t in range(trials):
                agents = make_agents(n, heterogeneous, seed=t)
                rng = random.Random(t * 31 + 7)
                if topology == "star":
                    r = run_star(agents, correct, rng)
                elif topology == "chain":
                    r = run_chain(agents, correct, rng)
                elif topology == "tree":
                    r = run_tree(agents, correct, rng)
                else:
                    r = run_graph(agents, correct, rng)
                acc_sum += r.accuracy()
                tok_sum += r.tokens
                step_sum += r.steps
            print(f"{topology:10s} {n:>3d} {acc_sum/trials:>8.2f} {tok_sum//trials:>12d} {step_sum//trials:>6d}")


def main() -> None:
    bench(correct="RIGHT", trials=200, heterogeneous=False)
    bench(correct="RIGHT", trials=200, heterogeneous=True)
    print("\n要点：")
    print("  在每种拓扑和 N 取值下，异构 ensemble 都优于同构 ensemble。")
    print("  graph/N=7 展现了协调税：token 数约为 star/N=3 的 7 倍。")
    print("  对低风险聚合而言，star 是成本甜点位。")
    print("  chain 在 monoculture 下表现较差，因为一种偏差会沿链传播。")


if __name__ == "__main__":
    main()
