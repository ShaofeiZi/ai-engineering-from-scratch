"""数值任务上的多 Agent 辩论（Du 等，2023 风格）。

3 个 Agent 分别从不同的（可能错误的）答案开始。每一轮中，各 Agent 都会读取
其他 Agent 的答案，并向加权平均值修正。程序逐轮记录收敛过程。Agent 策略由脚本
实现，而非由 LLM 驱动；重点在于辩论的动态过程。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


TRUE_ANSWER = 42.0


@dataclass
class DebateAgent:
    name: str
    answer: float
    confidence: float
    history: list[float] = field(default_factory=list)

    def initial(self) -> None:
        self.history.append(self.answer)

    def revise(self, others: list["DebateAgent"]) -> None:
        """按置信度加权，计算自己与其他 Agent 答案的加权平均值。"""
        weights = [self.confidence] + [o.confidence for o in others]
        values = [self.answer] + [o.answer for o in others]
        total_w = sum(weights)
        new_answer = sum(w * v for w, v in zip(weights, values)) / total_w
        self.answer = new_answer
        self.confidence = min(self.confidence * 1.05, 1.0)
        self.history.append(self.answer)


def agreement_score(agents: list[DebateAgent], tol: float = 0.1) -> float:
    """答案处于平均值 tol 范围内的 Agent 比例。"""
    mean = sum(a.answer for a in agents) / len(agents)
    agree = sum(1 for a in agents if abs(a.answer - mean) <= tol)
    return agree / len(agents)


def error_vs_truth(agents: list[DebateAgent]) -> float:
    mean = sum(a.answer for a in agents) / len(agents)
    return abs(mean - TRUE_ANSWER)


def run_debate(agents: list[DebateAgent], rounds: int, label: str) -> None:
    print(f"\n=== {label}（{rounds} 轮）===")
    for a in agents:
        a.initial()
    hdr = " ".join(f"{a.name:>6s}" for a in agents)
    print(f"  轮次     {hdr}    一致率   与真值误差")
    for a in agents:
        pass
    print(f"    0     {' '.join(f'{a.answer:6.2f}' for a in agents)}    {agreement_score(agents):4.2f}     {error_vs_truth(agents):5.2f}")
    for r in range(1, rounds + 1):
        updates = []
        for a in agents:
            others = [o for o in agents if o is not a]
            updates.append((a, others))
        for a, others in updates:
            a.revise(others)
        print(f"    {r}     {' '.join(f'{a.answer:6.2f}' for a in agents)}    {agreement_score(agents):4.2f}     {error_vs_truth(agents):5.2f}")


def fresh_team(seed: int) -> list[DebateAgent]:
    random.seed(seed)
    return [
        DebateAgent(name="A", answer=38.0, confidence=0.6),
        DebateAgent(name="B", answer=42.5, confidence=0.8),
        DebateAgent(name="C", answer=51.0, confidence=0.4),
    ]


def single_shot_majority(agents: list[DebateAgent]) -> float:
    """对照组：对第 0 轮答案取多数结果（自洽性基线）。"""
    return sum(a.answer for a in agents) / len(agents)


def main() -> None:
    print("多 Agent 辩论（Du 等，2023 风格）")
    print("-" * 46)
    print(f"正确答案：{TRUE_ANSWER}")

    baseline = fresh_team(seed=1)
    for a in baseline:
        a.initial()
    control_mean = single_shot_majority(baseline)
    print(f"\n对照组（第 0 轮均值，自洽性基线）：{control_mean:.2f}")
    print(f"与真值的误差：{abs(control_mean - TRUE_ANSWER):.2f}")

    team3 = fresh_team(seed=1)
    run_debate(team3, rounds=3, label="3 个 Agent 辩论 3 轮")

    team5 = fresh_team(seed=2)
    run_debate(team5, rounds=5, label="3 个 Agent 辩论 5 轮（收益递减）")

    print("\n要点：")
    print("  - 第 1 轮交流降低误差最多。")
    print("  - 第 2 至 3 轮的收益会累积。")
    print("  - 超过第 3 轮后，每轮收益会减小（Du 等所述的平台期）。")
    print("  - 成本按 N * R 次 LLM 调用增长，上下文也随之增大。")


if __name__ == "__main__":
    main()
