"""多智能体辩论：全网格（full mesh）与稀疏星型（star）拓扑。

带有不同意见漂移的脚本化辩手。衡量收敛答案、
达成共识的轮次，以及总批评操作数（作为成本代理）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Debater:
    name: str
    drift: Callable[[str, list[str]], str]


def _make_debater(name: str, bias: str,
                  corrections: dict[str, str]) -> Debater:
    def drift(question: str, peer_answers: list[str]) -> str:
        current = corrections.get(question, bias)
        if peer_answers:
            common = Counter(peer_answers).most_common(1)[0][0]
            if common != current and common != bias:
                return common
        return current
    return Debater(name=name, drift=drift)


def full_mesh_round(debaters: list[Debater], question: str,
                    prior: dict[str, str]) -> tuple[dict[str, str], int]:
    new_answers: dict[str, str] = {}
    ops = 0
    for debater in debaters:
        peers = [prior[d.name] for d in debaters if d.name != debater.name]
        new_answers[debater.name] = debater.drift(question, peers)
        ops += len(peers)
    return new_answers, ops


def sparse_star_round(hub: Debater, spokes: list[Debater], question: str,
                      prior: dict[str, str]) -> tuple[dict[str, str], int]:
    new_answers: dict[str, str] = {}
    ops = 0
    spoke_names = [s.name for s in spokes]
    new_answers[hub.name] = hub.drift(
        question, [prior[n] for n in spoke_names]
    )
    ops += len(spoke_names)
    for spoke in spokes:
        new_answers[spoke.name] = spoke.drift(
            question, [prior[hub.name]]
        )
        ops += 1
    return new_answers, ops


def run_debate(debaters: list[Debater], question: str, rounds: int,
               topology: str) -> tuple[str, int, int]:
    prior: dict[str, str] = {}
    for debater in debaters:
        prior[debater.name] = debater.drift(question, [])

    total_ops = 0
    converged_round = -1
    hub = debaters[0]
    spokes = debaters[1:]
    for r in range(rounds):
        if topology == "full_mesh":
            new, ops = full_mesh_round(debaters, question, prior)
        else:
            new, ops = sparse_star_round(hub, spokes, question, prior)
        total_ops += ops
        if all(v == list(new.values())[0] for v in new.values()) and converged_round == -1:
            converged_round = r + 1
        prior = new

    votes = Counter(prior.values()).most_common(1)[0][0]
    return votes, converged_round, total_ops


def main() -> None:
    print("=" * 70)
    print("多智能体辩论 — 第 14 阶段，第 25 课")
    print("=" * 70)

    questions_and_truth = {
        "capital_of_portugal": "Lisbon",
        "is_2_plus_2_equal_4": "yes",
        "chess_legal_e4": "legal",
    }

    debaters = [
        _make_debater(
            "alpha", bias="Lisbon",
            corrections={"is_2_plus_2_equal_4": "yes",
                         "chess_legal_e4": "legal"},
        ),
        _make_debater(
            "beta", bias="Madrid",
            corrections={"capital_of_portugal": "Lisbon",
                         "is_2_plus_2_equal_4": "yes",
                         "chess_legal_e4": "legal"},
        ),
        _make_debater(
            "gamma", bias="Porto",
            corrections={"capital_of_portugal": "Lisbon",
                         "is_2_plus_2_equal_4": "yes",
                         "chess_legal_e4": "legal"},
        ),
    ]

    for q, truth in questions_and_truth.items():
        print(f"\n--- {q}  (真实值: {truth}) ---")
        for topology in ("full_mesh", "sparse_star"):
            answer, converged, ops = run_debate(
                debaters, q, rounds=3, topology=topology,
            )
            correct = "CORRECT" if answer == truth else "WRONG"
            print(f"  {topology:12}  answer={answer:10}  "
                  f"converged_round={converged}  ops={ops}  {correct}")

    print()
    print("稀疏星型在准确率上与全网格相当，但批评操作更少。")
    print("辩论有助于事实性任务和基于规则的任务，但会增加延迟和成本。")


if __name__ == "__main__":
    main()
