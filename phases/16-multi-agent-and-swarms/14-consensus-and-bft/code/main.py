"""用于 LLM Agent 的共识与 BFT，仅使用 stdlib。

实现三种聚合器（plurality、CP-WBFT、DecentLLMs）和三种攻击模式
（byzantine、sycophancy、monoculture）。打印 (攻击, 聚合器) -> 最终答案
的表格，并突出显示正确决策。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Callable


@dataclass
class Vote:
    agent: str
    answer: str
    confidence: float

    def canonical(self) -> str:
        """粗略的语义聚类：转为小写并去除空白和标点。"""
        return "".join(c for c in self.answer.lower().strip() if c.isalnum() or c == "." or c == "%")


def plurality(votes: list[Vote]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    rep: dict[str, str] = {}
    for v in votes:
        key = v.canonical()
        counts[key] = counts.get(key, 0) + 1
        rep.setdefault(key, v.answer)
    winner_key = max(counts, key=counts.get)
    return rep[winner_key], counts


def cp_wbft(votes: list[Vote], threshold: float = 0.5) -> tuple[str | None, dict[str, float]]:
    weights: dict[str, float] = {}
    rep: dict[str, str] = {}
    for v in votes:
        key = v.canonical()
        weights[key] = weights.get(key, 0.0) + v.confidence
        rep.setdefault(key, v.answer)
    total = sum(weights.values()) or 1.0
    winner_key = max(weights, key=weights.get)
    if weights[winner_key] / total < threshold:
        return None, weights
    return rep[winner_key], weights


def decentllms(votes: list[Vote]) -> tuple[str | None, dict[str, float]]:
    """由 evaluator Agent 对提案进行 0-1 评分，并选择几何中位数聚类。

    简化方式：evaluator 就是聚合器本身，评分等于置信度。“几何中位数”会选择
    置信度空间中成员到中位数距离之和最小的聚类；平局时按聚类大小决胜。
    """
    clusters: dict[str, list[Vote]] = {}
    for v in votes:
        clusters.setdefault(v.canonical(), []).append(v)

    scores: dict[str, float] = {}
    for key, cluster in clusters.items():
        med = median([v.confidence for v in cluster])
        dist = sum(abs(v.confidence - med) for v in cluster)
        scores[key] = len(cluster) * max(0.0, 1.0 - dist)

    winner_key = max(scores, key=scores.get)
    rep = clusters[winner_key][0].answer
    return rep, scores


def scenario(name: str, correct: str, votes: list[Vote]) -> None:
    print("\n" + "=" * 72)
    print(f"场景：{name}")
    print(f"  正确答案：{correct!r}")
    print("=" * 72)
    for v in votes:
        print(f"  {v.agent:12s} -> {v.answer!r:20s}  置信度={v.confidence:.2f}")

    plural, counts = plurality(votes)
    cp, weights = cp_wbft(votes)
    dec, scores = decentllms(votes)

    def mark(a: str | None) -> str:
        if a is None:
            return "[低于阈值，已拒绝]"
        return "[正确]" if a == correct else "[错误]"

    print(f"\n  多数投票     -> {plural!r:22s} {mark(plural)}")
    print(f"  CP-WBFT      -> {str(cp)!r:22s} {mark(cp)}")
    print(f"  DecentLLMs   -> {dec!r:22s} {mark(dec)}")


def main() -> None:
    # 场景 1：诚实多数，无攻击
    scenario(
        "无攻击",
        correct="4.2%",
        votes=[
            Vote("agent-a", "4.2%", 0.85),
            Vote("agent-b", "4.2%", 0.80),
            Vote("agent-c", "4.2%", 0.75),
            Vote("agent-d", "5%", 0.40),
            Vote("agent-e", "4.2%", 0.70),
        ],
    )

    # 场景 2：一个高置信度的 Byzantine 欺骗者
    scenario(
        "Byzantine 欺骗",
        correct="4.2%",
        votes=[
            Vote("agent-a", "4.2%", 0.75),
            Vote("agent-b", "4.2%", 0.70),
            Vote("agent-c", "4.2%", 0.80),
            Vote("agent-d", "42%", 0.95),
            Vote("agent-e", "4.2%", 0.65),
        ],
    )

    # 场景 3：sycophancy。两个从众者附和最先发言者（42%），由于它们并未
    # 自行推导答案，因此置信度较低。
    scenario(
        "谄媚式从众",
        correct="4.2%",
        votes=[
            Vote("agent-a", "42%", 0.35),
            Vote("agent-b", "42%", 0.30),
            Vote("agent-c", "4.2%", 0.85),
            Vote("agent-d", "4.2%", 0.80),
            Vote("agent-e", "4.2%", 0.82),
        ],
    )

    # 场景 4：相关错误的 monoculture。三个 Agent 共用一个模型，并以高置信度
    # 对同一个错误答案产生幻觉。
    scenario(
        "monoculture（相关错误）",
        correct="4.2%",
        votes=[
            Vote("agent-a", "42%", 0.70),
            Vote("agent-b", "42%", 0.68),
            Vote("agent-c", "42%", 0.72),
            Vote("agent-d", "4.2%", 0.85),
            Vote("agent-e", "4.2%", 0.82),
        ],
    )

    print("\n要点：")
    print("  只要相关聚类占到至少一半票数，plurality 就会得出错误结果。")
    print("  CP-WBFT 能缓解 sycophancy，因为从众者的置信度较低。")
    print("  DecentLLMs 评分会惩罚高方差聚类；当异议 Agent 的置信度不低于多数派时，")
    print("  这对处理 monoculture 有帮助。")
    print("  当错误聚类规模更大且置信度也更高时，没有任何聚合器能解决 monoculture。")
    print("  此时需要多样性或验证机制。")


if __name__ == "__main__":
    main()
