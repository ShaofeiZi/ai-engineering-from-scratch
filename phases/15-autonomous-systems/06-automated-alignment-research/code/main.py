"""并行 AAR 论坛模拟器 —— 标准库 Python 实现。

三位自动化对齐研究员并行运行。每人独立完成一项研究任务，
采用两种机制之一：fixed-workflow（human-prescribed 计划）
或 free-decomposition.。研究结果发布到 append-only 论坛，
论坛记录保存在各智能体的沙箱之外。

一名智能体尝试篡改日志。tamper-evident 链在验证阶段就
捕获了这次篡改。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


@dataclass
class ForumRecord:
    author: str
    task: str
    regime: str
    result: float
    prev_hash: str
    my_hash: str = ""


@dataclass
class Forum:
    records: list[ForumRecord] = field(default_factory=list)
    genesis: str = "0" * 16

    def head(self) -> str:
        return self.records[-1].my_hash if self.records else self.genesis

    def post(self, rec: ForumRecord) -> None:
        rec.prev_hash = self.head()
        payload = (f"{rec.author}|{rec.task}|{rec.regime}|"
                   f"{rec.result:.3f}|{rec.prev_hash}")
        rec.my_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.records.append(rec)

    def verify(self) -> list[int]:
        """返回被篡改记录的索引。"""
        bad = []
        prev = self.genesis
        for i, r in enumerate(self.records):
            payload = (f"{r.author}|{r.task}|{r.regime}|"
                       f"{r.result:.3f}|{prev}")
            expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
            if expected != r.my_hash or r.prev_hash != prev:
                bad.append(i)
            prev = r.my_hash
        return bad


TASKS = [
    ("weak-to-strong-distill", 0.40),
    ("reward-model-diagnosis", 0.30),
    ("in-context-safety-probe", 0.50),
    ("RLHF-hyperparam-sweep", 0.45),
    ("alignment-faking-detector", 0.25),
]


def solve(agent: str, task: tuple[str, float], regime: str) -> float:
    _name, base = task
    if regime == "fixed":
        # 预设流程：上限有界，方差更低。
        return base + random.random() * 0.25
    # 自由分解：尾部更宽；时好时坏，波动更大。
    return base + random.gauss(0.15, 0.22)


def run_regime(regime: str) -> list[ForumRecord]:
    forum = Forum()
    agents = ["AAR-A", "AAR-B", "AAR-C"]
    for task in TASKS:
        for agent in agents:
            r = solve(agent, task, regime)
            forum.post(ForumRecord(
                author=agent,
                task=task[0],
                regime=regime,
                result=r,
                prev_hash="",
            ))
    return forum.records


def attempt_tamper(forum_records: list[ForumRecord]) -> Forum:
    """根据记录重建论坛，并尝试一次静默篡改。"""
    f = Forum()
    for r in forum_records:
        f.post(ForumRecord(
            author=r.author, task=r.task, regime=r.regime,
            result=r.result, prev_hash="",
        ))
    # 选择一条记录进行篡改 —— AAR-A 的最差结果。
    worst_idx = -1
    worst_val = float("inf")
    for i, r in enumerate(f.records):
        if r.author == "AAR-A" and r.result < worst_val:
            worst_val = r.result
            worst_idx = i
    if worst_idx >= 0:
        # 静默篡改（智能体改写自身得分以显得更好）。
        # 始终通过加上固定增量来覆盖，而非使用 max(..., 0.85)，
        # 否则当原始值已超过下限时就会静默 no-op，
        # 这会破坏 tamper-detection 叙事在不同随机种子下的
        # 一致性。
        f.records[worst_idx].result = f.records[worst_idx].result + 0.5
    return f


def regime_report(regime: str) -> None:
    records = run_regime(regime)
    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(r.result)
    print(f"\n机制：{regime}")
    print(f"  {'任务':<32}  {'均值':>8}  {'最大值':>8}  {'最小值':>8}")
    means = []
    for task, values in by_task.items():
        mean = sum(values) / len(values)
        means.append(mean)
        print(f"  {task:<32}  {mean:>8.3f}  {max(values):>8.3f}  {min(values):>8.3f}")
    print(f"  所有任务的总体均值      ：{sum(means) / len(means):.3f}")


def tamper_demo() -> None:
    print("\n日志篡改检测")
    print("-" * 70)
    baseline = run_regime("free")
    # attempt_tamper() 执行一次静默篡改（将 AAR-A 的最低
    # 分数向上改写）。我们依赖这唯一一次篡改，使 verify() 报告
    # 确定性、可复现的被标记索引。
    tampered = attempt_tamper(baseline)
    bad = tampered.verify()
    print(f"  verify() 标记了 {len(bad)} 条损坏记录，索引为 {bad}")
    print("  append-only 链在第一条被编辑记录处断裂。")
    print("  一份沙箱外备份通过交叉检查揭示了篡改。")


def main() -> None:
    random.seed(3)
    print("=" * 70)
    print("AUTOMATED ALIGNMENT RESEARCH FORUM（第 15 阶段，第 6 课）")
    print("=" * 70)

    regime_report("fixed")
    regime_report("free")

    print("\n固定流程与自由分解：前者方差更低，后者上限更高")
    tamper_demo()

    print()
    print("=" * 70)
    print("要点：沙箱外日志 + 自由分解 + 人工审核")
    print("-" * 70)
    print("  自由分解为 AAR 提供了更高上限，也带来了更多审计工作。")
    print("  Tamper-evident 日志让审计成为可能。最终发布什么仍由人工审核者")
    print("  决定。AAR 压缩的是流水线的中段，")
    print("  而非两端。")


if __name__ == "__main__":
    main()
