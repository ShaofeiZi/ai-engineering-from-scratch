"""多 Agent 基准测试记分卡生成器，仅使用 stdlib。

在玩具任务集上模拟 3 个多 Agent 系统。计算 MARBLE 风格的里程碑指标、随机基线
差值、每里程碑成本，并通过拆分已见/未见任务进行污染检查。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class SystemSim:
    name: str
    base_accuracy: float
    cost_per_task: float
    milestone_completion_rate: float
    training_contamination: float = 0.0  # 在已见任务上的额外准确率
    variance: float = 0.1


@dataclass
class TaskResult:
    task_id: str
    seen_in_training: bool
    accuracy: float
    milestones: int
    cost: float


SYSTEMS = [
    SystemSim("system-A", base_accuracy=0.70, cost_per_task=0.30,
              milestone_completion_rate=0.80, training_contamination=0.20),
    SystemSim("system-B", base_accuracy=0.64, cost_per_task=0.12,
              milestone_completion_rate=0.55, training_contamination=0.0),
    SystemSim("system-C", base_accuracy=0.55, cost_per_task=0.25,
              milestone_completion_rate=0.70, training_contamination=0.0),
]


def run_task(system: SystemSim, task_id: str, seen: bool, rng: random.Random) -> TaskResult:
    base = system.base_accuracy
    if seen:
        base += system.training_contamination
    base = max(0.0, min(1.0, base + rng.uniform(-system.variance, system.variance)))
    success = rng.random() < base
    milestones = 4 if success else int(4 * system.milestone_completion_rate * rng.random())
    return TaskResult(
        task_id=task_id,
        seen_in_training=seen,
        accuracy=1.0 if success else 0.0,
        milestones=milestones,
        cost=system.cost_per_task,
    )


def random_baseline(rng: random.Random) -> float:
    return 0.15  # 此任务族上的随机路由准确率


def run_bench(system: SystemSim, n_seen: int, n_held: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    results_seen: list[TaskResult] = []
    results_held: list[TaskResult] = []
    for i in range(n_seen):
        results_seen.append(run_task(system, f"seen-{i}", True, rng))
    for i in range(n_held):
        results_held.append(run_task(system, f"held-{i}", False, rng))
    return {
        "name": system.name,
        "accuracy_seen": sum(r.accuracy for r in results_seen) / len(results_seen),
        "accuracy_held": sum(r.accuracy for r in results_held) / len(results_held),
        "milestone_rate_seen": sum(r.milestones for r in results_seen) / (len(results_seen) * 4),
        "milestone_rate_held": sum(r.milestones for r in results_held) / (len(results_held) * 4),
        "cost_per_task": system.cost_per_task,
        "cost_per_milestone_held":
            system.cost_per_task / max(0.01, sum(r.milestones for r in results_held) / len(results_held) / 4),
    }


def format_scorecard() -> None:
    print("=" * 78)
    print("基准测试记分卡 — MARBLE 风格里程碑 + 污染检查")
    print("  污染检查：accuracy_seen - accuracy_held（delta > 0.1 时可疑）")
    print("=" * 78)
    print(f"{'系统':10s} {'已见准确率':>10s} {'留出准确率':>10s} {'Δ':>6s} "
          f"{'留出里程碑':>12s} {'每任务成本':>8s} {'每里程碑成本':>10s} {'相对随机':>12s}")

    rng = random.Random(0)
    rand_baseline = random_baseline(rng)
    for sys in SYSTEMS:
        r = run_bench(sys, n_seen=40, n_held=160, seed=17)
        delta = r["accuracy_seen"] - r["accuracy_held"]
        contam_flag = "*" if delta > 0.1 else " "
        vs_random = r["accuracy_held"] - rand_baseline
        print(f"{r['name']:10s} {r['accuracy_seen']:>10.3f} {r['accuracy_held']:>10.3f} "
              f"{delta:>5.2f}{contam_flag} {r['milestone_rate_held']:>12.3f} "
              f"${r['cost_per_task']:>7.2f} ${r['cost_per_milestone_held']:>9.3f} "
              f"+{vs_random:>10.3f}")

    print("\n  * = 污染标记；留出集准确率是规范指标")
    print(f"  随机基线准确率：{rand_baseline:.3f}")


def print_claim_scorecard() -> None:
    print("\n" + "=" * 78)
    print("结论检查清单 — 接受任何多 Agent 结果前请先阅读")
    print("=" * 78)
    checklist = [
        "使用哪个基准和划分？对前沿模型而言，Pro 与 Verified 相差 40 个百分点。",
        "污染检查：该基准是否晚于训练截止日期？",
        "基线比较：对比单 LLM、随机方案还是此前的多 Agent？",
        "统计显著性：N 次试验、p-value、置信区间？",
        "任务多样性：单个任务还是多个任务？能否泛化到其他领域？",
        "成本披露：每任务 token 数、每任务实际耗时？",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"  [{i}] {item}")


def main() -> None:
    format_scorecard()
    print_claim_scorecard()
    print("\n要点：")
    print("  system-A 在已见任务上得分最高，但存在污染信号（delta 较大）。")
    print("  system-B 的每里程碑成本最低；原始准确率最低，但结果透明。")
    print("  system-C 居中且没有污染标记，因此值得信任。")
    print("  按“原始准确率”和“每里程碑成本（留出集）”得到的排名可能大相径庭。")


if __name__ == "__main__":
    main()
