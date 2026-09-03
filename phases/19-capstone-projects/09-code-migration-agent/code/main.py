"""代码迁移智能体——确定性 recipe + 智能体循环回退脚手架。

关键架构原语是两层结构：首先执行确定性 recipe（快速、可审计、安全），然后针对
剩余失败运行有硬预算限制的智能体循环，并由失败分类步骤向分类看板提供数据。
此脚手架实现这两层，并运行包含多种结果的 50 个仓库模拟。

运行：python main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 仓库 + 失败分类法
# ---------------------------------------------------------------------------

FAILURE_CLASSES = [
    "dep_upgrade_required",
    "build_tool_drift",
    "custom_annotation",
    "test_flake",
    "syntax_edge_case",
    "budget_exhausted",
    "coverage_regression",
]


@dataclass
class Repo:
    name: str
    loc: int
    lang: str          # "java" | "python"
    hardness: float    # 0..1


@dataclass
class Attempt:
    repo: Repo
    recipe_applied: int = 0
    agent_turns: int = 0
    cost_usd: float = 0.0
    wall_min: float = 0.0
    status: str = "pending"  # "pass" | "fail"
    failure_class: str | None = None
    coverage_base: float = 80.0
    coverage_final: float = 80.0


# ---------------------------------------------------------------------------
# 确定性 recipe 阶段——OpenRewrite / libcst 替代实现
# ---------------------------------------------------------------------------

def run_recipes(repo: Repo) -> int:
    """返回已应用的重写次数。"""
    base = 20 + int(repo.loc / 500)
    return int(base * (1 - 0.2 * repo.hardness))


# ---------------------------------------------------------------------------
# 智能体循环——分类失败、应用修复、重试；感知预算
# ---------------------------------------------------------------------------

BUDGET_MIN = 30.0
BUDGET_USD = 8.0
BUDGET_TURNS = 20


def agent_loop(attempt: Attempt, rng: random.Random) -> None:
    """模拟 plan-act 循环，直到通过或预算耗尽。"""
    # 每轮成本随难度变化
    per_turn_min = 2.8 + attempt.repo.hardness * 2.0
    per_turn_usd = 0.45 + attempt.repo.hardness * 0.65

    # 每轮通过概率取决于难度（0.02-0.18）
    turn_pass_p = max(0.02, 0.22 * (1 - attempt.repo.hardness * 0.95))

    while True:
        if attempt.agent_turns >= BUDGET_TURNS:
            attempt.status = "fail"
            attempt.failure_class = "budget_exhausted"
            return
        if attempt.wall_min >= BUDGET_MIN or attempt.cost_usd >= BUDGET_USD:
            attempt.status = "fail"
            attempt.failure_class = "budget_exhausted"
            return

        attempt.agent_turns += 1
        attempt.wall_min += per_turn_min
        attempt.cost_usd += per_turn_usd

        if rng.random() < turn_pass_p:
            # 覆盖率检查
            delta = rng.gauss(0.0, 0.6)
            attempt.coverage_final = attempt.coverage_base + delta
            if attempt.coverage_final < attempt.coverage_base - 2.0:
                attempt.status = "fail"
                attempt.failure_class = "coverage_regression"
                return
            attempt.status = "pass"
            return


# ---------------------------------------------------------------------------
# 卡住仓库的分类——归入分类法中的类别
# ---------------------------------------------------------------------------

def classify_failure(rng: random.Random) -> str:
    """智能体失败分类器的替代实现。真实实现会读取构建日志和测试输出。"""
    weights = {
        "dep_upgrade_required": 0.30,
        "build_tool_drift": 0.20,
        "custom_annotation": 0.18,
        "test_flake": 0.15,
        "syntax_edge_case": 0.17,
    }
    r = rng.random()
    acc = 0.0
    for cls, w in weights.items():
        acc += w
        if r <= acc:
            return cls
    return "syntax_edge_case"


# ---------------------------------------------------------------------------
# 流水线——依次执行 recipe、智能体，再生成 PR/文件结果
# ---------------------------------------------------------------------------

def migrate(repo: Repo, rng: random.Random) -> Attempt:
    attempt = Attempt(repo=repo)
    attempt.recipe_applied = run_recipes(repo)

    # 简单仓库通常在 recipe 阶段后直接通过
    straight_through_p = 0.55 * (1 - repo.hardness)
    if rng.random() < straight_through_p:
        delta = rng.gauss(0.0, 0.4)
        attempt.coverage_final = attempt.coverage_base + delta
        attempt.status = "pass"
        attempt.wall_min = 3.0 + rng.random() * 4
        attempt.cost_usd = 0.30
        return attempt

    # 否则运行智能体循环
    agent_loop(attempt, rng)

    if attempt.status == "fail" and attempt.failure_class == "budget_exhausted":
        # 对预算耗尽的根因进行分类
        if rng.random() < 0.75:
            attempt.failure_class = classify_failure(rng)
    return attempt


# ---------------------------------------------------------------------------
# 50 个仓库的模拟
# ---------------------------------------------------------------------------

def synth_bench(rng: random.Random) -> list[Repo]:
    bench: list[Repo] = []
    for i in range(50):
        lang = "java" if rng.random() < 0.6 else "python"
        hardness = min(0.95, max(0.05, rng.gauss(0.65, 0.18)))
        bench.append(Repo(name=f"repo-{i:02d}-{lang}",
                          loc=rng.randint(800, 40_000),
                          lang=lang,
                          hardness=hardness))
    return bench


def main() -> None:
    rng = random.Random(19)
    bench = synth_bench(rng)

    results: list[Attempt] = []
    for repo in bench:
        results.append(migrate(repo, rng))

    passed = [a for a in results if a.status == "pass"]
    failed = [a for a in results if a.status == "fail"]

    print(f"=== migration-bench 运行（50 个仓库）===")
    print(f"通过：{len(passed):2d}  ({len(passed) / 50:.1%})")
    print(f"失败：{len(failed):2d}")

    print("\n失败分类：")
    taxonomy: dict[str, int] = {}
    for a in failed:
        taxonomy[a.failure_class or "unknown"] = taxonomy.get(a.failure_class or "unknown", 0) + 1
    for cls, n in sorted(taxonomy.items(), key=lambda x: -x[1]):
        print(f"  {cls:24s} {n}")

    if passed:
        mean_cost = sum(a.cost_usd for a in passed) / len(passed)
        mean_min = sum(a.wall_min for a in passed) / len(passed)
        mean_turns = sum(a.agent_turns for a in passed) / len(passed)
        mean_cov_delta = sum(a.coverage_final - a.coverage_base for a in passed) / len(passed)
        print("\n通过集合指标：")
        print(f"  每仓库平均成本：${mean_cost:.2f}")
        print(f"  平均墙钟分钟数：{mean_min:.1f}")
        print(f"  平均智能体轮次：{mean_turns:.1f}")
        print(f"  平均覆盖率变化：{mean_cov_delta:+.2f} 个百分点")


if __name__ == "__main__":
    main()
