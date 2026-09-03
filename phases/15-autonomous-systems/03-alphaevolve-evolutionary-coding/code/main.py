"""最小化 AlphaEvolve-like 进化循环 — 纯标准库 Python。

玩具级符号回归。"LLM" 对候选表达式提出一个小变异
（修改常量、修改运算符、增加一项）。
"evaluator" 在训练集和 held-out 测试点上对表达式评分。

MAP-elites 网格保持候选多样性：以（表达式深度、常量量级桶）为键的单元格。
没有 held-out 划分时循环会激进过拟合；有了划分后最佳候选能泛化。
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


DEFAULT_SEED = 1


# 循环试图重新发现的目标函数。
def target(x: float) -> float:
    return 2.0 * x * x + 3.0 * x - 1.0


Expr = tuple  # 递归: ("num", v) | ("x",) | ("add", a, b) | ("mul", a, b)


def evaluate_expr(e: Expr, x: float) -> float:
    tag = e[0]
    if tag == "num":
        return float(e[1])
    if tag == "x":
        return x
    if tag == "add":
        return evaluate_expr(e[1], x) + evaluate_expr(e[2], x)
    if tag == "mul":
        return evaluate_expr(e[1], x) * evaluate_expr(e[2], x)
    raise ValueError(tag)


def depth(e: Expr) -> int:
    tag = e[0]
    if tag in ("num", "x"):
        return 1
    return 1 + max(depth(e[1]), depth(e[2]))


def max_const(e: Expr) -> float:
    tag = e[0]
    if tag == "num":
        return abs(e[1])
    if tag == "x":
        return 0.0
    return max(max_const(e[1]), max_const(e[2]))


def mutate(e: Expr) -> Expr:
    """代替 LLM 执行定向编辑。"""
    choice = random.random()
    if choice < 0.25:
        return random_leaf()
    if choice < 0.5:
        return ("add", e, random_leaf())
    if choice < 0.75:
        return ("mul", e, random_leaf())
    # 在某处扰动一个常量
    return perturb(e)


def perturb(e: Expr) -> Expr:
    tag = e[0]
    if tag == "num":
        return ("num", e[1] + random.choice([-1.0, -0.5, 0.5, 1.0]))
    if tag == "x":
        return e
    return (tag, perturb(e[1]), e[2]) if random.random() < 0.5 else (tag, e[1], perturb(e[2]))


def random_leaf() -> Expr:
    if random.random() < 0.5:
        return ("x",)
    return ("num", float(random.choice([-2, -1, 0, 1, 2, 3])))


def render(e: Expr) -> str:
    tag = e[0]
    if tag == "num":
        return f"{e[1]:g}"
    if tag == "x":
        return "x"
    op = "+" if tag == "add" else "*"
    return f"({render(e[1])} {op} {render(e[2])})"


def mse(e: Expr, xs: list[float]) -> float:
    total = 0.0
    for x in xs:
        try:
            y = evaluate_expr(e, x)
        except (OverflowError, ValueError):
            return float("inf")
        total += (y - target(x)) ** 2
    return total / max(1, len(xs))


@dataclass
class Candidate:
    expr: Expr
    train_score: float
    test_score: float
    generation: int


def cell_key(e: Expr) -> tuple[int, int]:
    d = min(depth(e), 6)
    c = min(int(max_const(e) / 2), 4)
    return (d, c)


def seed_candidate(test_xs: list[float], train_xs: list[float], gen: int) -> Candidate:
    e = random_leaf()
    return Candidate(e, mse(e, train_xs), mse(e, test_xs), gen)


def run_loop(
    generations: int,
    pop: int,
    use_holdout: bool,
    seed: int | None = None,
) -> tuple[Candidate, list[float], list[float]]:
    if seed is not None:
        random.seed(seed)
    train_xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    test_xs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    def signal_of(c: Candidate) -> float:
        return 0.5 * (c.train_score + c.test_score) if use_holdout else c.train_score

    archive: dict[tuple[int, int], Candidate] = {}
    for _ in range(pop):
        c = seed_candidate(test_xs, train_xs, 0)
        key = cell_key(c.expr)
        incumbent = archive.get(key)
        if incumbent is None or signal_of(c) < signal_of(incumbent):
            archive[key] = c

    best_trace: list[float] = []
    test_trace: list[float] = []
    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child_expr = mutate(parent.expr)
        tr = mse(child_expr, train_xs)
        te = mse(child_expr, test_xs)
        child = Candidate(child_expr, tr, te, g)
        key = cell_key(child_expr)
        incumbent = archive.get(key)
        if incumbent is None or signal_of(child) < signal_of(incumbent):
            archive[key] = child

        best = min(archive.values(), key=lambda c: c.train_score)
        best_trace.append(best.train_score)
        test_trace.append(best.test_score)

    # 最终选择必须使用与搜索相同的信号：在这里
    # 当 use_holdout=False 时使用 held-out 测试会静默泄露
    # 验证集回 Run B，掩盖课程所展示的过拟合。
    best = min(archive.values(), key=signal_of)
    return best, best_trace, test_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="跳过留出测试评估器（仅运行 B；强制展示奖励破解）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ALPHAEVOLVE-STYLE LOOP（第 15 阶段，第 3 课）")
    print("=" * 70)
    print("目标: 2x^2 + 3x - 1")

    if not args.no_holdout:
        print("\n运行 A：评估器信号中包含留出测试")
        best, train_trace, _ = run_loop(
            generations=1500, pop=20, use_holdout=True, seed=DEFAULT_SEED
        )
        print(f"  最佳表达式 : {render(best.expr)}")
        print(f"  训练 MSE : {best.train_score:.4f}")
        print(f"  测试  MSE : {best.test_score:.4f}")
        print(f"  代次: {best.generation}")
        print("  进度：第 100 代训练={:.3f}，第 500 代训练={:.3f}，第 1500 代训练={:.3f}".format(
            train_trace[99], train_trace[499], train_trace[-1]))

    print("\n运行 B：无留出测试（仅训练集评估器 -> 奖励破解风险）")
    best, _train_trace, _test_trace = run_loop(
        generations=1500, pop=20, use_holdout=False, seed=DEFAULT_SEED
    )
    print(f"  最佳表达式 : {render(best.expr)}")
    print(f"  训练 MSE : {best.train_score:.4f}")
    print(f"  测试  MSE : {best.test_score:.4f}")
    print(f"  代次: {best.generation}")
    gap = best.test_score - best.train_score
    print(f"  训练集到测试集的差距：{gap:+.4f}（大差距 = 过拟合/奖励破解代理）")

    print()
    print("=" * 70)
    print("要点：评估器即架构")
    print("-" * 70)
    print("  运行 A 的训练 MSE 和测试 MSE 都收敛到较低水平。")
    print("  运行 B 的训练 MSE 收敛到较低水平；测试 MSE 仍然不稳定或更差。")
    print("  留出评估器决定了结果是发现还是奖励破解。")
    print("  AlphaEvolve 的优势来自存在此类")
    print("  评估器的领域。挑选那些领域才是困难之处。")


if __name__ == "__main__":
    main()
