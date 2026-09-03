"""Many-shot 越狱玩具示例——仅使用 Python 标准库。

目标：一种过滤器，其拒绝概率会随着上下文中服从样本对数量的增加而按幂律
衰减。无需训练模型即可复现 Anil 等人 2024 年论文图 2 的曲线形态。

用法：python3 code/main.py
"""

from __future__ import annotations

import math
import random


random.seed(41)


def target_asr(n_shots: int, alpha: float = 0.5, a0: float = 0.02) -> float:
    """目标的攻击成功率是 shot 数量的函数。
    幂律形态：ASR(n) = min(1, a0 + c * n^alpha)。

    这是 Anil 等人 2024 观察到的经验模式：5-shot 时稳定失败，约 32-shot
    开始成功，约 256-shot 时趋于饱和。
    """
    if n_shots <= 0:
        return 0.0
    c = 0.03
    return min(1.0, a0 + c * (n_shots ** alpha))


def defense_adjusted(n_shots: int, alpha: float = 0.5) -> float:
    """一种简单防御：分类器检测 many-shot 模式，并将有效 shot 数上限设为 16。
    ASR 曲线在 16-shot 的值处饱和。"""
    eff = min(n_shots, 16)
    return target_asr(eff, alpha)


def simulate(n_shots: int, asr_fn, trials: int = 500) -> float:
    p = asr_fn(n_shots)
    hits = sum(1 for _ in range(trials) if random.random() < p)
    return hits / trials


def fit_power_law(shots: list[int], asrs: list[float]) -> tuple[float, float]:
    """简单的 log-log 线性回归：log(ASR) = log(c) + alpha * log(n)。"""
    xs = [math.log(s) for s in shots if s > 0]
    ys = [math.log(max(a, 1e-4)) for a in asrs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    den = sum((xi - mx) ** 2 for xi in xs)
    alpha = num / den
    logc = my - alpha * mx
    return alpha, math.exp(logc)


def main() -> None:
    print("=" * 70)
    print("MANY-SHOT 越狱玩具示例（阶段 18，第 13 课）")
    print("=" * 70)

    shots = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print("\n-- 无防御目标（幂律 ASR 曲线）--")
    undef = []
    for s in shots:
        rate = simulate(s, target_asr)
        undef.append(rate)
        print(f"  shot 数={s:4d}   ASR={rate:.3f}")
    alpha, c = fit_power_law(shots, undef)
    print(f"\n  拟合幂律：ASR ~= {c:.3f} * n^{alpha:.3f}")

    print("\n-- 分类器防御目标（有效 shot 数上限为 16）--")
    for s in shots:
        rate = simulate(s, defense_adjusted)
        print(f"  shot 数={s:4d}   ASR={rate:.3f}")

    print("\n" + "=" * 70)
    print("要点：ASR 随 shot 数按幂律增长，防御机制限制了有效 shot 数。")
    print("要在保留无害 ICL 的同时抑制有害 ICL，需要分类器在上下文层面区分二者。")
    print("因此，基于分类器的提示词修改（Anthropic 2024）能在不破坏 ICL 的情况下")
    print("将成功率从 61% 降至 2%。")
    print("=" * 70)


if __name__ == "__main__":
    main()
