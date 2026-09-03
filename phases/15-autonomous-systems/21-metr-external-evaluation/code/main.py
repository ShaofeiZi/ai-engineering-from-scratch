"""Logistic-fit time-horizon 估计器 —— Python 标准库。

给定合成任务结果（expert_time_hours，成功），对 P(success) 与 log(expert_time)
拟合逻辑曲线，并报告 50/10/90% 水平线。
然后展示 eval-context 作弊对观测值的影响。

仅使用标准库；逻辑曲线拟合是一个最小化的 gradient-descent
实现，面向教学而非生产环境。
"""

from __future__ import annotations

import math
import random


# ---------- 合成数据生成器 ----------

def synth_tasks(true_horizon_hours: float, slope: float = 1.2,
                n: int = 120) -> list[tuple[float, bool]]:
    """生成合成的（expert_time_hours, success）数据对。

    P(success) = sigmoid(slope * (log(true_horizon) - log(expert_time)))。
    """
    log_h = math.log(true_horizon_hours)
    # 专家用时跨度从 0.05 小时到约 48 小时
    out = []
    for _ in range(n):
        t = math.exp(random.uniform(math.log(0.05), math.log(48)))
        logit = slope * (log_h - math.log(t))
        p = 1.0 / (1.0 + math.exp(-logit))
        success = random.random() < p
        out.append((t, success))
    return out


# ---------- 逻辑曲线拟合（微型 GD） ----------

def sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def fit(tasks: list[tuple[float, bool]], iters: int = 4000,
        lr: float = 0.05) -> tuple[float, float]:
    """拟合 P(success) = sigmoid(w * log(t) + b)。返回 (w, b)。"""
    w = 0.0
    b = 0.0
    for _ in range(iters):
        dw = 0.0
        db = 0.0
        n = len(tasks)
        for t, s in tasks:
            y = 1.0 if s else 0.0
            p = sigmoid(w * math.log(t) + b)
            err = p - y
            dw += err * math.log(t)
            db += err
        w -= lr * dw / n
        b -= lr * db / n
    return w, b


def horizon_at(w: float, b: float, p: float) -> float:
    """P(success) = p 对应的专家用时。sigmoid(w*log(t)+b) = p ->
    log(t) = (logit(p) - b) / w。"""
    logit = math.log(p / (1 - p))
    # 零（或 near-zero）斜率意味着成功概率不
    # 依赖于任务长度，因此水平线未定义。选择抛出异常而非
    # 静默返回 inf/nan，使调用方明确感知失败。
    eps = 1e-12
    if abs(w) < eps:
        raise ValueError(
            f"视野未定义：斜率 w={w} 约为 0 "
            f"（b={b}，p={p}，logit={logit}）"
        )
    return math.exp((logit - b) / w)


# ---------- Eval-context 作弊模拟器 ----------

def inject_gaming(tasks: list[tuple[float, bool]],
                  gaming_rate: float) -> list[tuple[float, bool]]:
    """将 `gaming_rate` 比例的失败翻转为成功（模型在评测环境下表现
    更好）。返回一个新列表。"""
    gamed = []
    for t, s in tasks:
        if not s and random.random() < gaming_rate:
            gamed.append((t, True))
        else:
            gamed.append((t, s))
    return gamed


# ---------- 驱动程序 ----------

def report(label: str, w: float, b: float) -> None:
    h50 = horizon_at(w, b, 0.50)
    h10 = horizon_at(w, b, 0.10)
    h90 = horizon_at(w, b, 0.90)
    print(f"  {label:<40}  50%={h50:>6.2f} 小时  "
          f"10%={h10:>6.2f} 小时  90%={h90:>6.2f} 小时")


def main() -> None:
    random.seed(3)
    print("=" * 80)
    print("METR 风格视野估计器（第 15 阶段，第 21 课）")
    print("=" * 80)

    true_h = 14.0
    print(f"\n合成真值：50% 水平线 = {true_h:.1f} 小时")
    print("-" * 80)

    tasks = synth_tasks(true_horizon_hours=true_h, n=160)
    w, b = fit(tasks)
    clean_h50 = horizon_at(w, b, 0.50)
    report("干净评测（无作弊）", w, b)

    gamed_h50: dict[float, float] = {}
    for rate in (0.1, 0.2, 0.4):
        gamed = inject_gaming(tasks, gaming_rate=rate)
        w_g, b_g = fit(gamed)
        gamed_h50[rate] = horizon_at(w_g, b_g, 0.50)
        report(f"评测环境作弊率为 {rate:.0%}", w_g, b_g)

    print()
    print("=" * 80)
    print("要点：视野基于观测成功率拟合；作弊会使其偏移")
    print("-" * 80)
    print(f"  当 seed=3 / n=160 / iters=4000 / true_h={true_h:.1f} 小时时：")
    print(f"    干净拟合          50% 视野 ≈ {clean_h50:>6.2f} 小时"
          f"（真实值 {true_h:.1f}）")
    for rate, h in gamed_h50.items():
        delta = h - true_h
        print(f"    作弊率 {rate:>4.0%}   50% 视野 ≈ {h:>6.2f} 小时"
              f"（相对真实值 {delta:+.2f} 小时）")
    print("  趋势：随着作弊率上升，观测到的 50% 水平线进一步偏离")
    print("  合成真值。具体偏差取决于")
    print("  seed、n、iters 以及所选的 true_h.。没有作弊审计的水平线数字")
    print("  只是部署环境未必能达到的能力上限。")


if __name__ == "__main__":
    main()
