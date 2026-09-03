"""METR-style time-horizon 模拟器 — 标准库 Python实现。

给定一个倍增时间和基线视野，预测跨越未来年份的 50% task-completion
视野。此外，展示 per-step 可靠性如何沿轨迹复合累积：一个 99% per-step
智能体在 70 步任务上仍然只有抛硬币的成功率。

用于教学，未经校准。目的是让你在信任智能体自主运行之前，
先在脑中建立起对这些数字的直觉。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HorizonConfig:
    baseline_hours: float
    baseline_month: int  # 自纪元起的月数（0 = 当前）
    doubling_months: float


def horizon_at(cfg: HorizonConfig, months_from_now: int) -> float:
    """预测给定月份偏移处的 50% 视野。"""
    delta = months_from_now - cfg.baseline_month
    return cfg.baseline_hours * (2 ** (delta / cfg.doubling_months))


def months_to_cross(cfg: HorizonConfig, target_hours: float) -> float:
    """视野达到 target_hours. 所需的月数"""
    ratio = target_hours / cfg.baseline_hours
    return cfg.baseline_month + cfg.doubling_months * math.log2(ratio)


def end_to_end_reliability(per_step: float, steps: int) -> float:
    """每一步依次全部成功的概率。"""
    return per_step ** steps


def max_steps_for_target(per_step: float, target: float) -> int:
    """满足 per_step**N >= 目标 的最大 N。"""
    if per_step >= 1.0:
        return 10**9
    return math.floor(math.log(target) / math.log(per_step))


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.1f} 分钟"
    if h < 24:
        return f"{h:.1f} 小时"
    return f"{h / 24:.1f} 天"


def horizon_projection() -> None:
    """使用 METR 拟合的斜率向前绘制视野。"""
    cfg = HorizonConfig(
        baseline_hours=14.0,
        baseline_month=0,
        doubling_months=7.0,
    )
    print("\nMETR-style 视野预测")
    print("-" * 70)
    print(f"  基线：第 0 个月为 {cfg.baseline_hours:.1f} 小时"
          f"（Claude Opus 4.6，2026 年 1 月）")
    print(f"  倍增时间：{cfg.doubling_months:.1f} 个月")
    print()
    print(f"  {'月份':>8}  {'视野':>12}  {'含义':<30}")
    for m in (0, 6, 12, 18, 24, 30, 36):
        h = horizon_at(cfg, m)
        tag = ""
        if h < 24:
            tag = "工作日尺度"
        elif h < 168:
            tag = "多日任务"
        elif h < 720:
            tag = "周尺度"
        else:
            tag = "月尺度"
        print(f"  {m:>8}  {fmt_hours(h):>12}  {tag:<30}")

    print()
    print("  目标交叉点")
    for target in (24, 48, 168, 720):
        m = months_to_cross(cfg, target)
        print(f"    {fmt_hours(target)}：第 {m:.1f} 个月")


def reliability_compounding() -> None:
    """展示 per-step 可靠性如何沿轨迹衰减。"""
    print("\nPer-step 可靠性 -> end-to-end 可靠性")
    print("-" * 70)
    print(f"  {'单步':>10}  {'步数':>8}  {'端到端':>12}  "
          f"{'标志':<20}")
    cases = [
        (0.90, 10),
        (0.90, 50),
        (0.95, 50),
        (0.99, 50),
        (0.99, 70),
        (0.99, 200),
        (0.995, 200),
        (0.999, 1000),
    ]
    for per_step, steps in cases:
        p = end_to_end_reliability(per_step, steps)
        flag = ""
        if p < 0.5:
            flag = "不高于抛硬币"
        elif p < 0.8:
            flag = "不适合生产"
        elif p < 0.95:
            flag = "脆弱"
        else:
            flag = "正常"
        print(f"  {per_step:>10.3f}  {steps:>8}  {p:>12.1%}  {flag:<20}")

    print()
    print("  50% end-to-end 成功的最大轨迹长度")
    for per_step in (0.90, 0.95, 0.99, 0.995, 0.999):
        n = max_steps_for_target(per_step, 0.50)
        print(f"    per-step {per_step:.3f}：最多 {n} 步")


def deploy_gap_note() -> None:
    """Eval-context-gaming 调整。"""
    print("\nEval-vs-deploy 调整")
    print("-" * 70)
    print("  METR 数据假设理想工具、无后果、")
    print("  且无 eval-context 博弈。Anthropic 2024 年的 alignment-faking")
    print("  研究发现 Claude 在基础测试中有 12% 伪造行为，")
    print("  在重训练尝试后比例高达 78%。")
    print()
    for horizon in (14.0, 48.0, 168.0):
        for gap in (0.0, 0.2, 0.4):
            effective = horizon * (1 - gap)
            print(f"  基准 {fmt_hours(horizon):>7}  "
                  f"差距 {gap:.0%}  ->  部署 "
                  f"{fmt_hours(effective):>7}")


def main() -> None:
    print("=" * 70)
    print("METR 时间视野与复合可靠性（第 15 阶段，第 1 课）")
    print("=" * 70)
    horizon_projection()
    reliability_compounding()
    deploy_gap_note()
    print()
    print("=" * 70)
    print("要点：视野指数增长，可靠性复合累积")
    print("-" * 70)
    print("  按照每 7 个月翻倍，multi-day 视野距现在约 1 年。")
    print("  在 99% 的 per-step 下，70 步轨迹已经只剩抛硬币的概率。")
    print("  两个数字同时重要。设计时需兼顾两者。")


if __name__ == "__main__":
    main()
