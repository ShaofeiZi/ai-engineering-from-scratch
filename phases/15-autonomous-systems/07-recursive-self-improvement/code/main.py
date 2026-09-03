"""Capability-vs-alignment 竞赛模拟器 — Python 标准库。

每个 RSI 周期有两个复合增长过程。能力增长率为 r_c，对齐
增长率为 r_a，各自带有可配置噪声。模拟器跟踪差距
M(t) = C(t) - A(t) 以及差距将跨越安全
阈值的周期。
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass


DEFAULT_SEED = 11


@dataclass
class Config:
    r_c: float
    r_a: float
    noise_c: float
    noise_a: float
    threshold: float = 1.5


def run(cycles: int, cfg: Config) -> list[tuple[int, float, float, float]]:
    c = 1.0
    a = 1.0
    out = [(0, c, a, c - a)]
    for cyc in range(1, cycles + 1):
        nc = cfg.r_c + random.gauss(0, cfg.noise_c)
        na = cfg.r_a + random.gauss(0, cfg.noise_a)
        c *= max(0.9, nc)
        a *= max(0.9, na)
        out.append((cyc, c, a, c - a))
    return out


def crossing_cycle(trajectory, threshold: float) -> int:
    for cyc, _c, _a, gap in trajectory:
        if gap >= threshold:
            return cyc
    return -1


def print_trajectory(label: str, cfg: Config, cycles: int = 40) -> None:
    traj = run(cycles, cfg)
    print(f"\n{label}")
    print(f"  r_c={cfg.r_c:.2f} r_a={cfg.r_a:.2f} "
          f"noise_c={cfg.noise_c:.3f} noise_a={cfg.noise_a:.3f}")
    print(f"  阈值 (C - A): {cfg.threshold:.2f}")
    print(f"  {'cycle':>6}  {'C(t)':>8}  {'A(t)':>8}  {'C-A':>8}  标志")
    # 打印大约九个快照，始终包含第 0 周期和 cycles 周期，
    # 这样修改 `cycles`（e.g. 用于练习）时不会悄悄丢失行。
    step = max(1, cycles // 8)
    for cyc, c, a, gap in traj:
        if cyc == 0 or cyc == cycles or cyc % step == 0:
            flag = "PAUSE" if gap >= cfg.threshold else "ok"
            print(f"  {cyc:>6}  {c:>8.2f}  {a:>8.2f}  {gap:>+8.2f}  {flag}")
    cross = crossing_cycle(traj, cfg.threshold)
    if cross >= 0:
        print(f"  -> 在第 {cross} 周期跨越阈值")
    else:
        print("  -> 在模拟窗口内未跨越阈值")


def monte_carlo(cfg: Config, cycles: int, trials: int) -> None:
    crossings = []
    for _ in range(trials):
        traj = run(cycles, cfg)
        cross = crossing_cycle(traj, cfg.threshold)
        if cross >= 0:
            crossings.append(cross)
    print(f"\n  monte-carlo 在 {trials} 次试验中，每次 {cycles} 个周期")
    print(f"  跨越次数: {len(crossings)} ({len(crossings)/trials:.0%})")
    if crossings:
        avg = sum(crossings) / len(crossings)
        p50 = statistics.median(crossings)
        print(f"  平均跨越周期: {avg:.1f}")
        print(f"  中位跨越周期: {p50}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="暂停差距阈值 C - A（默认值：%(default)s）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG 种子（默认值：%(default)s）")
    args = parser.parse_args()

    random.seed(args.seed)
    th = args.threshold
    print("=" * 70)
    print("CAPABILITY 对比 ALIGNMENT RACE（第 15 阶段，第 7 课）")
    print("=" * 70)

    # 场景 A：能力适度领先于对齐
    print_trajectory(
        "场景 A——能力增长快于对齐",
        Config(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03, threshold=th),
    )

    # 场景 B：对齐保持同步
    print_trajectory(
        "场景 B——增长率相同（仅由噪声导致漂移）",
        Config(r_c=1.10, r_a=1.10, noise_c=0.02, noise_a=0.03, threshold=th),
    )

    # 场景 C：对齐率更高，但能力有突发激增
    print_trajectory(
        "场景 C——对齐平均增长率更高，但能力出现激增",
        Config(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01, threshold=th),
    )

    print("\n场景 A 的 Monte-Carlo")
    monte_carlo(
        Config(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03, threshold=th),
        cycles=30, trials=500,
    )
    print("\n场景 C 的 Monte-Carlo")
    monte_carlo(
        Config(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01, threshold=th),
        cycles=30, trials=500,
    )

    print()
    print("=" * 70)
    print("要点：微小的增长率差异复合累积，导致越过安全阈值")
    print("-" * 70)
    print("  场景 A 在不到 10 个周期内跨越绝对 1.5 的差距 (C - A)。")
    print("  场景 B 保持有界——均值增长率相同，仅由噪声导致漂移。")
    print("  场景 C：如果能力出现大幅激增，更高的对齐均值也无法避免风险。")
    print("  噪声的影响与漂移一样大。")
    print("  RSI-style 管道需要内嵌 pause-on-gap 阈值。")


if __name__ == "__main__":
    main()
