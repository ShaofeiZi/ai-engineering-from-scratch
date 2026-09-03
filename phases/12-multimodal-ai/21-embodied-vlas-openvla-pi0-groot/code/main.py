"""具身 VLA 动作格式玩具 — 标准库。

三个小型实现：
  1. 离散区间动作分词（RT-2 / OpenVLA）。
  2. 一个 FAST 风格的 DCT 量化压缩器。
  3. 比较离散、FAST 与连续流的 token 数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def discretize(action: list[float], bins: int = 256) -> list[int]:
    """将 [-1,1]^D 动作映射到 D 个整数区间。"""
    tokens = []
    for a in action:
        idx = int((a + 1) / 2 * (bins - 1))
        idx = max(0, min(bins - 1, idx))
        tokens.append(idx)
    return tokens


def undiscretize(tokens: list[int], bins: int = 256) -> list[float]:
    return [(2 * t / (bins - 1)) - 1 for t in tokens]


def dct(x: list[float]) -> list[float]:
    """朴素 II 型 DCT。"""
    n = len(x)
    out = []
    for k in range(n):
        s = 0.0
        for i in range(n):
            s += x[i] * math.cos(math.pi / n * (i + 0.5) * k)
        out.append(s)
    return out


def fast_compress(trajectory: list[list[float]], keep_coeff: int = 4,
                  bins: int = 32) -> list[int]:
    """FAST 风格分词器：逐维 DCT + 保留低频系数 + 量化。
    轨迹：动作列表（浮点数列表），形状 (T, D)。
    返回一个扁平的整数 token 列表。"""
    if not trajectory:
        return []
    D = len(trajectory[0])
    tokens = []
    for d in range(D):
        series = [step[d] for step in trajectory]
        coeffs = dct(series)[:keep_coeff]
        for c in coeffs:
            c_norm = max(-1.0, min(1.0, c / len(series)))
            idx = int((c_norm + 1) / 2 * (bins - 1))
            tokens.append(idx)
    return tokens


def compare_formats() -> None:
    T = 30
    D = 10
    trajectory = [[math.sin(0.1 * t + 0.3 * d) for d in range(D)] for t in range(T)]

    print("\n动作 TOKEN 数（30 步轨迹，10 自由度）")
    print("-" * 60)
    per_step_discrete = len(discretize(trajectory[0]))
    total_discrete = per_step_discrete * T
    fast_tokens = fast_compress(trajectory, keep_coeff=4)
    total_fast = len(fast_tokens)
    continuous_flow_count = 1
    rows = [
        ("离散 256 区间（RT-2）",      total_discrete, "逐步自回归"),
        ("FAST 每维 4 个系数",         total_fast,     "序列压缩器"),
        ("流匹配（pi0）",              continuous_flow_count, "单头输出"),
    ]
    for name, count, note in rows:
        print(f"  {name:<28}  {count:>6} 个 token   ({note})")
    print(f"\n  加速：FAST ~{total_discrete / total_fast:.1f}x 相比离散区间")


def round_trip_demo() -> None:
    print("\n往返转换：10 自由度动作经过离散化与反离散化")
    print("-" * 60)
    action = [0.1, -0.5, 0.25, -0.75, 0.9, -0.1, 0.0, 0.33, -0.67, 0.5]
    tokens = discretize(action, bins=256)
    recovered = undiscretize(tokens, bins=256)
    print(f"  原始值    ：{[round(a, 3) for a in action]}")
    print(f"  token     ：{tokens}")
    print(f"  恢复值    ：{[round(r, 3) for r in recovered]}")
    max_err = max(abs(a - r) for a, r in zip(action, recovered))
    print(f"  最大绝对误差：{max_err:.4f}  （区间宽度 = 2/255 ~ 0.0078）")


def lineage_table() -> None:
    print("\nVLA 演进路线")
    print("-" * 60)
    rows = [
        ("RT-2",       "2023", "PaLM-X + 离散区间",       "闭源"),
        ("OpenVLA",    "2024", "Llama 7B + 离散区间",     "开放"),
        ("Octo",       "2024", "小型扩散头",             "开放"),
        ("pi0",        "2024", "流匹配头",               "开放"),
        ("pi0-FAST",   "2025", "流 + FAST 分词器",       "开放"),
        ("GR00T N1",   "2025", "双系统人形机器人",       "开放"),
        ("GR00T N1.7", "2025", "仿真到现实的数据规模化", "开放"),
    ]
    print(f"  {'模型':<12}{'年份':<6}{'模式':<28}{'开放/闭源'}")
    for r in rows:
        print(f"  {r[0]:<12}{r[1]:<6}{r[2]:<28}{r[3]}")


def main() -> None:
    print("=" * 60)
    print("具身 VLA（第 12 阶段，第 21 课）")
    print("=" * 60)

    round_trip_demo()
    compare_formats()
    lineage_table()

    print("\n联合微调比例（网络 VQA：机器人轨迹）")
    print("-" * 60)
    print("  RT-2       ：~1:1")
    print("  OpenVLA    ：网络数据与机器人数据之比约 0.5:1")
    print("  pi0        ：类似的比例")
    print("  VQA 过多 -> 遗忘动作；机器人数据过多 -> 丢失语言")


if __name__ == "__main__":
    main()
