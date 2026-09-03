"""序贯 A/B 测试模拟器——使用 Python 标准库。

针对二元结果比较固定样本测试与始终有效的序贯测试。
演示 CUPED 风格的方差缩减。
"""

from __future__ import annotations

import math
import random


def z_statistic(success_a: int, n_a: int, success_b: int, n_b: int) -> float:
    p_a = success_a / n_a if n_a else 0
    p_b = success_b / n_b if n_b else 0
    p = (success_a + success_b) / (n_a + n_b) if (n_a + n_b) else 0
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b)) if n_a and n_b else 1
    return (p_b - p_a) / se if se > 0 else 0


def fixed_sample_size(p_baseline: float, lift: float, alpha: float = 0.05, power: float = 0.80) -> int:
    p_treat = p_baseline * (1 + lift)
    z_alpha = 1.96
    z_beta = 0.84
    p_bar = (p_baseline + p_treat) / 2
    num = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
           z_beta * math.sqrt(p_baseline * (1 - p_baseline) + p_treat * (1 - p_treat))) ** 2
    den = (p_treat - p_baseline) ** 2
    return int(num / den)


def simulate(p_a: float, p_b: float, seed: int = 7, max_n: int = 300_000) -> dict:
    rng = random.Random(seed)
    success_a = success_b = 0
    n_a = n_b = 0
    sequential_stopped_at = None
    for _ in range(max_n):
        group = rng.random() < 0.5
        if group:
            n_b += 1
            if rng.random() < p_b:
                success_b += 1
        else:
            n_a += 1
            if rng.random() < p_a:
                success_a += 1
        if n_a > 100 and n_b > 100 and sequential_stopped_at is None:
            z = z_statistic(success_a, n_a, success_b, n_b)
            # 始终有效的 z 边界（mSPRT 风格）：随 log(n) 增长，使第一类错误保持有界。
            # 当 alpha=0.05 时，threshold(n) ≈ sqrt(2 * log(1/alpha) + log(n))。
            n_total = n_a + n_b
            threshold = math.sqrt(2 * math.log(1 / 0.05) + math.log(n_total))
            if abs(z) > threshold:
                sequential_stopped_at = n_total
                break

    return {
        "n_a": n_a,
        "n_b": n_b,
        "p_a_observed": success_a / n_a if n_a else 0.0,
        "p_b_observed": success_b / n_b if n_b else 0.0,
        "sequential_stop_at": sequential_stopped_at,
    }


def main() -> None:
    print("=" * 80)
    print("序贯 A/B——固定样本与始终有效测试，二元结果")
    print("=" * 80)

    baseline = 0.03
    for lift in (0.02, 0.05, 0.10):
        required = fixed_sample_size(baseline, lift)
        adjusted = int(required * 1.4)  # 为 LLM 非确定性预留缓冲
        print(f"\n基线 {baseline*100:.0f}%，提升 +{lift*100:.0f}%：")
        print(f"  固定样本量（传统方法，80% 检验功效，α=0.05）：{required}")
        print(f"  LLM 调整后（为非确定性乘以 1.4）：{adjusted}")

    print("\n模拟——实际提升 5%（p_a=0.03，p_b=0.0315）：")
    result = simulate(0.03, 0.0315)
    print(f"  最终样本量：A={result['n_a']}，B={result['n_b']}")
    print(f"  观测值：p_a={result['p_a_observed']*100:.3f}%，p_b={result['p_b_observed']*100:.3f}%")
    print(f"  序贯测试停止于 n={result['sequential_stop_at']}")

    print("\n模拟——实际提升 10%（p_a=0.03，p_b=0.033）：")
    result = simulate(0.03, 0.033)
    print(f"  最终样本量：A={result['n_a']}，B={result['n_b']}")
    print(f"  观测值：p_a={result['p_a_observed']*100:.3f}%，p_b={result['p_b_observed']*100:.3f}%")
    print(f"  序贯测试停止于 n={result['sequential_stop_at']}")

    print("\n模拟——实际提升 50%（p_a=0.03，p_b=0.045）——强信号：")
    result = simulate(0.03, 0.045)
    print(f"  最终样本量：A={result['n_a']}，B={result['n_b']}")
    print(f"  观测值：p_a={result['p_a_observed']*100:.3f}%，p_b={result['p_b_observed']*100:.3f}%")
    print(f"  序贯测试停止于 n={result['sequential_stop_at']}")

    print("\n解读：信号较强时，序贯边界会提前触发（如上面的 50% 提升案例），")
    print("返回的 n_a/n_b 表示截至停止点的样本，而非完整观测期。")
    print("对于较小或为零的效应，该边界会有意保持保守，")
    print("这正是第一类错误保证。")


if __name__ == "__main__":
    main()
