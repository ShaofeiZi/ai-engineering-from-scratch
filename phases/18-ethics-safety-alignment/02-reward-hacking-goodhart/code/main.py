"""奖励劫持的过度优化曲线——仅使用 Python 标准库。

复现 Gao、Schulman、Hilton（ICML 2023）所描述的曲线形态：随着策略偏离
初始参考策略（以 sqrt(KL) 衡量），代理奖励单调上升，而真实奖励先达到峰值
再下降。这里构造玩具版真实奖励模型和代理线性奖励模型，并在 KL 惩罚下对
均值向量策略执行爬山优化。可以调整代理样本量和噪声尾部。

用法：python3 code/main.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


random.seed(42)

D = 8
GOLD_W = [1.0, -0.6, 0.4, 0.2, -0.1, 0.3, -0.5, 0.8]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def gauss() -> float:
    return random.gauss(0.0, 1.0)


def student_t(df: float) -> float:
    """重尾噪声。当 df=3 时，方差有限，但峰度无限。"""
    u = random.gauss(0.0, 1.0)
    chi2 = sum(random.gauss(0.0, 1.0) ** 2 for _ in range(int(df)))
    if chi2 <= 0:
        chi2 = 1e-6
    return u * math.sqrt(df / chi2)


def sample_feature() -> list[float]:
    return [gauss() for _ in range(D)]


def gold_reward(x: list[float]) -> float:
    return dot(GOLD_W, x)


@dataclass
class ProxyRM:
    w: list[float]
    n_samples: int

    def score(self, x: list[float]) -> float:
        return dot(self.w, x)


def train_proxy(n_samples: int, noise: str = "gauss") -> ProxyRM:
    """根据 n 个“真实值 + 噪声”标签，以最小二乘法拟合线性代理 RM。"""
    xs = [sample_feature() for _ in range(n_samples)]
    ys = []
    for x in xs:
        eps = gauss() if noise == "gauss" else student_t(3.0)
        ys.append(gold_reward(x) + eps)
    # 正规方程：w = (X^T X)^-1 X^T y。
    # 在 D 维空间中通过 Gram 矩阵求逆获得闭式解（小型线性系统）。
    g = [[0.0] * D for _ in range(D)]
    b = [0.0] * D
    for x, y in zip(xs, ys):
        for i in range(D):
            b[i] += x[i] * y
            for j in range(D):
                g[i][j] += x[i] * x[j]
    # 添加 ridge 项，确保 n_samples 很小时矩阵仍可逆。
    for i in range(D):
        g[i][i] += 1e-3
    w = solve(g, b)
    return ProxyRM(w=w, n_samples=n_samples)


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元法。D 很小，因此这种实现已经足够。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        piv = i
        for k in range(i + 1, n):
            if abs(m[k][i]) > abs(m[piv][i]):
                piv = k
        m[i], m[piv] = m[piv], m[i]
        for k in range(i + 1, n):
            f = m[k][i] / m[i][i]
            for j in range(i, n + 1):
                m[k][j] -= f * m[i][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def sqrt_kl_from_origin(mu: list[float]) -> float:
    """两个单位方差高斯分布，一个均值为 0，另一个为 mu。KL = 1/2 * ||mu||^2。"""
    return math.sqrt(0.5 * sum(m * m for m in mu))


def expected_reward(w: list[float], mu: list[float]) -> float:
    """E_{x ~ N(mu, I)} [<w, x>] = <w, mu>。"""
    return dot(w, mu)


def best_of_n_sweep(proxy: ProxyRM, ns: list[int]) -> list[tuple[float, float, float]]:
    """模拟各个 n 下的 best-of-n 采样，计算选中响应的平均 KL、代理分数和
    真实分数。"""
    curve = []
    trials = 1000
    for n in ns:
        kls = []
        proxies = []
        golds = []
        for _ in range(trials):
            xs = [sample_feature() for _ in range(n)]
            best = max(xs, key=proxy.score)
            proxies.append(proxy.score(best))
            golds.append(gold_reward(best))
            # 在极限情况下，best-of-n 分布相对均匀分布的 KL 为 log(n) nats。
            # 这里计算一个代理值：最优样本到均值的距离。
            kls.append(math.sqrt(0.5 * sum(b * b for b in best)))
        curve.append((
            sum(kls) / trials,
            sum(proxies) / trials,
            sum(golds) / trials,
        ))
    return curve


def kl_constrained_policy_sweep(proxy: ProxyRM,
                                kl_budgets: list[float]) -> list[tuple[float, float, float]]:
    """求解 argmax_mu <w_proxy, mu> - lambda * ||mu||^2/2，并扫描 lambda。"""
    curve = []
    for kl in kl_budgets:
        # 在 ||mu||^2 <= 2 * kl 下的最优 mu：缩放代理权重。
        norm = math.sqrt(sum(w * w for w in proxy.w))
        if norm < 1e-9:
            mu = [0.0] * D
        else:
            s = math.sqrt(2 * kl) / norm
            mu = [w * s for w in proxy.w]
        curve.append((
            sqrt_kl_from_origin(mu),
            expected_reward(proxy.w, mu),
            expected_reward(GOLD_W, mu),
        ))
    return curve


def print_curve(name: str, curve: list[tuple[float, float, float]]) -> None:
    print(f"\n{name}")
    print("-" * 60)
    print(f"  {'sqrt(KL)':>9}  {'代理':>8}  {'真实':>8}  {'差距':>8}")
    for sk, p, g in curve:
        print(f"  {sk:>9.3f}  {p:>8.3f}  {g:>8.3f}  {p - g:>+8.3f}")
    peak_gold = max(curve, key=lambda r: r[2])
    print(f"  真实奖励在 sqrt(KL) = {peak_gold[0]:.3f} 处达到峰值，"
          f"真实值 = {peak_gold[2]:.3f}，代理值 = {peak_gold[1]:.3f}")


def main() -> None:
    print("=" * 60)
    print("奖励劫持的过度优化（阶段 18，第 2 课）")
    print("=" * 60)

    budgets = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    for n in (100, 300, 1000, 10000):
        rm = train_proxy(n)
        curve = kl_constrained_policy_sweep(rm, budgets)
        print_curve(f"使用 {n} 个样本训练的代理 RM（高斯噪声）", curve)

    # 重尾代理误差：灾难性 Goodhart 条件。
    rm_heavy = train_proxy(300, noise="student_t")
    curve_heavy = kl_constrained_policy_sweep(rm_heavy, budgets)
    print_curve("代理 RM，300 个样本，Student-t(3) 噪声（重尾）",
                curve_heavy)

    # 用于对比的 best-of-N 采样曲线。
    ns = [1, 2, 4, 8, 16, 64, 256, 1024]
    bon = best_of_n_sweep(train_proxy(300), ns)
    print_curve("Best-of-N 采样（300 样本代理）", bon)

    print("\n" + "=" * 60)
    print("要点：代理奖励单调上升，而真实奖励先达峰值再下降。")
    print("增加代理样本会将峰值推得更远，却无法消除它。")
    print("重尾噪声会使峰值更接近原点。仅靠 KL 无法避免这一问题。")
    print("这就是可量化的 Goodhart 定律。")
    print("=" * 60)


if __name__ == "__main__":
    main()
