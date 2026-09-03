"""玩具分类器上的三种群体公平性准则——仅使用 Python 标准库。

二元分类：敏感属性 A 属于 {0, 1}，且两组基础率不同。
训练一个简单逻辑分类器，并报告：demographic parity、equalized odds 和
conditional use accuracy equality。然后应用面向 demographic parity 的
重新加权，并观察它给另外两项准则带来的代价。

用法：python3 code/main.py
"""

from __future__ import annotations

import math
import random


random.seed(53)


def gen(n: int) -> list[tuple[list[float], int, int]]:
    """返回 (features, label, sensitive_attribute) 列表。

    各组基础率不同：A=0 时 P(y=1)=0.3；A=1 时 P(y=1)=0.6。
    特征与 y 相关，并含有一定噪声。"""
    data = []
    for _ in range(n):
        a = random.choice([0, 1])
        base = 0.3 if a == 0 else 0.6
        y = 1 if random.random() < base else 0
        x0 = random.gauss(0.8 * y, 1.0)
        x1 = random.gauss(-0.3 + a * 0.5, 1.0)
        data.append(([x0, x1, float(a)], y, a))
    return data


def train(data, steps: int = 200, lr: float = 0.1, sample_weights=None) -> list[float]:
    w = [0.0, 0.0, 0.0]
    b = 0.0
    if sample_weights is None:
        paired = [(ex, 1.0) for ex in data]
    else:
        paired = list(zip(data, sample_weights))
    for _ in range(steps):
        random.shuffle(paired)
        for (x, y, a), wt in paired:
            z = b + sum(wi * xi for wi, xi in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            for i in range(3):
                w[i] -= lr * wt * err * x[i]
            b -= lr * wt * err
    return w + [b]


def predict(model, data):
    w, b = model[:3], model[3]
    preds = []
    for x, y, a in data:
        z = b + sum(wi * xi for wi, xi in zip(w, x))
        preds.append((1 if z > 0 else 0, y, a))
    return preds


def demographic_parity(preds) -> tuple[float, float]:
    rate0 = sum(1 for p, _, a in preds if a == 0 and p == 1) / max(1, sum(1 for _, _, a in preds if a == 0))
    rate1 = sum(1 for p, _, a in preds if a == 1 and p == 1) / max(1, sum(1 for _, _, a in preds if a == 1))
    return rate0, rate1


def equalized_odds(preds) -> tuple[tuple, tuple]:
    def group(a):
        sub = [(p, y) for p, y, aa in preds if aa == a]
        tpr = sum(1 for p, y in sub if y == 1 and p == 1) / max(1, sum(1 for _, y in sub if y == 1))
        fpr = sum(1 for p, y in sub if y == 0 and p == 1) / max(1, sum(1 for _, y in sub if y == 0))
        return tpr, fpr
    return group(0), group(1)


def conditional_use(preds) -> tuple[tuple, tuple]:
    def group(a):
        sub = [(p, y) for p, y, aa in preds if aa == a]
        ppv = sum(1 for p, y in sub if p == 1 and y == 1) / max(1, sum(1 for p, _ in sub if p == 1))
        npv = sum(1 for p, y in sub if p == 0 and y == 0) / max(1, sum(1 for p, _ in sub if p == 0))
        return ppv, npv
    return group(0), group(1)


def report(name: str, preds):
    dp = demographic_parity(preds)
    eo = equalized_odds(preds)
    cu = conditional_use(preds)
    print(f"\n{name}")
    print(f"  demographic parity：组0={dp[0]:.3f}  组1={dp[1]:.3f}  差距={dp[1]-dp[0]:+.3f}")
    print(f"  equalized odds (TPR)：组0={eo[0][0]:.3f}  组1={eo[1][0]:.3f}")
    print(f"  equalized odds (FPR)：组0={eo[0][1]:.3f}  组1={eo[1][1]:.3f}")
    print(f"  conditional use (PPV)：组0={cu[0][0]:.3f}  组1={cu[1][0]:.3f}")
    print(f"  conditional use (NPV)：组0={cu[0][1]:.3f}  组1={cu[1][1]:.3f}")


def main() -> None:
    print("=" * 70)
    print("三种群体公平性准则（阶段 18，第 21 课）")
    print("=" * 70)

    train_data = gen(1000)
    test_data = gen(500)

    baseline = train(train_data)
    preds = predict(baseline, test_data)
    report("基线分类器", preds)

    # 面向 demographic parity 重新加权：提高 group0 y=1，降低 group1 y=1。
    weights = []
    for x, y, a in train_data:
        if a == 0 and y == 1:
            weights.append(2.0)
        elif a == 1 and y == 1:
            weights.append(0.5)
        else:
            weights.append(1.0)
    dp_reweighted = train(train_data, sample_weights=weights)
    preds2 = predict(dp_reweighted, test_data)
    report("经 DP 重新加权的分类器", preds2)

    print("\n" + "=" * 70)
    print("要点：基础率相等是三种准则同时成立的条件。当基础率不同时，面向 DP 的")
    print("重新加权会缩小 DP 差距，却以 equalized odds 和 conditional use")
    print("accuracy 为代价。这是 Chouldechova / KMR 2017 的缩影。准则选择")
    print("属于政策决策；基础率不等时，没有任何统计方法能同时满足三种准则。")
    print("=" * 70)


if __name__ == "__main__":
    main()
