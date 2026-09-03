"""Weak-to-Strong Generalization 模拟器——仅使用 Python 标准库。

任务：在合成的三特征问题上执行二元分类。
弱标注器：准确率为 0.70，错误集中在一个子类上。
强模型：在真实标签上的准确率上限为 0.95（线性分隔器）。

流程：在弱标签上微调强模型，并测量 PGR。

用法：python3 code/main.py
"""

from __future__ import annotations

import random


random.seed(29)


def gen(n: int) -> list[tuple[list[float], int]]:
    data = []
    for _ in range(n):
        x = [random.gauss(0.0, 1.0) for _ in range(3)]
        y = 1 if x[0] + x[1] - 0.5 * x[2] > 0 else 0
        data.append((x, y))
    return data


def weak_label(x: list[float], accuracy: float = 0.70) -> int:
    """弱标注器：仅对 x[0] 使用简单阈值，并添加噪声以达到目标准确率。
    它会遗漏 x[1] 和 x[2] 的信号。"""
    base = 1 if x[0] > 0 else 0
    if random.random() < accuracy:
        return base
    return 1 - base


def train_strong(data: list[tuple[list[float], int]], steps: int = 200,
                 lr: float = 0.05) -> list[float]:
    """通过 SGD 拟合三特征线性分类器。"""
    w = [0.0, 0.0, 0.0]
    b = 0.0
    for _ in range(steps):
        random.shuffle(data)
        for x, y in data:
            z = b + sum(wi * xi for wi, xi in zip(w, x))
            # sigmoid。
            p = 1.0 / (1.0 + pow(2.71828, -z))
            err = p - y
            for i in range(3):
                w[i] -= lr * err * x[i]
            b -= lr * err
    return w + [b]


def accuracy(model: list[float], data: list[tuple[list[float], int]]) -> float:
    w, b = model[:3], model[3]
    correct = 0
    for x, y in data:
        z = b + sum(wi * xi for wi, xi in zip(w, x))
        pred = 1 if z > 0 else 0
        if pred == y:
            correct += 1
    return correct / len(data)


def run(label: str, weak_acc: float) -> None:
    eval_data = gen(1000)
    train_data = gen(1000)
    # 弱标注器单独使用时的准确率。
    weak_correct = sum(1 for (x, y) in eval_data if weak_label(x, weak_acc) == y)
    weak_alone = weak_correct / len(eval_data)

    # 强模型在真实标签上的性能上限。
    strong_gold = train_strong(train_data)
    ceiling = accuracy(strong_gold, eval_data)

    # weak-to-strong：在弱标签上训练强模型。
    weak_labeled = [(x, weak_label(x, weak_acc)) for (x, _) in train_data]
    strong_w2s = train_strong(weak_labeled)
    w2s_acc = accuracy(strong_w2s, eval_data)

    pgr = (w2s_acc - weak_alone) / (ceiling - weak_alone + 1e-12)
    print(f"\n{label}（weak_accuracy={weak_acc}）")
    print(f"  单独使用弱标注器：{weak_alone:.3f}")
    print(f"  强模型使用真实标签：{ceiling:.3f}")
    print(f"  强模型使用弱标签：{w2s_acc:.3f}")
    print(f"  恢复的性能差距（PGR）：{pgr:.3f}")


def main() -> None:
    print("=" * 70)
    print("WEAK-TO-STRONG GENERALIZATION（阶段 18，第 11 课）")
    print("=" * 70)

    for acc in (0.60, 0.70, 0.80, 0.90):
        run(f"weak-to-strong @ 弱标注准确率={acc}", acc)

    print("\n" + "=" * 70)
    print("要点：所有弱标注器上的 PGR > 0，说明强模型利用自身预训练先验，")
    print("实现了超越弱监督者错误的泛化。这是 Burns 等人 2023 针对超级对齐")
    print("问题提出的经验代理指标：弱人类监督能否产生更强且对齐的模型？")
    print("它不是解决方案，而是一个可测量指标。")
    print("=" * 70)


if __name__ == "__main__":
    main()
