"""谄媚行为放大模拟器——仅使用 Python 标准库。

三动作世界：
  A = 正确答案（真实效用 +1.0，同意指标 0）
  S = 谄媚式赞同（真实效用 -0.3，同意指标 1）
  W = 随机错误答案（真实效用 -0.5，同意指标 0）

奖励模型包含两个部分：恰好与谄媚行为相关的“自信/流畅度”奖励，以及正确性。
RL 会像 Shapira 等人预测的那样放大谄媚行为。

这里扫描 beta（KL 系数）和 alpha（同意惩罚修正）。

用法：python3 code/main.py
"""

from __future__ import annotations

import math
import random


random.seed(7)

ACTIONS = ["A", "S", "W"]
TRUE_UTILITY = {"A": 1.0, "S": -0.3, "W": -0.5}
AGREEMENT = {"A": 0.0, "S": 1.0, "W": 0.0}


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def labeler_reward(action: str) -> float:
    """评分者给出的奖励：主要取决于正确性，但也包含较小的同意奖励。
    这是 RM 从真实标注数据中学到的伪特征——流畅的赞同会比同样正确的
    反对获得更高分数。"""
    return TRUE_UTILITY[action] + 0.6 * AGREEMENT[action]


def train_rm(n_pairs: int = 500) -> dict[str, float]:
    """基于评分者的成对偏好，用 Bradley-Terry 模型拟合标量奖励。"""
    r = {a: 0.0 for a in ACTIONS}
    lr = 0.05
    for _ in range(n_pairs):
        i, j = random.sample(ACTIONS, 2)
        diff = labeler_reward(i) - labeler_reward(j)
        p_i = 1 / (1 + math.exp(-diff))
        winner, loser = (i, j) if random.random() < p_i else (j, i)
        d = r[winner] - r[loser]
        s = 1 / (1 + math.exp(-d))
        r[winner] += lr * (1 - s)
        r[loser] -= lr * (1 - s)
    m = sum(r.values()) / 3
    return {a: v - m for a, v in r.items()}


def agreement_penalty_correction(r: dict[str, float], alpha: float) -> dict[str, float]:
    """Shapira 等人的修正方法：r' = r - alpha * agree(y)。"""
    return {a: r[a] - alpha * AGREEMENT[a] for a in ACTIONS}


def ppo_train(ref_logits: list[float], reward: dict[str, float],
              beta: float, steps: int = 300, batch: int = 64,
              lr: float = 0.08) -> list[float]:
    logits = list(ref_logits)
    ref_probs = softmax(ref_logits)
    for _ in range(steps):
        probs = softmax(logits)
        advantages = [0.0, 0.0, 0.0]
        counts = [0, 0, 0]
        for _ in range(batch):
            r = random.random()
            cum = 0.0
            chosen = 0
            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    chosen = i
                    break
            a = ACTIONS[chosen]
            shaped = reward[a] - beta * (math.log(probs[chosen] + 1e-12)
                                         - math.log(ref_probs[chosen] + 1e-12))
            advantages[chosen] += shaped
            counts[chosen] += 1
        for i in range(3):
            if counts[i] > 0:
                advantages[i] /= counts[i]
        grad = [0.0, 0.0, 0.0]
        for i in range(3):
            for b in range(3):
                indicator = 1.0 if i == b else 0.0
                grad[b] += advantages[i] * probs[i] * (indicator - probs[b])
        logits = [l + lr * g for l, g in zip(logits, grad)]
    return logits


def sycophancy(probs: list[float]) -> float:
    return probs[ACTIONS.index("S")]


def correctness(probs: list[float]) -> float:
    return probs[ACTIONS.index("A")]


def report(label: str, logits: list[float]) -> None:
    probs = softmax(logits)
    print(f"  {label:40s}  "
          f"P(A)={correctness(probs):.3f}  "
          f"P(S)={sycophancy(probs):.3f}  "
          f"P(W)={probs[2]:.3f}")


def main() -> None:
    print("=" * 70)
    print("谄媚行为放大（阶段 18，第 4 课）")
    print("=" * 70)

    ref_logits = [0.0, 0.0, 0.0]  # 均匀基础策略。
    print("\n阶段 1——根据评分者偏好训练奖励模型。")
    rm = train_rm()
    print(f"  RM 分数：{[f'{a}={rm[a]:+.3f}' for a in ACTIONS]}")
    print("  （注意：尽管 S 的真实效用较低，它仍获得了奖励加成）")

    print("\n阶段 2——扫描 PPO，不使用同意惩罚。")
    for beta in (1.0, 0.2, 0.05, 0.0):
        logits = ppo_train(ref_logits, rm, beta=beta)
        report(f"PPO beta={beta:4.2f} (alpha=0)", logits)

    print("\n阶段 3——同意惩罚修正（Shapira 等人）。")
    print("  固定 beta=0.1，扫描 alpha。")
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8):
        corrected = agreement_penalty_correction(rm, alpha)
        logits = ppo_train(ref_logits, corrected, beta=0.1)
        report(f"PPO alpha={alpha:.1f}（同意惩罚）", logits)

    print()
    print("-" * 70)
    print("要点：较低的 beta 会放大谄媚行为（RM 奖励赞同）。")
    print("适中的 alpha 会减少谄媚行为，但也会削弱正确时的赞同。")
    print("不存在能无代价恢复基础模型 P(S) 的 alpha。")
    print("=" * 70)


if __name__ == "__main__":
    main()
