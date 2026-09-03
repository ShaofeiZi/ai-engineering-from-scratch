"""玩具偏好数据上的 DPO 系列损失——仅使用 Python 标准库。

使用六种损失在成对偏好数据集上拟合四动作 softmax 策略：DPO、IPO、KTO、
SimPO、ORPO、BPO。比较最终的 chosen log-prob、rejected log-prob、隐式
奖励差距和胜率。

这是玩具级实现，目标是并排理解各种损失公式，而非复现生产环境数值。

用法：python3 code/main.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


random.seed(1)

N_ACTIONS = 4
TRUE_UTILITY = [0.2, 1.0, -0.4, -0.8]


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


def logsoftmax(logits: list[float]) -> list[float]:
    m = max(logits)
    z = math.log(sum(math.exp(x - m) for x in logits)) + m
    return [x - z for x in logits]


def sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def sample_pref_pair() -> tuple[int, int, float]:
    """采样偏好对 (y_w, y_l)，真实偏好强度为 p_w。"""
    i, j = random.sample(range(N_ACTIONS), 2)
    p_i_beats_j = sigmoid(TRUE_UTILITY[i] - TRUE_UTILITY[j])
    if random.random() < p_i_beats_j:
        return i, j, p_i_beats_j
    return j, i, 1 - p_i_beats_j


@dataclass
class Policy:
    logits: list[float]

    def logprob(self, a: int) -> float:
        return logsoftmax(self.logits)[a]

    def grad_logprob(self, a: int) -> list[float]:
        probs = softmax(self.logits)
        return [(1.0 if b == a else 0.0) - probs[b] for b in range(N_ACTIONS)]


def apply_grad(p: Policy, grad: list[float], lr: float) -> None:
    p.logits = [l - lr * g for l, g in zip(p.logits, grad)]


def make_policy_and_ref() -> tuple[Policy, Policy]:
    ref_logits = [0.1, 0.2, -0.1, -0.2]
    return Policy(list(ref_logits)), Policy(list(ref_logits))


def train_dpo(pairs: list[tuple[int, int, float]], beta: float = 0.1,
              steps: int = 2000, lr: float = 0.05,
              variant: str = "dpo") -> Policy:
    pi, ref = make_policy_and_ref()
    for _ in range(steps):
        w, l, strength = random.choice(pairs)
        log_pi_w = pi.logprob(w)
        log_pi_l = pi.logprob(l)
        log_ref_w = ref.logprob(w)
        log_ref_l = ref.logprob(l)
        margin = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
        gw = pi.grad_logprob(w)
        gl = pi.grad_logprob(l)
        if variant == "dpo":
            # L = -log sigmoid(margin)。dL/dmargin = -(1 - sigmoid(margin))。
            g_margin = -(1.0 - sigmoid(margin))
            grad = [beta * (g_margin * gw_i - g_margin * gl_i)
                    for gw_i, gl_i in zip(gw, gl)]
        elif variant == "ipo":
            target = 1.0 / (2 * beta)
            diff = (log_pi_w - log_ref_w) - (log_pi_l - log_ref_l) - target
            g_margin = 2 * diff
            grad = [g_margin * (gw_i - gl_i) for gw_i, gl_i in zip(gw, gl)]
        elif variant == "bpo":
            # DPO + 对 log_pi_w 下降的惩罚。
            g_margin = -(1.0 - sigmoid(margin))
            anchor_pen = -1.0 * (log_pi_w - log_ref_w)  # 将 chosen 推向参考值或更高。
            grad = [beta * (g_margin * gw_i - g_margin * gl_i) - 0.05 * anchor_pen * gw_i
                    for gw_i, gl_i in zip(gw, gl)]
        else:
            raise ValueError(variant)
        apply_grad(pi, grad, lr)
    return pi


def train_simpo(pairs: list[tuple[int, int, float]], beta: float = 1.5,
                gamma: float = 0.5, steps: int = 2000, lr: float = 0.05) -> Policy:
    pi, _ = make_policy_and_ref()
    lens = [1, 1, 1, 1]  # 在单动作玩具示例中很简单，仅作演示。
    for _ in range(steps):
        w, l, _ = random.choice(pairs)
        log_pi_w = pi.logprob(w) / lens[w]
        log_pi_l = pi.logprob(l) / lens[l]
        margin = beta * (log_pi_w - log_pi_l) - gamma
        gw = pi.grad_logprob(w)
        gl = pi.grad_logprob(l)
        g_margin = -(1.0 - sigmoid(margin))
        grad = [beta * (g_margin * gw_i / lens[w] - g_margin * gl_i / lens[l])
                for gw_i, gl_i in zip(gw, gl)]
        apply_grad(pi, grad, lr)
    return pi


def train_kto(labels: list[tuple[int, bool]], beta: float = 0.1,
              steps: int = 2000, lr: float = 0.05) -> Policy:
    pi, ref = make_policy_and_ref()
    z_ref = 0.0
    for _ in range(steps):
        y, desirable = random.choice(labels)
        log_pi_y = pi.logprob(y)
        log_ref_y = ref.logprob(y)
        value = beta * (log_pi_y - log_ref_y) - z_ref
        if desirable:
            v = sigmoid(value)  # 希望提高。
            g_value = -(1 - v)
        else:
            v = sigmoid(-value)
            g_value = (1 - v) * 2.0  # 损失厌恶权重。
        gy = pi.grad_logprob(y)
        grad = [beta * g_value * gy_i for gy_i in gy]
        apply_grad(pi, grad, lr)
    return pi


def train_orpo(pairs: list[tuple[int, int, float]], lam: float = 0.1,
               steps: int = 2000, lr: float = 0.05) -> Policy:
    pi, _ = make_policy_and_ref()
    for _ in range(steps):
        w, l, _ = random.choice(pairs)
        log_pi_w = pi.logprob(w)
        log_pi_l = pi.logprob(l)
        # NLL 项。
        gw = pi.grad_logprob(w)
        # odds ratio 项（简化版）。
        odds_w = math.exp(log_pi_w) / (1 - math.exp(log_pi_w) + 1e-6)
        odds_l = math.exp(log_pi_l) / (1 - math.exp(log_pi_l) + 1e-6)
        log_ratio = math.log(odds_w + 1e-6) - math.log(odds_l + 1e-6)
        g_or = -(1 - sigmoid(log_ratio))
        gl = pi.grad_logprob(l)
        grad = [-gw_i + lam * g_or * (gw_i - gl_i)
                for gw_i, gl_i in zip(gw, gl)]
        apply_grad(pi, grad, lr)
    return pi


def win_rate(pi: Policy) -> float:
    probs = softmax(pi.logits)
    true_probs = softmax(TRUE_UTILITY)
    ranked = sorted(range(N_ACTIONS), key=lambda a: -true_probs[a])
    best = ranked[0]
    return probs[best]


def report(name: str, pi: Policy) -> None:
    print(f"  {name:8s}  概率={[f'{p:.3f}' for p in softmax(pi.logits)]}  "
          f"胜率={win_rate(pi):.3f}  logits={[f'{l:+.2f}' for l in pi.logits]}")


def main() -> None:
    print("=" * 70)
    print("玩具四动作偏好数据上的 DPO 系列方法（阶段 18，第 3 课）")
    print("=" * 70)
    print(f"  真实效用：{TRUE_UTILITY}")
    print(f"  真实最优分布：{[f'{p:.3f}' for p in softmax(TRUE_UTILITY)]}")
    print()

    pairs = [sample_pref_pair() for _ in range(500)]
    labels = []
    for _ in range(500):
        a = random.randrange(N_ACTIONS)
        desirable = random.random() < sigmoid(TRUE_UTILITY[a])
        labels.append((a, desirable))

    ref, _ = make_policy_and_ref()
    report("REF", ref)

    pi_dpo = train_dpo(pairs, variant="dpo")
    report("DPO", pi_dpo)

    pi_ipo = train_dpo(pairs, variant="ipo")
    report("IPO", pi_ipo)

    pi_bpo = train_dpo(pairs, variant="bpo")
    report("BPO", pi_bpo)

    pi_simpo = train_simpo(pairs)
    report("SimPO", pi_simpo)

    pi_kto = train_kto(labels)
    report("KTO", pi_kto)

    pi_orpo = train_orpo(pairs)
    report("ORPO", pi_orpo)

    print()
    print("-" * 70)
    print("要点：六种方法都会将概率质量移向动作 1（真实效用最高）。")
    print("它们的差异在于对参考策略的锚定程度、处理偏好强度的方式，")
    print("以及是否需要成对数据。")
    print("=" * 70)


if __name__ == "__main__":
    main()
