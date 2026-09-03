"""标本解码(Leviathan 2023),带有N-token草稿和KV回滚.

执行完整的生产投机-解码循环:
- p(廉价)的Ntokens草案
- 以一个平行q前方核查N位置
- 拒绝规则:以分钟(1、q(d)/p(d)接受)
- 拒绝时的残余采样：(q-p)
完全接受时的奖金token
- KV cache 回滚记账

仅限斯德利布. 数字与第 7 阶段相匹配 16 个数学证明和什么
第10阶段 -- -- 12个作业阶段。 我们一起缝合
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


def sample(probs: List[float], rng: random.Random) -> int:
    u = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def residual(q: List[float], p: List[float]) -> List[float]:
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0.0:
        return list(q)
    return [r / s for r in raw]


def kl(q: List[float], p: List[float]) -> float:
    total = 0.0
    for qi, pi in zip(q, p):
        if qi > 0 and pi > 0:
            total += qi * math.log(qi / pi)
    return total


@dataclass
class KVBuffer:
    """跟踪验证器的逻辑缓存长度 。 物理字节是名义的."""
    length: int = 0

    def extend(self, n: int) -> None:
        self.length += n

    def truncate_to(self, n: int) -> None:
        self.length = n


def spec_step(q: List[float], p: List[float], N: int, kv: KVBuffer,
              rng: random.Random) -> tuple[List[int], int]:
    """一个推测步骤:p中的N tokens草案,与q校验.

返回(托存 发送、核查 前置 使用)。 验证器( F)
这里总是有一点——这就是重点。 指示符  发送符为 1 到 N+ 1 之间 。

就教学简便而言,q和p是共享的无上下文分布
横跨阵地。 数学延伸至位置依赖 q i, p i 无
改变循环。
"""
    prefix_len = kv.length
    drafts: List[int] = []
    p_probs: List[float] = []
    for _ in range(N):
        d = sample(p, rng)
        drafts.append(d)
        p_probs.append(p[d])

    emitted: List[int] = []
    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q[d]
        p_prob = p_probs[i]
        ratio = q_prob / p_prob if p_prob > 0 else float("inf")
        if u < min(1.0, ratio):
            emitted.append(d)
            kv.extend(1)
        else:
            correction = sample(residual(q, p), rng)
            emitted.append(correction)
            kv.truncate_to(prefix_len + len(emitted))
            return emitted, 1

    bonus = sample(q, rng)
    emitted.append(bonus)
    kv.extend(1)
    return emitted, 1


def direct_sample(q: List[float], n: int, rng: random.Random) -> List[int]:
    return [sample(q, rng) for _ in range(n)]


def distribution_check(q: List[float], p: List[float], n_steps: int,
                       rng: random.Random) -> tuple[List[int], List[int]]:
    """请检查date=中的日期值 (帮助) First exploaded token (Leviathan-sampled one)
作为 q 分发。 接受草案;拒绝草案;
剩余更正。 完全接受之后的奖金token是
也以 q 形式分发,但为第二张图,不应混入
在这里。"""
    spec_counts = [0] * len(q)
    direct_counts = [0] * len(q)
    for _ in range(n_steps):
        kv = KVBuffer()
        tokens, _ = spec_step(q, p, N=1, kv=kv, rng=rng)
        spec_counts[tokens[0]] += 1
        direct_counts[sample(q, rng)] += 1
    return spec_counts, direct_counts


def chi_square(observed: List[int], expected: List[int]) -> float:
    total_obs = sum(observed)
    total_exp = sum(expected)
    if total_obs == 0 or total_exp == 0:
        return 0.0
    result = 0.0
    for o, e in zip(observed, expected):
        e_norm = e * total_obs / total_exp
        if e_norm > 0:
            result += (o - e_norm) ** 2 / e_norm
    return result


def measure_alpha(q: List[float], p: List[float], n_samples: int,
                  rng: random.Random) -> float:
    hits = 0
    for _ in range(n_samples):
        d = sample(p, rng)
        u = rng.random()
        q_prob = q[d]
        p_prob = p[d]
        if p_prob > 0 and u < min(1.0, q_prob / p_prob):
            hits += 1
    return hits / n_samples


def expected_tokens_per_verify(alpha: float, N: int) -> float:
    if alpha >= 1.0:
        return N + 1
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (N + 1)) / (1.0 - alpha)


def wall_time_per_token(alpha: float, N: int, c: float) -> float:
    """草案费用为每token个c,相对于核查者(费用1.0)。

草案每个核查员的呼叫费用为1.0加N * c。 预期 tokens
排放为(1-α^(N+1))/(1-α).
"""
    return (1.0 + N * c) / expected_tokens_per_verify(alpha, N)


def perturb(q: List[float], amount: float, rng: random.Random) -> List[float]:
    p = [max(1e-6, qi + amount * rng.gauss(0, 1)) for qi in q]
    s = sum(p)
    return [pi / s for pi in p]


def main() -> None:
    rng = random.Random(42)

    q = [0.30, 0.22, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03]
    p_eagle3 = perturb(q, amount=0.005, rng=random.Random(1))
    p_eagle1 = perturb(q, amount=0.02, rng=random.Random(2))
    p_vanilla = perturb(q, amount=0.08, rng=random.Random(3))

    print("=" * 70)
    print("投机解码与 EAGLE-3（第 10 阶段，第 15 课）")
    print("=" * 70)
    print()
    print("验证模型分布 q：" + " ".join(f"{qi:.3f}" for qi in q))
    print()

    print("-" * 70)
    print("步骤 1：Leviathan 分布等价性检查（N=1，50000 次试验）")
    print("-" * 70)
    spec_c, direct_c = distribution_check(q, p_eagle1, 50000, rng)
    chi = chi_square(spec_c, direct_c)
    print(f"  投机采样计数：{spec_c}")
    print(f"  直接采样计数：{direct_c}")
    print(f"  chi^2 = {chi:.2f}  (df={len(q) - 1}; 95% crit ~14.07)")
    verdict = "PASS" if chi < 14.07 else "CHECK"
    print(f"  结论：{verdict}（投机解码分布与验证模型一致）")
    print()

    print("-" * 70)
    print("步骤 2：不同 draft 质量对应的接受率 α")
    print("-" * 70)
    print(f"  {'草案':<12} {'KL(q||p)':>10} {'α':>8}")
    for name, p in [("vanilla", p_vanilla), ("eagle-1", p_eagle1),
                    ("eagle-3", p_eagle3)]:
        a = measure_alpha(q, p, 20000, random.Random(7))
        print(f"  {name:<12} {kl(q, p):>10.4f} {a:>8.3f}")
    print()

    print("-" * 70)
    print("步骤 3：每次验证调用的期望 token 数（理论值）")
    print("-" * 70)
    Ns = [1, 3, 5, 7, 10]
    alphas = [0.55, 0.70, 0.80, 0.90, 0.95]
    print(f"  {'α':>6}  " + "".join(f"{f'N={N}':>8}" for N in Ns))
    for a in alphas:
        row = f"  {a:>6.2f}  " + "".join(
            f"{expected_tokens_per_verify(a, N):>8.2f}" for N in Ns
        )
        print(row)
    print()

    print("-" * 70)
    print("步骤 4：c=0.04 时每个 token 的墙钟时间（EAGLE-3 级 draft 成本）")
    print("-" * 70)
    print(f"  {'α':>6}  " + "".join(f"{f'N={N}':>8}" for N in Ns))
    for a in alphas:
        row = f"  {a:>6.2f}  " + "".join(
            f"{wall_time_per_token(a, N, c=0.04):>8.3f}" for N in Ns
        )
        print(row)
    print("（数值越低越快；无投机解码的基线为每个 token 1.000）")
    print()

    print("-" * 70)
    print("步骤 5：端到端模拟（N=5，draft=eagle-3，1000 轮）")
    print("-" * 70)
    kv = KVBuffer()
    total_tokens = 0
    total_forwards = 0
    accepted_per_round: List[int] = []
    for _ in range(1000):
        tokens, forwards = spec_step(q, p_eagle3, N=5, kv=kv, rng=rng)
        total_tokens += len(tokens)
        total_forwards += forwards
        accepted_per_round.append(len(tokens))
    mean_tokens = total_tokens / 1000
    print(f"  输出 token 总数：{total_tokens}")
    print(f"  验证模型前向传播次数：{total_forwards}")
    print(f"  每次前向传播的平均 token 数：{mean_tokens:.2f}")
    print(f"  KV 逻辑长度：{kv.length}（跟踪已接受的前缀）")
    print(f"  alpha=0.95、N=5 时的期望值："
          f"{expected_tokens_per_verify(0.95, 5):.2f}")
    print()

    print("要点：N=5 且 EAGLE-3 类 draft 质量达到 α≈0.9 时，")
    print("每次验证模型前向传播可产生约 4–5 个 token。")
    print("EAGLE-3 论文报告的 3.6–5 倍加速还包含树搜索和 TTT 收益。")


if __name__ == "__main__":
    main()
