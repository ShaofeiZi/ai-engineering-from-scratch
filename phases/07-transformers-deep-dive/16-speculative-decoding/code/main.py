"""推测解码：核心算法与分布等价性。

实现内容：
- 使用 min(1, q/p) 进行伯努利接受/拒绝
- 拒绝后的回退残差分布 (q - p)_+
- 全部接受时附赠一个 token
- 以经验统计检查边际分布是否与直接采样一致
- 扫描接受率与 KL 散度的关系
"""

import math
import random


def sample(probs, rng):
    u = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0.0:
        return list(q)
    return [r / s for r in raw]


def kl(q, p):
    total = 0.0
    for qi, pi in zip(q, p):
        if qi > 0 and pi > 0:
            total += qi * math.log(qi / pi)
    return total


def spec_step_one_token(q, p, rng):
    """从 p 草拟 1 个 token，用 q 验证。返回（接受的 token，是否已接受）。"""
    d = sample(p, rng)
    p_prob = p[d]
    q_prob = q[d]
    u = rng.random()
    if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
        return d, True
    return sample(residual(q, p), rng), False


def spec_step_n(q, p, N, rng):
    """草拟 N 个 token（相同上下文），然后一次性验证。
    返回 (final_token, n_accepted)。简化条件：每次调用的 q 和 p 固定。
    """
    accepted = 0
    for _ in range(N):
        d = sample(p, rng)
        p_prob = p[d]
        q_prob = q[d]
        u = rng.random()
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            accepted += 1
        else:
            return sample(residual(q, p), rng), accepted
    bonus = sample(q, rng)
    return bonus, accepted + 1


def run_distribution_check(q, p, n_samples, rng):
    spec_counts = [0] * len(q)
    direct_counts = [0] * len(q)
    for _ in range(n_samples):
        d, _ = spec_step_one_token(q, p, rng)
        spec_counts[d] += 1
        direct_counts[sample(q, rng)] += 1
    return spec_counts, direct_counts


def chi_square(observed, expected):
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


def acceptance_rate(q, p, n_samples, rng):
    hits = 0
    for _ in range(n_samples):
        _, was = spec_step_one_token(q, p, rng)
        if was:
            hits += 1
    return hits / n_samples


def perturb(q, amount, rng):
    p = [max(1e-6, qi + amount * rng.gauss(0, 1)) for qi in q]
    s = sum(p)
    return [pi / s for pi in p]


def expected_tokens_per_verify(alpha, N):
    if alpha >= 1.0:
        return N + 1
    if alpha == 0:
        return 1
    return (1 - alpha ** (N + 1)) / (1 - alpha)


def main():
    rng = random.Random(7)
    V = 8
    q = [0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]
    p_good = perturb(q, amount=0.02, rng=rng)
    p_bad = perturb(q, amount=0.25, rng=rng)

    print("=== 验证器分布 ===")
    print("  q: " + " ".join(f"{qi:.3f}" for qi in q))
    print()

    print("=== 推测采样与直接采样（分布等价性）===")
    spec_c, direct_c = run_distribution_check(q, p_good, 50000, rng)
    chi = chi_square(spec_c, direct_c)
    print(f"  推测采样计数（50000 个样本）：{spec_c}")
    print(f"  直接采样计数（50000 个样本）：{direct_c}")
    print(f"  chi^2 = {chi:.2f}   （V-1 = {V-1} 自由度；值大表示分布不同）")
    print(f"  {'通过' if chi < 30 else '失败'}：推测解码 token 与验证器分布一致")
    print()

    print("=== 接受率与 KL(q || p) 的关系 ===")
    print(f"  {'KL(q||p)':>10}  {'接受率 α':>14}")
    for noise in (0.005, 0.02, 0.05, 0.10, 0.25, 0.5):
        p = perturb(q, amount=noise, rng=random.Random(noise * 1000))
        alpha = acceptance_rate(q, p, 5000, rng)
        print(f"  {kl(q, p):>10.4f}  {alpha:>14.3f}")
    print()

    print("=== 每次验证器调用的预期 token 数（理论值）===")
    print(f"  {'α':>5} " + "".join(f"  N={N:>2}" for N in (1, 3, 5, 7, 10)))
    for alpha in (0.3, 0.5, 0.7, 0.85, 0.95):
        row = f"  {alpha:>5.2f} " + "".join(
            f"  {expected_tokens_per_verify(alpha, N):>4.2f}" for N in (1, 3, 5, 7, 10)
        )
        print(row)
    print()

    print("要点：Leviathan 定理成立——推测解码分布等于验证器分布。")
    print("      当 α=0.85 且 N=5 时：每次验证器调用约生成 4.1 个 token，")
    print("      大模型前向传播次数约减少 4 倍（未扣除草拟模型开销）。")


if __name__ == "__main__":
    main()
