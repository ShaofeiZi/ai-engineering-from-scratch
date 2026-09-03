"""斑点解码套:准确的拒绝规则,α扫码,树面具.

这文件证明了三件事,在合成玩具的分布 所以数学
保持可见 :

1. 利维坦-卡莱-马蒂亚斯拒绝规则维护目标
抽样分布。 经验的总变量距离
纯目标取样和带有推测的取样草稿为 < 0.01
超过5万张图
2. 预期的-tokens-逐项验证公式保持不变。 接受率
α和草稿长度 K, E [tokens] = (1- alpha^(K+1)) / (1- α)
匹配取样噪音中测量的吞吐量。
3. 树起草验证一个目标中的多个候选路径
通过地形因子掩体前进。 我们造一棵深K树,发射
并确认每个节点只参加其
祖辈们

Stdlib + numpy 仅此而已.

运行 :
python 主页.py
python 主.py --vocab 64 -- alpha 0.75 -- k 4 -- -- 样本 50000
"""

from __future__ import annotations

import argparse
import numpy as np


def make_target(vocab: int, rng: np.random.Generator) -> np.ndarray:
    logits = rng.standard_normal(vocab) * 1.4
    e = np.exp(logits - logits.max())
    return e / e.sum()


def make_draft(target: np.ndarray, alpha_hint: float,
               rng: np.random.Generator) -> np.ndarray:
    """一份预期token级接受度接近的分发草稿
阿尔法 int. 我们线性地将目标与统一的分布混合;
混合比控制着草稿离目标有多近."""
    vocab = target.size
    uniform = np.full(vocab, 1.0 / vocab)
    draft = alpha_hint * target + (1.0 - alpha_hint) * uniform
    noise = rng.uniform(0.95, 1.05, size=vocab)
    draft = draft * noise
    return draft / draft.sum()


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(probs.size, p=probs))


def speculative_step(target: np.ndarray, draft: np.ndarray, K: int,
                     rng: np.random.Generator) -> list[int]:
    """一轮. 返回 1..K+1 tokens,其分布等于目标。"""
    proposed: list[int] = []
    q_at: list[float] = []
    for _ in range(K):
        t = sample(draft, rng)
        proposed.append(t)
        q_at.append(float(draft[t]))

    accepted: list[int] = []
    for k, tok in enumerate(proposed):
        ratio = float(target[tok]) / max(q_at[k], 1e-12)
        if rng.random() < min(1.0, ratio):
            accepted.append(tok)
        else:
            residual = np.maximum(target - draft, 0.0)
            s = residual.sum()
            if s == 0.0:
                accepted.append(sample(target, rng))
            else:
                accepted.append(sample(residual / s, rng))
            return accepted
    accepted.append(sample(target, rng))
    return accepted


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def empirical_dist(samples: list[int], vocab: int) -> np.ndarray:
    counts = np.bincount(samples, minlength=vocab).astype(np.float64)
    return counts / counts.sum()


def verify_distribution(target: np.ndarray, draft: np.ndarray, K: int,
                        n_samples: int, rng: np.random.Generator
                        ) -> tuple[float, float]:
    """在纯目标取样和
投机性取样。 它们必须在统计上无法区分。"""
    vocab = target.size
    plain = [sample(target, rng) for _ in range(n_samples)]
    spec_first: list[int] = []
    while len(spec_first) < n_samples:
        toks = speculative_step(target, draft, K, rng)
        spec_first.append(toks[0])
    p_plain = empirical_dist(plain, vocab)
    p_spec = empirical_dist(spec_first, vocab)
    return total_variation(p_plain, target), total_variation(p_spec, target)


def measure_alpha(target: np.ndarray, draft: np.ndarray,
                  n_samples: int, rng: np.random.Generator) -> float:
    accepted = 0
    for _ in range(n_samples):
        t = sample(draft, rng)
        ratio = float(target[t]) / max(float(draft[t]), 1e-12)
        if rng.random() < min(1.0, ratio):
            accepted += 1
    return accepted / n_samples


def expected_tokens(alpha: float, K: int) -> float:
    if alpha >= 1.0:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


def measure_throughput(target: np.ndarray, draft: np.ndarray, K: int,
                       n_rounds: int, rng: np.random.Generator) -> float:
    total = 0
    for _ in range(n_rounds):
        total += len(speculative_step(target, draft, K, rng))
    return total / n_rounds


def build_tree(branch_factor: tuple[int, ...]) -> list[tuple[int, list[int]]]:
    """返回节点为( parent index, 深度路径) 。 索引 0为根."""
    tree: list[tuple[int, list[int]]] = [(-1, [])]
    frontier = [0]
    for depth, b in enumerate(branch_factor):
        next_frontier: list[int] = []
        for parent in frontier:
            for _ in range(b):
                tree.append((parent, tree[parent][1] + [len(tree)]))
                next_frontier.append(len(tree) - 1)
        frontier = next_frontier
    return tree


def tree_attention_mask(tree: list[tuple[int, list[int]]]) -> np.ndarray:
    """Nx N 因果面罩,每行只关注祖先."""
    n = len(tree)
    mask = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        cur = i
        while cur != -1:
            mask[i, cur] = 1
            cur = tree[cur][0]
    return mask


def validate_tree_mask(mask: np.ndarray,
                       tree: list[tuple[int, list[int]]]) -> bool:
    n = len(tree)
    for i in range(n):
        cur = i
        ancestors = set()
        while cur != -1:
            ancestors.add(cur)
            cur = tree[cur][0]
        attends = {j for j in range(n) if mask[i, j] == 1}
        if attends != ancestors:
            return False
    return True


def _positive_int(value: str, *, minimum: int = 1) -> int:
    n = int(value)
    if n < minimum:
        raise argparse.ArgumentTypeError(f"value must be >= {minimum}, got {n}")
    return n


def _unit_float(value: str) -> float:
    f = float(value)
    if not (0.0 < f <= 1.0):
        raise argparse.ArgumentTypeError(f"value must be in (0, 1], got {f}")
    return f


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=lambda v: _positive_int(v, minimum=2), default=32,
                        help="vocab size (>= 2)")
    parser.add_argument("--alpha", type=_unit_float, default=0.75,
                        help="target acceptance rate in (0, 1]")
    parser.add_argument("--k", type=lambda v: _positive_int(v, minimum=1), default=4,
                        help="draft length (>= 1)")
    parser.add_argument("--samples", type=lambda v: _positive_int(v, minimum=2), default=20000,
                        help="sample count (>= 2)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    target = make_target(args.vocab, rng)
    draft = make_draft(target, args.alpha, rng)

    tv_plain, tv_spec = verify_distribution(
        target, draft, args.k, args.samples, rng
    )
    print(f"分布检查（n={args.samples}）：")
    print(f"  TV(plain_target_sampling, target)       = {tv_plain:.4f}")
    print(f"  TV(speculative_sampling, target)         = {tv_spec:.4f}")
    print(f"  TV 差值（投机采样与直接采样）            = {abs(tv_spec - tv_plain):.4f}")

    alpha_hat = measure_alpha(target, draft, args.samples // 2, rng)
    print()
    print(f"alpha 测量（vocab={args.vocab}，alpha 提示值={args.alpha}）：")
    print(f"  实测 alpha = {alpha_hat:.3f}")

    throughput = measure_throughput(target, draft, args.k, 2000, rng)
    expected = expected_tokens(alpha_hat, args.k)
    print()
    print(f"K={args.k} 时的吞吐量：")
    print(f"  实测 E[token/验证] = {throughput:.3f}")
    print(f"  预测 E[token/验证] = {expected:.3f}  (1 - a^(K+1)) / (1 - a)")

    print()
    print("α扫描, K=4 :")
    for a in (0.3, 0.5, 0.7, 0.85, 0.95):
        print(f"  alpha={a:.2f}  期望 token 数={expected_tokens(a, args.k):.2f}")

    print()
    print("树起草演示:深-3树,树枝=(3, 2, 2)")
    tree = build_tree((3, 2, 2))
    mask = tree_attention_mask(tree)
    print(f"  候选节点总数：{len(tree)}（一次验证前向传播覆盖全部节点）")
    print(f"  mask 形状：{mask.shape}")
    print(f"  mask 与祖先集合的一致性：{validate_tree_mask(mask, tree)}")
    print(f"  各节点可关注数量（按行）：{mask.sum(axis=1).tolist()}")


if __name__ == "__main__":
    main()
