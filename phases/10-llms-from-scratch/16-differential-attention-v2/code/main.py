"""difficial attention (Ye等人,ICLR 2025) in stdlib Python.

从分割的Q, K 中构建两个 softmax 地图, 从第一个中减去第二个
以学习的羊排缩放,乘以V。
在合成长文本查询中产生的 attention权重比率
并比较标准 softmax attention。 还打印参数计数
DIFF V1和DIFF V2在基准 Transformer 上的参数。

纯斯德利布. 无[ph7],无火炬.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


def dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def softmax_row(row: List[float]) -> List[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def standard_attention(Q: List[List[float]], K: List[List[float]],
                       V: List[List[float]]) -> tuple[List[List[float]], List[List[float]]]:
    d = len(Q[0])
    scale = math.sqrt(d)
    weights = []
    for q in Q:
        row = [dot(q, k) / scale for k in K]
        weights.append(softmax_row(row))
    out = []
    d_v = len(V[0])
    for w in weights:
        o = [sum(w[j] * V[j][c] for j in range(len(V))) for c in range(d_v)]
        out.append(o)
    return weights, out


def diff_attention(Q1: List[List[float]], K1: List[List[float]],
                   Q2: List[List[float]], K2: List[List[float]],
                   V: List[List[float]],
                   lam: float) -> tuple[List[List[float]], List[List[float]]]:
    """差别attention :
A1 = softmax(Q1 K1^T / sqrt(d))
A2 = softmax(Q2 K2^T / sqrt(d))
输出=(A1 - lam * A2) V
"""
    d = len(Q1[0])
    scale = math.sqrt(d)
    weights = []
    for q1, q2 in zip(Q1, Q2):
        row1 = softmax_row([dot(q1, k) / scale for k in K1])
        row2 = softmax_row([dot(q2, k) / scale for k in K2])
        diff = [a - lam * b for a, b in zip(row1, row2)]
        weights.append(diff)
    out = []
    d_v = len(V[0])
    for w in weights:
        o = [sum(w[j] * V[j][c] for j in range(len(V))) for c in range(d_v)]
        out.append(o)
    return weights, out


def random_projection(d_in: int, d_out: int,
                      rng: random.Random) -> List[List[float]]:
    """d in x d out 投影矩阵,带有单位变量列。"""
    return [[rng.gauss(0, 1.0 / math.sqrt(d_in)) for _ in range(d_out)]
            for _ in range(d_in)]


def matmul(X: List[List[float]], W: List[List[float]]) -> List[List[float]]:
    out = []
    d_out = len(W[0])
    for row in X:
        o = [sum(row[k] * W[k][c] for k in range(len(row))) for c in range(d_out)]
        out.append(o)
    return out


def build_signal_plus_noise(
    n_tokens: int, signal_pos: int, d_embed: int, noise_scale: float,
    rng: random.Random,
) -> tuple[List[List[float]], List[float]]:
    """返回一个输入 embedding 序列 X [n tokens][d embed] 和一个查询
矢量 q. 位置信号  pos 带有特定模式;查询是
符合这个模式。 所有其他位置都是高斯噪音.

Q,K的预测在ATER这一建设步骤之后应用,以便两者都
差分的分支看到相同的基础序列,但投影
通过不同的矩阵——DIFF attention的忠实模拟.
"""
    pattern = [rng.gauss(0, 1) for _ in range(d_embed)]
    norm = math.sqrt(sum(x * x for x in pattern))
    pattern = [x / norm for x in pattern]
    X = []
    for i in range(n_tokens):
        if i == signal_pos:
            X.append([p + rng.gauss(0, noise_scale * 0.1) for p in pattern])
        else:
            X.append([rng.gauss(0, noise_scale) for _ in range(d_embed)])
    q = list(pattern)
    return X, q


def snr(weights_row: List[float], signal_pos: int) -> float:
    sig = abs(weights_row[signal_pos])
    noise_vals = [abs(w) for i, w in enumerate(weights_row) if i != signal_pos]
    mean_noise = sum(noise_vals) / len(noise_vals)
    if mean_noise == 0:
        return float("inf")
    return sig / mean_noise


@dataclass
class ParamDiff:
    baseline: int
    diff_v1: int
    diff_v2: int
    extra_v1: int
    extra_v2: int


def attention_params_baseline(hidden: int) -> int:
    return 4 * hidden * hidden


def attention_params_diff_v1(hidden: int, n_heads: int, d_head: int) -> int:
    q_params = 2 * (hidden * (n_heads * d_head // 2))
    k_params = 2 * (hidden * (n_heads * d_head // 2))
    v_params = hidden * hidden
    o_params = hidden * hidden
    lam_params = 4 * n_heads * (d_head // 2)
    return q_params + k_params + v_params + o_params + lam_params


def attention_params_diff_v2(hidden: int, n_heads: int, d_head: int,
                             kv_heads: int) -> int:
    q_params = hidden * (2 * n_heads * d_head)
    k_params = hidden * (kv_heads * d_head)
    v_params = hidden * (kv_heads * d_head)
    o_params = (2 * n_heads * d_head) * hidden
    lam_params = 4 * n_heads * d_head
    return q_params + k_params + v_params + o_params + lam_params


def compute_param_diff(hidden: int, n_heads: int, kv_heads: int) -> ParamDiff:
    d_head = hidden // n_heads
    base = attention_params_baseline(hidden)
    v1 = attention_params_diff_v1(hidden, n_heads, d_head)
    v2 = attention_params_diff_v2(hidden, n_heads, d_head, kv_heads)
    return ParamDiff(
        baseline=base,
        diff_v1=v1,
        diff_v2=v2,
        extra_v1=v1 - base,
        extra_v2=v2 - base,
    )


def fmt_m(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return f"{n}"


def main() -> None:
    rng = random.Random(17)
    print("=" * 70)
    print("差分注意力 V2（第 10 阶段，第 16 课）")
    print("=" * 70)
    print()

    n_tokens = 1024
    signal_pos = 500

    print("-" * 70)
    print(f"步骤 1：直接构造 logits 的示例，长度 {n_tokens}，信号位置 {signal_pos}")
    print("-" * 70)
    print("两个分支都在 q·K logits 上计算 softmax。分支 1 是")
    print("能够正确放大信号的已训练 attention head，分支 2 是")
    print("未训练且主要观察噪声的 head。Differential Attention 会减去")
    print("两个分支共享的噪声底。")
    print()

    signal_logit = 4.0
    noise_std = 0.5
    logits_trained = [rng.gauss(0, noise_std) for _ in range(n_tokens)]
    logits_trained[signal_pos] = signal_logit
    logits_untrained = [rng.gauss(0, noise_std) for _ in range(n_tokens)]

    A1 = softmax_row(logits_trained)
    A2 = softmax_row(logits_untrained)

    std_snr = snr(A1, signal_pos)
    std_signal = A1[signal_pos]
    std_noise = sum(abs(w) for i, w in enumerate(A1) if i != signal_pos)
    print("  标准 softmax attention（仅分支 1）：")
    print(f"    信号位置的权重： {std_signal:.6f}")
    print(f"    |噪声| 权重之和：{std_noise:.6f}")
    print(f"    信噪比：          {std_snr:.2f}")
    print()

    for lam in (0.0, 0.3, 0.6, 0.8, 1.0):
        diff = [a1 - lam * a2 for a1, a2 in zip(A1, A2)]
        dsnr = snr(diff, signal_pos)
        d_signal = diff[signal_pos]
        d_noise = sum(abs(w) for i, w in enumerate(diff) if i != signal_pos)
        print(f"  differential attention（lambda={lam:.1f}）：")
        print(f"    信号位置的权重： {d_signal:+.6f}")
        print(f"    |噪声| 权重之和：{d_noise:.6f}")
        print(f"    信噪比：          {dsnr:.2f}")
    print()

    print("-" * 70)
    print("第2步:噪声放大扫描(较高 = 噪声上下文)")
    print("-" * 70)
    print(f"  {'噪声尺度':>10}  {'标准 SNR':>9}  {'Diff SNR（lambda=0.8）':>20}")
    for noise_scale in (0.25, 0.50, 1.0, 1.5, 2.0):
        lrng = random.Random(int(noise_scale * 100))
        l1 = [lrng.gauss(0, noise_scale) for _ in range(n_tokens)]
        l1[signal_pos] = signal_logit
        l2 = [lrng.gauss(0, noise_scale) for _ in range(n_tokens)]
        A1s = softmax_row(l1)
        A2s = softmax_row(l2)
        diff_s = [a - 0.8 * b for a, b in zip(A1s, A2s)]
        s_snr = snr(A1s, signal_pos)
        d_snr = snr(diff_s, signal_pos)
        print(f"  {noise_scale:>10.2f}  {s_snr:>9.2f}  {d_snr:>20.2f}")
    print()

    print("-" * 70)
    print("步骤 3：7B 级配置的参数量差异")
    print("-" * 70)
    pd = compute_param_diff(hidden=4096, n_heads=32, kv_heads=8)
    print(f"  基线 attention：{fmt_m(pd.baseline)}")
    print(f"  DIFF V1 attention    : {fmt_m(pd.diff_v1)}  (delta {fmt_m(pd.extra_v1)})")
    print(f"  DIFF V2 attention    : {fmt_m(pd.diff_v2)}  (delta {fmt_m(pd.extra_v2)})")
    print()

    print("要点：Differential Attention 能稳定提高长上下文查询的信噪比。")
    print("V2 通过将 Q head 数加倍，而不是将 head dim 减半，")
    print("降低了额外参数成本，同时保持接近基线的解码速度。")


if __name__ == "__main__":
    main()
