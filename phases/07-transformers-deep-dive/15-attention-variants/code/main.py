"""注意力变体：完整注意力、滑动窗口、局部+步进稀疏注意力、差分注意力。

仅使用标准库。在真实的长上下文预算下，对比各变体的分数掩码结构
及其 KV cache 大小。
"""

import math


NEG_INF = float("-inf")


def causal_mask(n):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            M[i][j] = 0.0
    return M


def swa_mask(n, window):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M


def strided_mask(n, window, stride):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M


def count_nonmasked(M):
    return sum(1 for row in M for v in row if v == 0.0)


def render(M, label):
    n = len(M)
    print(f"{label}  （关注 {count_nonmasked(M)} / {n*n} 个单元）")
    for i in range(n):
        cells = "".join("x" if M[i][j] == 0.0 else "." for j in range(n))
        print(f"  {i:>2} | {cells}")
    print()


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def attention_row(q, Ks, Vs, mask_row):
    d = len(q)
    scores = []
    for k, m in zip(Ks, mask_row):
        if m == NEG_INF:
            scores.append(NEG_INF)
        else:
            s = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)
            scores.append(s)
    finite = [s for s in scores if s != NEG_INF]
    if not finite:
        return [0.0] * len(Vs[0]), [0.0] * len(scores)
    shifted = softmax(finite)
    weights = []
    k = 0
    for s in scores:
        if s == NEG_INF:
            weights.append(0.0)
        else:
            weights.append(shifted[k])
            k += 1
    d_v = len(Vs[0])
    out = [0.0] * d_v
    for w, v in zip(weights, Vs):
        for j in range(d_v):
            out[j] += w * v[j]
    return out, weights


def diff_attention_row(q1, q2, K1, K2, V, mask_row, lam):
    _, w1 = attention_row(q1, K1, V, mask_row)
    _, w2 = attention_row(q2, K2, V, mask_row)
    diff = [a - lam * b for a, b in zip(w1, w2)]
    d_v = len(V[0])
    out = [0.0] * d_v
    for w, v in zip(diff, V):
        for j in range(d_v):
            out[j] += w * v[j]
    return out, diff


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * dtype_bytes


def main():
    print("=== 8-token 序列上的注意力掩码形状 ===")
    print()
    render(causal_mask(8), "完整因果注意力")
    render(swa_mask(8, window=4), "滑动窗口（W=4）")
    render(strided_mask(8, window=2, stride=3), "局部（W=2）+ 跨步（stride=3）")

    print("=== 注意力汇聚：8 个随机 token 上的一个“噪声”查询 ===")
    import random
    rng = random.Random(0)
    d = 8
    K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(8)]
    V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(8)]
    q = [rng.gauss(0, 1) for _ in range(d)]
    mask = causal_mask(8)[7]
    _, w_single = attention_row(q, K, V, mask)
    print(f"单路注意力权重：" + " ".join(f"{w:.3f}" for w in w_single))
    print(f"  （注意权重泄漏到位置 0——即注意力汇聚）")

    q1 = q[:]
    q2 = [x + 0.2 * rng.gauss(0, 1) for x in q]
    K2 = [[x + 0.2 * rng.gauss(0, 1) for x in row] for row in K]
    _, w_diff = diff_attention_row(q1, q2, K, K2, V, mask, lam=0.5)
    print(f"差分注意力权重：" + " ".join(f"{w:+.3f}" for w in w_diff))
    print(f"  （lambda=0.5 会减去汇聚分量；允许负权重）")
    print()

    print("=== 128K 上下文下的 KV cache，类 Llama-3-70B（80 层、8 个 KV 头、d_head=128、fp16）===")
    n_layers, n_kv_heads, d_head = 80, 8, 128
    N = 131072
    full = kv_cache_bytes(n_layers, n_kv_heads, d_head, N)

    print(f"  完整注意力                  ：{full / 1e9:>6.1f} GB")
    for window in (4096, 1024):
        reduced = full * (window / N)
        print(f"  SWA 窗口={window:>5}               : {reduced / 1e9:>6.1f} GB   （缩小 {N/window:.0f} 倍）")

    gemma3_ratio = 1 / 6
    gemma_total = full * (5 / 6) * (1024 / N) + full * (1 / 6)
    print(f"  Gemma-3 混合（5:1，W=1024）：{gemma_total / 1e9:>6.1f} GB   （缩小 {full/gemma_total:.1f} 倍）")

    diff = full * 2
    print(f"  差分注意力（2x）             : {diff / 1e9:>6.1f} GB   （以 2 倍成本换取无汇聚权重）")
    print()
    print("要点：SWA 是成本最低的长上下文优化。")
    print("      Gemma 3 的 5:1 混合保留足够的全局层用于检索，")
    print("      同时将 KV 缩小至纯完整注意力的约 1/6。")
    print("      DIFF 注意力以 2 倍 KV 为代价，换取无汇聚且更精准的检索。")


if __name__ == "__main__":
    main()
