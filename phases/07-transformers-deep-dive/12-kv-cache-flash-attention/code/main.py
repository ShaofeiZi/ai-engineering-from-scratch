"""仅使用标准库实现 KV cache + 分块（Flash 风格）注意力。

展示内容：
- 朴素 O(N^2) 增量解码器与使用 KV cache 的 O(N) 解码器对比
- 逐分块产生逐位相同输出的运行最大值 softmax
- 面向真实 2026 年模型的 KV cache 大小计算
"""

import math
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def attention_full(q, Ks, Vs):
    """针对完整键值列表计算单查询注意力。"""
    scores = [dot(q, k) / math.sqrt(len(q)) for k in Ks]
    weights = softmax(scores)
    out = [0.0] * len(Vs[0])
    for w, v in zip(weights, Vs):
        for j in range(len(out)):
            out[j] += w * v[j]
    return out


def tiled_softmax_dot(q, Ks, Vs, tile=4):
    """以 `tile` 为分块大小，计算 Flash Attention 风格的增量 softmax(qK^T)V。"""
    d_head = len(Vs[0])
    scale = 1.0 / math.sqrt(len(q))
    m = float("-inf")
    s = 0.0
    out = [0.0] * d_head
    for start in range(0, len(Ks), tile):
        k_block = Ks[start:start + tile]
        v_block = Vs[start:start + tile]
        scores = [dot(q, k) * scale for k in k_block]
        new_m = max(m, *scores)
        if m == float("-inf"):
            exp_old = 0.0
        else:
            exp_old = math.exp(m - new_m)
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(d_head):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]


class KVCache:
    def __init__(self):
        self.K = []
        self.V = []

    def append(self, k, v):
        self.K.append(k)
        self.V.append(v)

    def __len__(self):
        return len(self.K)


def decode_naive(all_K, all_V, all_queries):
    """每一步都在完整前缀上重新计算注意力。
    返回输出列表，每个生成的 token 对应一项。操作数 = 1+2+...+N = N(N+1)/2。
    """
    outputs = []
    ops = 0
    for t, q in enumerate(all_queries):
        Ks = all_K[:t + 1]
        Vs = all_V[:t + 1]
        out = attention_full(q, Ks, Vs)
        ops += t + 1
        outputs.append(out)
    return outputs, ops


def decode_cached(all_K, all_V, all_queries):
    """KV cache：每个新步骤追加一组 K、V，并对缓存执行查询。"""
    cache = KVCache()
    outputs = []
    ops = 0
    for q, k, v in zip(all_queries, all_K, all_V):
        cache.append(k, v)
        out = attention_full(q, cache.K, cache.V)
        ops += len(cache)
        outputs.append(out)
    return outputs, ops


def kv_cache_bytes(N, n_layers, n_heads_kv, d_head, dtype=2):
    """KV cache 总字节数。fp16/bf16 的 dtype=2，int8 为 1，fp32 为 4。"""
    return 2 * N * n_layers * n_heads_kv * d_head * dtype


def main():
    rng = random.Random(42)
    d_head = 8
    N = 10

    # 为 10-token 单头序列随机生成 Q、K、V。
    all_Q = [[rng.gauss(0, 1) for _ in range(d_head)] for _ in range(N)]
    all_K = [[rng.gauss(0, 1) for _ in range(d_head)] for _ in range(N)]
    all_V = [[rng.gauss(0, 1) for _ in range(d_head)] for _ in range(N)]

    naive, naive_ops = decode_naive(all_K, all_V, all_Q)
    cached, cached_ops = decode_cached(all_K, all_V, all_Q)

    print(f"=== N={N} 个 token 上的朴素解码与 KV cache 解码 ===")
    print(f"朴素注意力操作数：{naive_ops}  （O(N^2) = {N * (N + 1) // 2}）")
    print(f"缓存注意力操作数：{cached_ops}  （逐步成本为 O(N)，不变）")
    print("输出匹配（所有 token 上的最大绝对差）：",
          f"{max(abs(a - b) for va, vb in zip(naive, cached) for a, b in zip(va, vb)):.2e}")
    print()
    print("* 朴素方法的逐步成本相同；节省来自不再重新计算早期隐藏状态。")
    print("  若计入 K、V 的重新计算，朴素方法的矩阵乘法将达到 O(N^2)。")
    print()

    print("=== 分块 softmax（Flash）与标准 softmax 的一致性 ===")
    q = all_Q[-1]
    std = attention_full(q, all_K, all_V)
    for tile in [1, 2, 4, 8, 32]:
        tiled = tiled_softmax_dot(q, all_K, all_V, tile=tile)
        err = max(abs(a - b) for a, b in zip(std, tiled))
        print(f"  分块大小={tile:>3}  最大绝对差 = {err:.2e}")
    print("  除浮点重结合外逐位一致，不使用近似。")
    print()

    print("=== KV cache 大小表（fp16）===")
    configs = [
        ("Llama-3.2-3B",  28, 8,   128),
        ("Llama-3-8B",    32, 8,   128),
        ("Llama-3-70B",   80, 8,   128),
        ("Llama-3.1-405B", 126, 8, 128),
        ("Qwen2.5-72B",   80, 8,   128),
        ("DeepSeek-V3 (MLA)", 61, 1, 512),  # MLA 压缩至潜在表示；粗略估算
    ]
    for name, L, h_kv, d_h in configs:
        for N_ctx in [2048, 32768, 131072]:
            b = kv_cache_bytes(N_ctx, L, h_kv, d_h, dtype=2)
            print(f"  {name:<24}  N={N_ctx:>7}  -> {b / 1e9:.2f} GB")
    print()
    print("要点：在 128K 上下文下，70B 级稠密模型仅 KV 就需 10 GB 以上。")
    print("GQA 和 MLA 让现代长上下文推理的成本得以承受。")


if __name__ == "__main__":
    main()
