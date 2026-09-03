"""Q-Former cross-attention 玩具示例 — 纯标准库 Python。

构建一个最小的 BLIP-2-style 模态桥接：
  - 来自一个模拟 ViT 的 256 个 "patch token"
  - 32 个可学习的 query 向量
  - 一个 cross-attention 块（Q 来自 query，K/V 来自 patch）
  - 线性投影到 LLM 的隐藏维度
  - 打印注意力权重，让读者看到每个 query
    拉取了哪个 patch

纯 Python 向量和列表。不使用 numpy，不使用 torch。算术运算虽然慢，
但精确；适合用于检查行为。
"""

from __future__ import annotations

import math
import random

NUM_PATCH = 64
PATCH_DIM = 16
NUM_QUERY = 8
QUERY_DIM = 16
LLM_DIM = 24

rng = random.Random(42)


def vec(n: int) -> list[float]:
    return [rng.gauss(0, 1) for _ in range(n)]


def mat(rows: int, cols: int) -> list[list[float]]:
    return [vec(cols) for _ in range(rows)]


def matmul_vec(M: list[list[float]], v: list[float]) -> list[float]:
    return [sum(r * x for r, x in zip(row, v)) for row in M]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def make_patches() -> list[list[float]]:
    """模拟来自冻结 ViT 的 64 个 16 维 patch token。"""
    return [vec(PATCH_DIM) for _ in range(NUM_PATCH)]


def make_queries() -> list[list[float]]:
    """32 个 16 维可学习 query 向量。"""
    return [vec(QUERY_DIM) for _ in range(NUM_QUERY)]


def cross_attention(queries: list[list[float]],
                    patches: list[list[float]],
                    W_q: list[list[float]],
                    W_k: list[list[float]],
                    W_v: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """缩放点积交叉注意力。
    queries: (Nq, Dq) -> Q = queries @ W_q^T 形状 (Nq, D)
    patches: (Np, Dp) -> K, V
    返回 (attended, attn_weights)
    """
    Q = [matmul_vec(W_q, q) for q in queries]
    K = [matmul_vec(W_k, p) for p in patches]
    V = [matmul_vec(W_v, p) for p in patches]
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    attn_weights = []
    out = []
    for q in Q:
        logits = [dot(q, k) * scale for k in K]
        weights = softmax(logits)
        attn_weights.append(weights)
        mixed = [0.0] * d
        for i, w in enumerate(weights):
            for j in range(d):
                mixed[j] += w * V[i][j]
        out.append(mixed)
    return out, attn_weights


def linear_project(xs: list[list[float]],
                   W: list[list[float]]) -> list[list[float]]:
    return [matmul_vec(W, x) for x in xs]


def top_patches_per_query(attn: list[list[float]], k: int = 3) -> list[list[int]]:
    out = []
    for weights in attn:
        idxs = sorted(range(len(weights)), key=lambda i: -weights[i])[:k]
        out.append(idxs)
    return out


def summarize_attention(attn: list[list[float]]) -> None:
    print("\n注意力权重摘要（在 64 个 patch 上做 softmax）")
    print("-" * 60)
    top = top_patches_per_query(attn, k=5)
    entropies = []
    for weights in attn:
        e = -sum(w * math.log(w + 1e-12) for w in weights)
        entropies.append(e)
    avg_e = sum(entropies) / len(entropies)
    max_e = math.log(NUM_PATCH)
    for i, (idxs, e) in enumerate(zip(top, entropies)):
        top_str = ", ".join(f"p{x:02d}({attn[i][x]:.3f})" for x in idxs[:5])
        print(f"  query {i}: 熵 {e:.3f}/{max_e:.3f}, top-5 {top_str}")
    print(f"  平均熵: {avg_e:.3f}  （均匀分布基线: {max_e:.3f}）")


def demo_untrained() -> None:
    print("\n演示：8 个 query 在 64 个 patch 上做注意力")
    print("-" * 60)
    patches = make_patches()
    queries = make_queries()
    W_q = mat(QUERY_DIM, QUERY_DIM)
    W_k = mat(QUERY_DIM, PATCH_DIM)
    W_v = mat(QUERY_DIM, PATCH_DIM)
    attended, attn = cross_attention(queries, patches, W_q, W_k, W_v)
    summarize_attention(attn)
    W_out = mat(LLM_DIM, QUERY_DIM)
    projected = linear_project(attended, W_out)
    print(f"\n输出：{len(projected)} 个维度为 {LLM_DIM} 的 token，已可输入 LLM")
    print(f"第一个 token（已截断）: {[round(x, 2) for x in projected[0][:8]]}")


def demo_biased() -> None:
    """展示如果 query 学会了与特定 patch 对齐，注意力
    会集中（熵更低）。这里通过复用几个 patch
    向量作为 query 本身来模拟。"""
    print("\n演示：从特定 patch 初始化 query，令注意力集中")
    print("-" * 60)
    patches = make_patches()
    favored = [5, 17, 33, 48, 60, 2, 11, 27]
    queries = [list(patches[i]) for i in favored]
    W_q = [[1.0 if i == j else 0.0 for j in range(QUERY_DIM)]
           for i in range(QUERY_DIM)]
    W_k = [[1.0 if i == j else 0.0 for j in range(PATCH_DIM)]
           for i in range(QUERY_DIM)]
    W_v = [[1.0 if i == j else 0.0 for j in range(PATCH_DIM)]
           for i in range(QUERY_DIM)]
    _, attn = cross_attention(queries, patches, W_q, W_k, W_v)
    print("  query_i 应当对 patch[favored[i]] 的注意力最高：")
    for i, weights in enumerate(attn):
        top = max(range(len(weights)), key=lambda k: weights[k])
        hit = "命中" if top == favored[i] else "未命中"
        print(f"    query {i}：排名第一的 patch {top}（目标 {favored[i]}），"
              f"权重 {weights[top]:.3f}（{hit}）")


def main() -> None:
    print("=" * 60)
    print("BLIP-2 Q-FORMER 交叉注意力玩具示例（第 12 阶段，第 03 课）")
    print("=" * 60)
    demo_untrained()
    demo_biased()
    print("\n" + "=" * 60)
    print("要点")
    print("-" * 60)
    print("  · query 是桥接器中固定数量的可学习参数")
    print("  · 交叉注意力将（32 个 query，256 个 patch）映射为 32 个摘要")
    print("  · 投影到 LLM 隐藏维度后，前置于文本输入")
    print("  · BLIP-2 第 1 阶段使用 ITC+ITM+ITG 训练桥接器，不含 LLM")
    print("  · BLIP-2 第 2 阶段使用 LM 损失训练桥接器与投影器")


if __name__ == "__main__":
    main()
