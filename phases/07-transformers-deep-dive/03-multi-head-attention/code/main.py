"""仅使用标准库从零实现多头注意力。

不使用 numpy 或 torch。由一个微型 Matrix 类承载所需运算。
演示内容包括：拆分注意力头、逐头缩放点积注意力、合并注意力头、
输出投影，以及分组查询注意力变体。
"""

import math
import random
from typing import List


class Matrix:
    """浮点数行主序二维矩阵，仅实现注意力所需的运算。"""

    __slots__ = ("rows", "cols", "data")

    def __init__(self, rows: int, cols: int, fill: float = 0.0, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = data
        else:
            self.data = [fill] * (rows * cols)

    def get(self, i: int, j: int) -> float:
        return self.data[i * self.cols + j]

    def set(self, i: int, j: int, v: float) -> None:
        self.data[i * self.cols + j] = v

    def row(self, i: int) -> List[float]:
        return self.data[i * self.cols:(i + 1) * self.cols]


def randn_matrix(rows, cols, rng, scale=None):
    if scale is None:
        scale = math.sqrt(2.0 / (rows + cols))
    m = Matrix(rows, cols)
    for i in range(rows * cols):
        m.data[i] = rng.gauss(0.0, scale)
    return m


def matmul(A: Matrix, B: Matrix) -> Matrix:
    assert A.cols == B.rows, f"{A.cols} vs {B.rows}"
    out = Matrix(A.rows, B.cols)
    for i in range(A.rows):
        for k in range(A.cols):
            aik = A.get(i, k)
            if aik == 0.0:
                continue
            base_i = i * B.cols
            base_k = k * B.cols
            for j in range(B.cols):
                out.data[base_i + j] += aik * B.data[base_k + j]
    return out


def transpose(A: Matrix) -> Matrix:
    out = Matrix(A.cols, A.rows)
    for i in range(A.rows):
        for j in range(A.cols):
            out.set(j, i, A.get(i, j))
    return out


def softmax_rows(A: Matrix) -> Matrix:
    out = Matrix(A.rows, A.cols)
    for i in range(A.rows):
        row = A.row(i)
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        s = sum(exps)
        for j, e in enumerate(exps):
            out.set(i, j, e / s)
    return out


def scaled_dot_product_attention(Q: Matrix, K: Matrix, V: Matrix):
    dk = Q.cols
    scale = 1.0 / math.sqrt(dk)
    scores = matmul(Q, transpose(K))
    for i in range(scores.rows * scores.cols):
        scores.data[i] *= scale
    weights = softmax_rows(scores)
    out = matmul(weights, V)
    return out, weights


def split_heads(X: Matrix, n_heads: int) -> List[Matrix]:
    assert X.cols % n_heads == 0, "d_model 不能被 n_heads 整除"
    d_head = X.cols // n_heads
    heads = []
    for h in range(n_heads):
        H = Matrix(X.rows, d_head)
        for i in range(X.rows):
            for j in range(d_head):
                H.set(i, j, X.get(i, h * d_head + j))
        heads.append(H)
    return heads


def combine_heads(heads: List[Matrix]) -> Matrix:
    n = heads[0].rows
    d_head = heads[0].cols
    d_model = d_head * len(heads)
    out = Matrix(n, d_model)
    for h, H in enumerate(heads):
        for i in range(n):
            for j in range(d_head):
                out.set(i, h * d_head + j, H.get(i, j))
    return out


def multi_head_attention(X: Matrix, Wq, Wk, Wv, Wo, n_heads: int):
    Q = matmul(X, Wq)
    K = matmul(X, Wk)
    V = matmul(X, Wv)
    Qh = split_heads(Q, n_heads)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    head_outs = []
    per_head_weights = []
    for q, k, v in zip(Qh, Kh, Vh):
        o, w = scaled_dot_product_attention(q, k, v)
        head_outs.append(o)
        per_head_weights.append(w)
    concat = combine_heads(head_outs)
    return matmul(concat, Wo), per_head_weights


def grouped_query_attention(X: Matrix, Wq, Wk, Wv, Wo, n_heads: int, n_kv_heads: int):
    """与 MHA 相同，但 K 和 V 的头更少，并通过重复与 Q 匹配。"""
    Q = matmul(X, Wq)
    K = matmul(X, Wk)
    V = matmul(X, Wv)
    Qh = split_heads(Q, n_heads)
    Kh_small = split_heads(K, n_kv_heads)
    Vh_small = split_heads(V, n_kv_heads)
    repeat = n_heads // n_kv_heads
    Kh = [Kh_small[i // repeat] for i in range(n_heads)]
    Vh = [Vh_small[i // repeat] for i in range(n_heads)]
    head_outs = []
    for q, k, v in zip(Qh, Kh, Vh):
        o, _ = scaled_dot_product_attention(q, k, v)
        head_outs.append(o)
    concat = combine_heads(head_outs)
    return matmul(concat, Wo)


def print_matrix(name, M: Matrix, width=6, prec=3):
    print(f"-- {name} ({M.rows}x{M.cols}) --")
    for i in range(M.rows):
        row = M.row(i)
        print("  " + "  ".join(f"{v:>{width}.{prec}f}" for v in row))


def main():
    rng = random.Random(42)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)
    d_model = 8
    n_heads = 2

    X = randn_matrix(n, d_model, rng, scale=1.0)
    Wq = randn_matrix(d_model, d_model, rng)
    Wk = randn_matrix(d_model, d_model, rng)
    Wv = randn_matrix(d_model, d_model, rng)
    Wo = randn_matrix(d_model, d_model, rng)

    out, weights = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=n_heads)

    print(f"=== 多头注意力：{n_heads} 个头，d_model={d_model}，d_head={d_model // n_heads} ===")
    print(f"输入形状：({X.rows}, {X.cols})")
    print(f"输出形状：({out.rows}, {out.cols})")
    print()
    for h, W in enumerate(weights):
        print(f"-- 第 {h} 个头的注意力权重 --")
        print(f"{'':>6}", end="")
        for t in tokens:
            print(f"{t:>7}", end="")
        print()
        for i in range(n):
            print(f"{tokens[i]:>6}", end="")
            for j in range(n):
                print(f"{W.get(i, j):>7.3f}", end="")
            print()
        print()

    # GQA 演示：4 个 Q 头，2 个 KV 头
    d_model = 8
    n_heads = 4
    n_kv = 2
    Wq = randn_matrix(d_model, d_model, rng)
    Wk = randn_matrix(d_model, (d_model // n_heads) * n_kv, rng)
    Wv = randn_matrix(d_model, (d_model // n_heads) * n_kv, rng)
    Wo = randn_matrix(d_model, d_model, rng)
    out_gqa = grouped_query_attention(X, Wq, Wk, Wv, Wo, n_heads=n_heads, n_kv_heads=n_kv)
    print(f"=== GQA：{n_heads} 个 Q 头，{n_kv} 个 KV 头 ===")
    print(f"输出形状：({out_gqa.rows}, {out_gqa.cols})")
    kv_cache_full = n_heads * n * (d_model // n_heads) * 2
    kv_cache_gqa = n_kv * n * (d_model // n_heads) * 2
    print(f"KV cache 元素数（MHA）：{kv_cache_full}")
    print(f"KV cache 元素数（GQA）：{kv_cache_gqa}  （缩小 {kv_cache_full // kv_cache_gqa} 倍）")


if __name__ == "__main__":
    main()
