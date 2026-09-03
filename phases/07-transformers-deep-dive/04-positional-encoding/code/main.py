"""位置编码——正弦编码、RoPE、ALiBi。

仅使用标准库。每种编码方案都封装为可复用的小函数。
通过数值演示验证 RoPE 的相对距离特性。
"""

import math
import random


def sinusoidal_pe(n, d, base=10000.0):
    pe = [[0.0] * d for _ in range(n)]
    for pos in range(n):
        for i in range(d // 2):
            theta = pos / (base ** (2 * i / d))
            pe[pos][2 * i] = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe


def apply_rope(x, pos, base=10000.0):
    """将 x 的奇偶元素对旋转 pos * theta_i 角度。"""
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c = math.cos(theta)
        s = math.sin(theta)
        a = x[2 * i]
        b = x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def alibi_slopes(n_heads):
    return [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]


def alibi_bias(n_heads, seq_len, causal=True):
    slopes = alibi_slopes(n_heads)
    out = []
    for m in slopes:
        head_bias = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if causal and j > i:
                    row.append(float("-inf"))
                else:
                    row.append(-m * abs(i - j))
            head_bias.append(row)
        out.append(head_bias)
    return out


def demo_sinusoidal():
    print("=== 正弦位置编码 ===")
    pe = sinusoidal_pe(n=8, d=8)
    print("前 4 个位置、前 4 个维度：")
    for pos in range(4):
        print(f"  pos={pos}: " + "  ".join(f"{v:+.3f}" for v in pe[pos][:4]))
    print()


def demo_rope_relative():
    print("=== RoPE：点积仅取决于相对距离 ===")
    rng = random.Random(0)
    d = 16
    q = [rng.gauss(0, 1) for _ in range(d)]
    k = [rng.gauss(0, 1) for _ in range(d)]

    pairs = [(3, 5), (7, 9), (100, 102), (1024, 1026)]
    print(f"{'pos_q':>6}  {'pos_k':>6}  {'gap':>4}  {'<q_rot, k_rot>':>18}")
    for pq, pk in pairs:
        q_rot = apply_rope(q, pq)
        k_rot = apply_rope(k, pk)
        d_prod = dot(q_rot, k_rot)
        print(f"{pq:>6}  {pk:>6}  {pk - pq:>4}  {d_prod:>18.6f}")
    print("间隔为 2 的所有行都应具有相同点积。")
    print()


def demo_rope_base_scaling():
    print("=== RoPE 基数缩放（面向长上下文的 NTK-aware 方法）===")
    rng = random.Random(1)
    d = 8
    q = [rng.gauss(0, 1) for _ in range(d)]
    k = [rng.gauss(0, 1) for _ in range(d)]

    for base in [10000, 100000, 1_000_000]:
        q_rot = apply_rope(q, pos=4096, base=base)
        k_rot = apply_rope(k, pos=4098, base=base)
        print(f"  基数={base:>8d}  得分={dot(q_rot, k_rot):+.6f}")
    print("基数越大 = 旋转越慢 = 不发生相位环绕的上下文越长。")
    print()


def demo_alibi():
    print("=== ALiBi 偏置矩阵 ===")
    n_heads = 4
    slopes = alibi_slopes(n_heads)
    print(f"头数为 {n_heads} 时的斜率：" + ", ".join(f"{s:.4f}" for s in slopes))
    bias = alibi_bias(n_heads, seq_len=6, causal=False)
    print(f"第 0 个头的偏置（token 越近，惩罚越小）：")
    for row in bias[0]:
        print("  " + "  ".join(f"{v:+6.2f}" for v in row))
    print()


def main():
    demo_sinusoidal()
    demo_rope_relative()
    demo_rope_base_scaling()
    demo_alibi()
    print("要点：RoPE 在点积本身中编码相对位置。")
    print("ALiBi 完全跳过嵌入。到 2026 年，正弦位置编码已退居次要位置。")


if __name__ == "__main__":
    main()
