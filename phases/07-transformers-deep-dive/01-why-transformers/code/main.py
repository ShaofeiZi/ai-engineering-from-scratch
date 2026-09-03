"""为何使用 transformer——演示 RNN 风格递归与注意力风格并行归约之间的
串行深度差距。

仅使用标准库运行，不需要 numpy 或 torch。
"""

import math
import time


def rnn_style(xs, decay=0.9):
    """顺序递归：h_t 依赖 h_{t-1}，无法并行。"""
    h = 0.0
    for x in xs:
        h = decay * h + x
    return h


def attention_style(xs):
    """顺序无关的归约：每个元素相互独立。"""
    return sum(xs) / len(xs)


def serial_scan(xs):
    """串行计算前缀和，深度为 O(N)。"""
    out = []
    acc = 0.0
    for x in xs:
        acc += x
        out.append(acc)
    return out


def parallel_scan(xs):
    """Hillis-Steele 并行前缀和，深度为 O(log N)。

    在纯 Python 中每一步仍然串行，但数据依赖图的深度为 log2(N)。
    在具有 N 路 SIMD 的真实硬件上，这会得到对数深度的扫描；在 CPU 上
    实际耗时相同，但对 GPU 内核而言，重要的是计算图的形状。
    """
    out = list(xs)
    step = 1
    n = len(out)
    while step < n:
        new = list(out)
        for i in range(step, n):
            new[i] = out[i] + out[i - step]
        out = new
        step *= 2
    return out


def benchmark(n, reps=3):
    xs = [0.001 * (i % 17) for i in range(n)]

    best_rnn = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = rnn_style(xs)
        best_rnn = min(best_rnn, time.perf_counter() - t0)

    best_attn = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = attention_style(xs)
        best_attn = min(best_attn, time.perf_counter() - t0)

    return best_rnn, best_attn


def depth(n):
    """计算 RNN 与注意力风格归约的串行深度。"""
    rnn_depth = n
    attn_depth = max(1, math.ceil(math.log2(n)))
    return rnn_depth, attn_depth


def main():
    print("=== 串行深度对比 ===")
    print(f"{'N':>8}  {'RNN 深度':>12}  {'注意力深度':>12}  {'加速比（操作数）':>16}")
    for n in [64, 512, 4096, 32768, 262144]:
        rd, ad = depth(n)
        print(f"{n:>8}  {rd:>12}  {ad:>12}  {rd / ad:>15.0f}x")

    print()
    print("=== 本机实际耗时（纯 Python）===")
    print(f"{'N':>8}  {'RNN (ms)':>10}  {'注意力 (ms)':>10}  {'比率':>8}")
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        rnn_t, attn_t = benchmark(n)
        ratio = rnn_t / attn_t if attn_t > 0 else float("inf")
        print(f"{n:>8}  {rnn_t * 1000:>10.2f}  {attn_t * 1000:>10.2f}  {ratio:>7.1f}x")

    print()
    print("=== 前缀和等价性检查 ===")
    xs = [float(i) for i in range(16)]
    ser = serial_scan(xs)
    par = parallel_scan(xs)
    mismatches = sum(1 for a, b in zip(ser, par) if abs(a - b) > 1e-9)
    print(f"长度：{len(xs)}，串行与并行扫描间的不匹配数：{mismatches}")
    print(f"最后一个值（串行）：{ser[-1]}")
    print(f"最后一个值（并行）：{par[-1]}")

    print()
    print("要点：除内存外，注意力在各个维度都更胜一筹。")
    print("完整注意力的内存成本为 O(N^2)；课程 12 将介绍解决方案。")


if __name__ == "__main__":
    main()
