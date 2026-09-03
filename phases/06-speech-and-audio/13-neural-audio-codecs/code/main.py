"""从零实现残差向量量化（RVQ）。

构建简化的一维信号，使用级联的微型码本进行量化，并测量增加码本时的
重建误差。说明现代音频编解码器为何使用 RVQ，而非单个巨型码本。

仅使用标准库。运行：python3 code/main.py
"""

import math
import random


def generate_signal(n=1000, seed=0):
    rng = random.Random(seed)
    return [math.sin(2 * math.pi * i / 100) + 0.3 * rng.gauss(0, 1.0) for i in range(n)]


def learn_codebook(values, size, iterations=20, seed=0):
    rng = random.Random(seed)
    if not values:
        return [0.0] * size
    lo, hi = min(values), max(values)
    centroids = [lo + (hi - lo) * rng.random() for _ in range(size)]
    for _ in range(iterations):
        buckets = [[] for _ in range(size)]
        for v in values:
            idx = min(range(size), key=lambda i: abs(centroids[i] - v))
            buckets[idx].append(v)
        for i in range(size):
            if buckets[i]:
                centroids[i] = sum(buckets[i]) / len(buckets[i])
    return sorted(centroids)


def quantize_with_codebook(values, codebook):
    indices = []
    residuals = []
    for v in values:
        idx = min(range(len(codebook)), key=lambda i: abs(codebook[i] - v))
        indices.append(idx)
        residuals.append(v - codebook[idx])
    return indices, residuals


def rvq_encode(values, codebook_size=8, n_codebooks=4):
    residuals = list(values)
    codebooks = []
    all_indices = []
    for cb_i in range(n_codebooks):
        cb = learn_codebook(residuals, codebook_size, seed=cb_i)
        codebooks.append(cb)
        indices, residuals = quantize_with_codebook(residuals, cb)
        all_indices.append(indices)
    return all_indices, codebooks


def rvq_decode(all_indices, codebooks, length):
    out = [0.0] * length
    for indices, cb in zip(all_indices, codebooks):
        for i, idx in enumerate(indices):
            out[i] += cb[idx]
    return out


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def main():
    print("=== 步骤 1：生成信号 ===")
    sig = generate_signal(n=1000)
    print(f"  长度：{len(sig)}   范围：[{min(sig):.2f}, {max(sig):.2f}]   均值：{sum(sig)/len(sig):.3f}")

    print()
    print("=== 步骤 2：RVQ 重建误差与码本数量的关系 ===")
    print("  codebook_size = 8   每个码本的值数量")
    print("  | 码本数量    | 位/帧      | MSE        | 50 fps 码率     |")

    for n_cb in [1, 2, 4, 8, 12]:
        indices, codebooks = rvq_encode(sig, codebook_size=8, n_codebooks=n_cb)
        recon = rvq_decode(indices, codebooks, length=len(sig))
        err = mse(sig, recon)
        bits_per_frame = n_cb * 3
        bitrate = bits_per_frame * 50
        print(f"  | {n_cb:>11} | {bits_per_frame:>10} | {err:.6f}   | {bitrate:>5} bps       |")

    print()
    print("=== 步骤 3：2026 年编解码器对比（语音 @ 6 kbps）===")
    rows = [
        ("EnCodec-24k", "75 Hz",   "3.2 PESQ", "通用音频、MusicGen"),
        ("DAC-44.1k",   "86 Hz",   "3.5 PESQ", "最高保真度"),
        ("SNAC-24k",    "~12 Hz",  "3.3 PESQ", "多尺度、AR-LM"),
        ("Mimi",        "12.5 Hz", "3.1 PESQ", "语义+声学、Moshi"),
    ]
    print("  | 编解码器     | 帧率       | 质量       | 用例                     |")
    for name, fr, q, u in rows:
        print(f"  | {name:<12} | {fr:<10} | {q:<10} | {u:<24} |")

    print()
    print("=== 步骤 4：语义 token 与声学 token（Mimi 概念演示）===")
    print("  码本 0    →  从 WavLM 蒸馏而来  →  内容（说了什么）")
    print("  码本 1–7  →  声学残差            →  音色、说话人、噪声")
    print()
    print("  LM 首先生成码本 0（文本 → 语义），然后")
    print("  以语义 + 说话人参考为条件生成码本 1–7")
    print("  = 可清晰支持声音克隆的因子化生成")

    print()
    print("要点：")
    print("  - RVQ：级联小码本优于单个巨型码本")
    print("  - 语义/声学拆分（Mimi、AudioLM）是 2024–2026 年的转变")
    print("  - 12.5 Hz Mimi × 8 个码本 = 每 10 秒音频片段 1000 个 token")
    print("  - 这使得音频 transformer LM 最终能在 2026 年的规模下工作")


if __name__ == "__main__":
    main()
