"""Flamingo 门控 cross-attention + Perceiver 重采样器玩具 — 纯标准库 Python。

演示内容：
  - Perceiver 重采样器：可变长度的 patch token -> 固定长度的 latent
  - 门控 cross-attention：tanh(alpha) * 交叉注意力 + x 残差
  - alpha=0 -> 视觉贡献恰好为零（冻结的 LLM 被保留）
  - 交错序列注意力掩码，用于 (img1, txt1, img2, txt2)

纯 Python 实现，不依赖 numpy，不依赖 torch。
"""

from __future__ import annotations

import math
import random

rng = random.Random(7)


def vec(n: int) -> list[float]:
    return [rng.gauss(0, 0.3) for _ in range(n)]


def mat(rows: int, cols: int) -> list[list[float]]:
    return [vec(cols) for _ in range(rows)]


def matvec(M: list[list[float]], v: list[float]) -> list[float]:
    return [sum(r * x for r, x in zip(row, v)) for row in M]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


def cross_attention(queries: list[list[float]],
                    keys: list[list[float]],
                    values: list[list[float]]) -> list[list[float]]:
    d = len(queries[0])
    scale_f = 1.0 / math.sqrt(d)
    out = []
    for q in queries:
        logits = [dot(q, k) * scale_f for k in keys]
        w = softmax(logits)
        mixed = [0.0] * d
        for i, wi in enumerate(w):
            for j in range(d):
                mixed[j] += wi * values[i][j]
        out.append(mixed)
    return out


def perceiver_resampler(patches: list[list[float]], num_latents: int,
                        num_blocks: int = 2) -> list[list[float]]:
    """可变数量 patch -> 固定 K 个 latent，通过 cross-attention."""
    dim = len(patches[0])
    latents = [vec(dim) for _ in range(num_latents)]
    for _ in range(num_blocks):
        attended = cross_attention(latents, patches, patches)
        latents = [add(lat, att) for lat, att in zip(latents, attended)]
    return latents


def gated_cross_attention_step(text_hidden: list[list[float]],
                               visual_tokens: list[list[float]],
                               alpha: float) -> list[list[float]]:
    """计算门控交叉注意力：y = tanh(alpha) * cross_attn(text, visual) + text_hidden。"""
    cross = cross_attention(text_hidden, visual_tokens, visual_tokens)
    gate = math.tanh(alpha)
    out = [add(t, scale(c, gate)) for t, c in zip(text_hidden, cross)]
    return out


def interleaved_mask(sequence: list[str]) -> list[list[bool]]:
    """构建一个 cross-attn 掩码，使每个文本 token 只关注
    最近的前一个图像。
    序列：标签形如 ['IMG0', 'txt0a', 'txt0b', 'IMG1', 'txt1a', 'txt1b']。
    返回一个 (文本 token) x (图像 token) 的掩码，True = 允许关注。
    """
    text_positions = [i for i, s in enumerate(sequence) if not s.startswith("IMG")]
    image_positions = [i for i, s in enumerate(sequence) if s.startswith("IMG")]

    mask = [[False] * len(image_positions) for _ in text_positions]
    for ti, tpos in enumerate(text_positions):
        preceding = [i for i in image_positions if i < tpos]
        if not preceding:
            continue
        most_recent_img = preceding[-1]
        img_index = image_positions.index(most_recent_img)
        mask[ti][img_index] = True
    return mask


def demo_resampler() -> None:
    print("\n演示 1：Perceiver 重采样器")
    print("-" * 60)
    for num_patches in (36, 196, 900):
        patches = [vec(16) for _ in range(num_patches)]
        latents = perceiver_resampler(patches, num_latents=8, num_blocks=2)
        print(f"  输入 {num_patches} 个 patch -> 输出 {len(latents)} 个维度为 "
              f"{len(latents[0])} 的 latent（形状不随输入变化）")


def demo_gate() -> None:
    print("\n演示 2：门控 cross-attention")
    print("-" * 60)
    text_hidden = [vec(16) for _ in range(5)]
    visual = [vec(16) for _ in range(8)]

    out_closed = gated_cross_attention_step(text_hidden, visual, alpha=0.0)
    deltas = [max(abs(a - b) for a, b in zip(o, t))
              for o, t in zip(out_closed, text_hidden)]
    print(f"  alpha=0.0 (tanh=0.0)：相对输入的最大差值 = {max(deltas):.6f}")
    print("  -> 冻结的 LLM 在初始化时被精确保留")

    out_open = gated_cross_attention_step(text_hidden, visual, alpha=2.0)
    deltas = [sum(abs(a - b) for a, b in zip(o, t)) / len(o)
              for o, t in zip(out_open, text_hidden)]
    print(f"  alpha=2.0 (tanh=0.96)：相对输入的平均差值 = {sum(deltas)/len(deltas):.4f}")
    print("  -> 视觉贡献已混入")

    for a in (0.0, 0.5, 1.0, 2.0, 5.0):
        g = math.tanh(a)
        print(f"    alpha={a:4.1f}  tanh(alpha)={g:+.4f}")


def demo_interleaved_mask() -> None:
    print("\n演示 3：交错注意力掩码")
    print("-" * 60)
    seq = ["IMG0", "t0a", "t0b", "IMG1", "t1a", "t1b", "t1c", "IMG2", "t2a"]
    mask = interleaved_mask(seq)
    image_labels = [s for s in seq if s.startswith("IMG")]
    text_labels = [s for s in seq if not s.startswith("IMG")]

    header = "         " + "  ".join(f"{x:4s}" for x in image_labels)
    print(header)
    for i, tk in enumerate(text_labels):
        row = "  ".join(" 是 " if mask[i][j] else "  . " for j in range(len(image_labels)))
        print(f"  {tk:5s}：{row}")
    print("  每个文本 token 只看到最近的前一个图像")


def main() -> None:
    print("=" * 60)
    print("FLAMINGO 门控交叉注意力玩具示例（第 12 阶段，第 04 课）")
    print("=" * 60)
    demo_resampler()
    demo_gate()
    demo_interleaved_mask()
    print("\n" + "=" * 60)
    print("要点")
    print("-" * 60)
    print("  · 无论输入大小如何，Perceiver 重采样器都输出固定 K 个 latent")
    print("  · alpha=0 时 tanh(alpha) 门为恒等操作，初始化时保留 LLM")
    print("  · 交错掩码让文本 token 关注前面的图像")
    print("  · Flamingo 每隔 4 个 LLM 层插入门控交叉注意力")


if __name__ == "__main__":
    main()
