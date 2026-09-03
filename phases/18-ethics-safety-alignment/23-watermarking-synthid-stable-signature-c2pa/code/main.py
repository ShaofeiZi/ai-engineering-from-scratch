"""token 水印玩具示例（SynthID-text 风格）——仅使用 Python 标准库。

词表：整数 0..N-1。每个解码步骤对前 k 个 token 求哈希并对 N 取模，将词表
划分为绿色（偶数哈希）和红色（奇数哈希）集合。采样偏向绿色集合。检测器
计算绿色 token 的 z-score，并在 1000 个 token 时报告结果。

用法：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import math
import random


random.seed(61)


VOCAB = 200
K = 4  # 哈希上下文长度。


def green_set(prev_tokens: list[int]) -> set[int]:
    """将词表伪随机划分出绿色集合（占一半）。"""
    seed = ",".join(str(t) for t in prev_tokens[-K:])
    digest = hashlib.sha256(seed.encode()).hexdigest()
    h = int(digest, 16)
    # 划分规则：当且仅当 (token + h) mod 2 == 0 时，token 为绿色。
    return {t for t in range(VOCAB) if (t + h) % 2 == 0}


def unwatermarked_sample(n: int, seed_prefix: list[int]) -> list[int]:
    out = list(seed_prefix)
    for _ in range(n):
        out.append(random.randrange(VOCAB))
    return out


def watermarked_sample(n: int, seed_prefix: list[int], bias: float = 0.9) -> list[int]:
    """Bias = 从绿色集合中采样的概率。"""
    out = list(seed_prefix)
    for _ in range(n):
        greens = green_set(out)
        use_green = random.random() < bias
        pool = list(greens) if use_green else list(set(range(VOCAB)) - greens)
        out.append(random.choice(pool))
    return out


def detect(tokens: list[int]) -> float:
    """返回 z-score：(绿色计数 - 期望值) / sqrt(期望值 * p(1-p))。"""
    if len(tokens) <= K:
        return 0.0
    green_count = 0
    for i in range(K, len(tokens)):
        greens = green_set(tokens[:i])
        if tokens[i] in greens:
            green_count += 1
    n = len(tokens) - K
    expected = n * 0.5
    std = math.sqrt(n * 0.5 * 0.5)
    return (green_count - expected) / std


def paraphrase(tokens: list[int], ratio: float = 0.3) -> list[int]:
    """随机将指定比例的 token 替换为随机 token。"""
    out = list(tokens)
    for i in range(len(out)):
        if random.random() < ratio:
            out[i] = random.randrange(VOCAB)
    return out


def main() -> None:
    print("=" * 70)
    print("TOKEN 水印玩具示例（阶段 18，第 23 课）")
    print("=" * 70)

    seed = [random.randrange(VOCAB) for _ in range(K)]

    watermarked = watermarked_sample(1000, seed)
    plain = unwatermarked_sample(1000, seed)

    print(f"\n带水印文本的 z-score：{detect(watermarked):.2f}")
    print(f"无水印文本的 z-score：{detect(plain):.2f}")
    print("（z >= 4 是存在水印的有力证据。）")

    # 改写攻击。
    para = paraphrase(watermarked, ratio=0.3)
    print(f"改写 30% 后：{detect(para):.2f}")
    para2 = paraphrase(watermarked, ratio=0.6)
    print(f"改写 60% 后：{detect(para2):.2f}")

    # 人类文本上的 FPR。
    fprs = [detect(unwatermarked_sample(1000, seed)) for _ in range(100)]
    fpr_above_4 = sum(1 for z in fprs if z >= 4) / len(fprs)
    print(f"\n100 次人类文本抽样中的 FPR（z >= 4）：{fpr_above_4:.3f}")

    print("\n" + "=" * 70)
    print("要点：文本达到 1000 个 token 时，可以凭借较高的 z-score 检测水印，")
    print("且 z=4 时 FPR <1%。改写 30% 会削弱信号，改写 60% 则会摧毁信号。")
    print("文本水印无法经受改写。部署时应组合 C2PA 元数据和水印：水印能经受")
    print("压缩，元数据只要不被移除也能保留。")
    print("=" * 70)


if __name__ == "__main__":
    main()
