import math
import random


def sample_mixture(n, rng):
    """双峰高斯混合：模式 A 位于 -2（sigma 0.6），模式 B 位于 +2（sigma 0.9）。"""
    samples = []
    for _ in range(n):
        if rng.random() < 0.4:
            samples.append(rng.gauss(-2.0, 0.6))
        else:
            samples.append(rng.gauss(2.0, 0.9))
    return samples


def histogram_density(samples, x, bin_width=0.25):
    """用直方图计算显式密度，返回 p(x) = 区间内样本数 / (n * bin_width)。"""
    n = len(samples)
    lo, hi = x - bin_width / 2, x + bin_width / 2
    count = sum(1 for s in samples if lo <= s < hi)
    return count / (n * bin_width)


def kde_density(samples, x, bandwidth=0.3):
    """用高斯核密度估计近似计算密度。"""
    n = len(samples)
    total = 0.0
    for s in samples:
        u = (x - s) / bandwidth
        total += math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    return total / (n * bandwidth)


def implicit_generator(samples, k, rng):
    """隐式生成器：采样一个训练点并加入微小噪声，不提供 p(x)。"""
    out = []
    for _ in range(k):
        base = rng.choice(samples)
        out.append(base + rng.gauss(0.0, 0.1))
    return out


def integrate_density(density_fn, samples, lo, hi, steps=200):
    """用梯形法则在 [lo, hi] 上对密度积分。"""
    xs = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    total = 0.0
    for i in range(steps):
        a, b = xs[i], xs[i + 1]
        total += 0.5 * (density_fn(samples, a) + density_fn(samples, b)) * (b - a)
    return total


def ascii_histogram(samples, lo=-5.0, hi=5.0, bins=40, height=12):
    """生成微型字符直方图，无需绘图库也能观察两个模式。"""
    width = (hi - lo) / bins
    counts = [0] * bins
    for s in samples:
        if lo <= s < hi:
            counts[int((s - lo) / width)] += 1
    peak = max(counts) or 1
    rows = []
    for row in range(height, 0, -1):
        threshold = peak * row / height
        line = "".join("#" if c >= threshold else " " for c in counts)
        rows.append(line)
    rows.append("-" * bins)
    rows.append(f"{lo:<.1f}" + " " * (bins - 8) + f"{hi:>.1f}")
    return "\n".join(rows)


def main():
    rng = random.Random(42)
    samples = sample_mixture(2000, rng)

    print("=== 从双峰高斯混合中采样 2000 个样本 ===")
    print(ascii_histogram(samples))
    print()

    query = 0.0
    print(f"用三种方式计算 p(x={query})：")
    print(f"  直方图密度：{histogram_density(samples, query):.4f}")
    print(f"  核密度：    {kde_density(samples, query):.4f}")
    print("  隐式生成器：N/A（只能采样，不提供密度）")
    print()

    p_hist = integrate_density(histogram_density, samples, -0.5, 0.5)
    p_kde = integrate_density(kde_density, samples, -0.5, 0.5)
    print("对 p(x in [-0.5, 0.5]) 积分：")
    print(f"  直方图：{p_hist:.3f}")
    print(f"  KDE：   {p_kde:.3f}")
    print()

    new_samples = implicit_generator(samples, 10, rng)
    print("隐式（类似 GAN）生成器生成的 10 个新样本：")
    print("  " + ", ".join(f"{s:+.2f}" for s in new_samples))
    print()

    print("要点：显式密度（文档中的第 1-2 类）可以回答")
    print("“这个点出现的可能性有多大？”，隐式模型（第 3 类）则不能。")


if __name__ == "__main__":
    main()
