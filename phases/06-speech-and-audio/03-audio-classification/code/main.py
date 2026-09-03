"""音频分类基线：对均值+方差池化后的 MFCC 使用 k-NN。

四分类合成数据集：频率为 {200, 400, 800, 1600} Hz 且带高斯噪声的纯音。
训练并测试模型，然后输出混淆矩阵。

运行：python3 code/main.py
"""

import math
import random
from collections import Counter


def sine(freq_hz, sr, seconds, amp=0.5, phase=0.0):
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr + phase) for i in range(n)]


def add_noise(signal, sigma=0.05):
    return [s + random.gauss(0, sigma) for s in signal]


def hann(N):
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * n / (N - 1))) for n in range(N)]


def dft_mag(x):
    n = len(x)
    half = n // 2 + 1
    out = []
    for k in range(half):
        re = 0.0
        im = 0.0
        for j in range(n):
            angle = -2.0 * math.pi * k * j / n
            re += x[j] * math.cos(angle)
            im += x[j] * math.sin(angle)
        out.append(math.sqrt(re * re + im * im))
    return out


def frame_signal(sig, frame_len, hop):
    n = 1 + max(0, (len(sig) - frame_len) // hop)
    return [sig[i * hop : i * hop + frame_len] for i in range(n)]


def stft_mag(sig, frame_len, hop):
    w = hann(frame_len)
    frames = frame_signal(sig, frame_len, hop)
    return [dft_mag([w[j] * f[j] for j in range(frame_len)]) for f in frames]


def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sr):
    fmin, fmax = 0.0, sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    half = n_fft // 2 + 1
    bins = [min(half - 1, int(round(h * n_fft / sr))) for h in hzs]
    fb = [[0.0] * half for _ in range(n_mels)]
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            fb[m][k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[m][k] = (right - k) / max(1, right - center)
    return fb


def apply_filterbank(spec, fb):
    out = []
    for frame in spec:
        row = [sum(w * frame[k] for k, w in enumerate(f) if w) for f in fb]
        out.append(row)
    return out


def log_transform(x, eps=1e-10):
    return [[math.log(max(v, eps)) for v in row] for row in x]


def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]


def featurize(signal, sr, n_mfcc=13, n_mels=40, frame_len=256, hop=128):
    mag = stft_mag(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    lm = log_transform(mels)
    return [dct_ii(f, n_mfcc) for f in lm]


def summarize(frames):
    n = len(frames[0])
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(n)]
    var = [sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames) for i in range(n)]
    return mean + var


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)


def knn(q, bank, labels, k=3):
    idx = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in idx)
    return votes.most_common(1)[0][0]


def main():
    random.seed(42)
    sr = 8000
    duration = 0.25
    classes = {"low": 200, "mid_low": 400, "mid_high": 800, "high": 1600}
    per_class_train = 12
    per_class_test = 5

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("=== 构建四分类合成数据集（纯音 + 噪声）===")
    for label, freq in classes.items():
        for _ in range(per_class_train):
            sig = add_noise(sine(freq, sr, duration))
            X_train.append(summarize(featurize(sig, sr)))
            y_train.append(label)
        for _ in range(per_class_test):
            sig = add_noise(sine(freq, sr, duration))
            X_test.append(summarize(featurize(sig, sr)))
            y_test.append(label)
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print(f"  特征维度：{len(X_train[0])}  （13 个 MFCC 的均值+方差）")

    print()
    print("=== k-NN 分类（k=3）===")
    correct = 0
    confusion = {c: Counter() for c in classes}
    for feat, gold in zip(X_test, y_test):
        pred = knn(feat, X_train, y_train, k=3)
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
    acc = correct / len(X_test)
    print(f"  测试准确率：{acc:.3f} ({correct}/{len(X_test)})")

    print()
    print("=== 混淆矩阵（行=标准答案，列=预测结果）===")
    header = "  " + " ".join(f"{c[:8]:>10}" for c in classes)
    print(header)
    for gold in classes:
        row = f"  {gold[:8]:>8}"
        for pred in classes:
            row += f" {confusion[gold][pred]:>10}"
        print(row)

    print()
    print("要点：")
    print("  - 对均值+方差池化 MFCC 使用 k-NN，是一个出人意料的强基线")
    print("  - 真正的流水线使用 BEATs / AST 微调 + SpecAugment + mixup")


if __name__ == "__main__":
    main()
