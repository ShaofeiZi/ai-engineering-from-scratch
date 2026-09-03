"""使用标准库数学运算构建频谱图、Mel 滤波器组和 MFCC。

运行：python3 code/main.py
"""

import math


def sine(freq_hz, sr, seconds, amp=0.5, phase=0.0):
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr + phase) for i in range(n)]


def chirp(f0, f1, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    out = []
    for i in range(n):
        t = i / sr
        f = f0 + (f1 - f0) * (t / seconds)
        out.append(amp * math.sin(2.0 * math.pi * f * t))
    return out


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


def frame_signal(signal, frame_len, hop):
    if len(signal) < frame_len:
        return []
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]


def stft_magnitude(signal, frame_len, hop):
    w = hann(frame_len)
    frames = frame_signal(signal, frame_len, hop)
    return [dft_mag([w[j] * f[j] for j in range(frame_len)]) for f in frames]


def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    m_lo = hz_to_mel(fmin)
    m_hi = hz_to_mel(fmax)
    mels = [m_lo + (m_hi - m_lo) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    half = n_fft // 2 + 1
    bins = [min(half - 1, int(round(h * n_fft / sr))) for h in hzs]
    fb = [[0.0] * half for _ in range(n_mels)]
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            denom = max(1, center - left)
            fb[m][k] = (k - left) / denom
        for k in range(center, right):
            denom = max(1, right - center)
            fb[m][k] = (right - k) / denom
    return fb


def apply_filterbank(stft_mag, fb):
    n_mels = len(fb)
    result = []
    for spec in stft_mag:
        frame_mels = []
        for m in range(n_mels):
            val = 0.0
            for k, w in enumerate(fb[m]):
                if w:
                    val += spec[k] * w
            frame_mels.append(val)
        result.append(frame_mels)
    return result


def log_transform(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]


def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]


def main():
    sr = 8000
    frame_len = 256
    hop = 128
    n_mels = 40
    n_fft = frame_len

    print("=== 步骤 1：对 0.5 秒、2 kHz 音调分帧 ===")
    tone = sine(2000.0, sr, 0.5)
    frames = frame_signal(tone, frame_len, hop)
    print(f"  样本数：{len(tone)}，帧数：{len(frames)}，帧长：{frame_len}，步长：{hop}")

    print()
    print("=== 步骤 2：Hann 窗衰减帧边缘 ===")
    w = hann(frame_len)
    print(f"  hann(起点) = {w[0]:.4f}   hann(中点) = {w[frame_len // 2]:.4f}   hann(终点) = {w[-1]:.4f}")

    print()
    print("=== 步骤 3：音调的 STFT；最大频点位于 2000 Hz ===")
    mag = stft_magnitude(tone, frame_len, hop)
    mid = mag[len(mag) // 2]
    k_peak = max(range(len(mid)), key=lambda i: mid[i])
    print(f"  帧数：{len(mag)}，每帧频点数：{len(mid)}")
    print(f"  峰值频点：{k_peak}，频率：{k_peak * sr / n_fft:.1f} Hz（预期 2000 Hz）")

    print()
    print("=== 步骤 4：Mel 滤波器组，40 个 Mel，0–4000 Hz ===")
    fb = mel_filterbank(n_mels, n_fft, sr)
    mel_widths = [sum(1 for x in f if x > 0) for f in fb]
    print(f"  滤波器组形状：{n_mels} x {len(fb[0])}")
    print(f"  频点宽度（前 6 个）：{mel_widths[:6]}   （后 6 个）：{mel_widths[-6:]}")
    print("  注意：低 Mel 滤波器较窄（密集），高 Mel 滤波器较宽（稀疏）。")

    print()
    print("=== 步骤 5：200 Hz -> 4000 Hz 啁啾信号；各帧最大 Mel 频点 ===")
    c = chirp(200.0, 4000.0, sr, 0.4)
    cmag = stft_magnitude(c, frame_len, hop)
    mel_spec = apply_filterbank(cmag, fb)
    lm = log_transform(mel_spec)
    print("  帧 -> 最大 Mel 频点：")
    step = max(1, len(lm) // 10)
    for i in range(0, len(lm), step):
        am = max(range(n_mels), key=lambda m: lm[i][m])
        print(f"    t={i:3d}  argmax_mel={am:2d}")

    print()
    print("=== 步骤 6：单个 Mel 帧的 MFCC-13 ===")
    mfcc = dct_ii(lm[len(lm) // 2], 13)
    print(f"  MFCC（13 个系数，中间帧）：{[round(c, 3) for c in mfcc]}")
    print("  注意：系数 0 编码总体能量，通常会在下游丢弃。")


if __name__ == "__main__":
    main()
