"""从零学习音频基础：合成、DFT、峰值检测和混叠演示。

仅使用标准库：math、wave、struct、os、tempfile。
运行：python3 code/main.py
"""

import math
import os
import struct
import tempfile
import wave


def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr) for i in range(n)]


def mix(*signals):
    length = min(len(s) for s in signals)
    return [sum(s[i] for s in signals) / len(signals) for i in range(length)]


def write_wav(path, samples, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples)
        w.writeframes(frames)


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    ints = struct.unpack("<" + "h" * n, raw)
    return [x / 32768.0 for x in ints], sr


def dft(x):
    n = len(x)
    out = []
    for k in range(n):
        re = 0.0
        im = 0.0
        for j in range(n):
            angle = -2.0 * math.pi * k * j / n
            re += x[j] * math.cos(angle)
            im += x[j] * math.sin(angle)
        out.append((re, im))
    return out


def magnitudes(spectrum):
    return [math.sqrt(re * re + im * im) for re, im in spectrum]


def peak_freq(samples, sr):
    mags = magnitudes(dft(samples))
    half = len(mags) // 2
    mags = mags[:half]
    k = max(range(len(mags)), key=lambda i: mags[i])
    return k * sr / len(samples), k


def downsample_naive(samples, factor):
    return samples[::factor]


def main():
    sr = 8000
    duration = 0.064

    print("=== 步骤 1：合成 440 Hz 正弦波，8 kHz，64 ms ===")
    a = sine(440.0, sr, duration)
    print(f"  样本数：{len(a)}")
    print(f"  前 5 个：{[round(x, 4) for x in a[:5]]}")

    print()
    print("=== 步骤 2：通过 WAV 文件往返转换 ===")
    tmpdir = tempfile.mkdtemp(prefix="audio_fundamentals_")
    path = os.path.join(tmpdir, "a440.wav")
    write_wav(path, a, sr)
    loaded, loaded_sr = read_wav(path)
    size = os.path.getsize(path)
    print(f"  已写入 {path}（{size} 字节，采样率={loaded_sr}）")
    diff = max(abs(a[i] - loaded[i]) for i in range(len(a)))
    print(f"  往返转换最大绝对误差（16 位量化）：{diff:.5f}")

    print()
    print("=== 步骤 3：对 440 Hz 信号进行 DFT 峰值检测 ===")
    freq, k = peak_freq(a, sr)
    print(f"  峰值频点 k={k}，频率={freq:.1f} Hz（预期约 440.0 Hz，频点分辨率 {sr / len(a):.2f} Hz）")

    print()
    print("=== 步骤 4：混合信号（220 + 440 + 880）===")
    mixed = mix(sine(220, sr, duration), sine(440, sr, duration), sine(880, sr, duration))
    mags = magnitudes(dft(mixed))[: len(mixed) // 2]
    top3 = sorted(range(len(mags)), key=lambda i: -mags[i])[:3]
    peaks_hz = sorted(round(k * sr / len(mixed), 1) for k in top3)
    print(f"  最高的 3 个峰值：{peaks_hz} Hz")

    print()
    print("=== 步骤 5：混叠——以 10 kHz 对 7 kHz 音调采样 ===")
    alias_sr = 10000
    tone = sine(7000.0, alias_sr, 0.0512)
    alias_freq, _ = peak_freq(tone, alias_sr)
    folded = alias_sr - 7000.0
    print(f"  真实频率：7000.0 Hz（高于 Nyquist 频率 {alias_sr / 2} Hz）")
    print(f"  DFT 报告：{alias_freq:.1f} Hz")
    print(f"  预期混叠：{folded:.1f} Hz（= sr - f_true）")

    print()
    print("=== 步骤 6：正确降采样与朴素抽取对比 ===")
    orig_sr = 24000
    sig = sine(7000.0, orig_sr, 0.032)
    decimated = downsample_naive(sig, 3)
    new_sr = orig_sr // 3
    peak_new, _ = peak_freq(decimated, new_sr)
    print(f"  对 24 kHz 采样率的 7 kHz 音调不经低通滤波直接抽取至 8 kHz：")
    print(f"    抽取后峰值：{peak_new:.1f} Hz（折叠后应为 1000 Hz）")
    print(f"    结论：抽取前始终要进行低通滤波")


if __name__ == "__main__":
    main()
