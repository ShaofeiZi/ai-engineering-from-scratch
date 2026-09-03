import math
from pathlib import Path


class Complex:
    def __init__(self, real=0.0, imag=0.0):
        self.real = float(real)
        self.imag = float(imag)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real + other, self.imag)
        return Complex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real - other, self.imag)
        return Complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other)
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __rmul__(self, other):
        return self.__mul__(other)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def __repr__(self):
        if abs(self.imag) < 1e-12:
            return f"{self.real:.6f}"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6f} {sign} {abs(self.imag):.6f}i"


def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))


def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * euler(angle)
        result.append(total)
    return result


def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            xk = X[k] if isinstance(X[k], Complex) else Complex(X[k])
            total = total + xk * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result


def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = euler(angle)
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result


def ifft(X):
    N = len(X)
    conj_X = [xk.conjugate() if isinstance(xk, Complex) else Complex(xk) for xk in X]
    result = fft(conj_X)
    return [Complex(r.real / N, -r.imag / N) for r in result]


def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]


def magnitude_spectrum(X):
    return [xk.magnitude() for xk in X]


def spectral_analysis(signal, sample_rate):
    N = len(signal)
    X = fft(signal)
    magnitudes = magnitude_spectrum(X)
    freqs = [k * sample_rate / N for k in range(N)]
    return freqs[:N // 2 + 1], magnitudes[:N // 2 + 1]


def hann_window(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]


def hamming_window(N):
    return [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]


def apply_window(signal, window):
    return [s * w for s, w in zip(signal, window)]


def convolve_direct(x, h):
    N = len(x)
    M = len(h)
    out_len = N + M - 1
    result = [0.0] * out_len
    for n in range(out_len):
        total = 0.0
        for k in range(M):
            if 0 <= n - k < N:
                total += x[n - k] * h[k]
        result[n] = total
    return result


def convolve_fft(x, h):
    if len(x) == 0 or len(h) == 0:
        return []
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = list(x) + [0.0] * (padded_N - len(x))
    h_padded = list(h) + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = ifft(Y)
    return [y[n].real for n in range(N)]


def generate_signal(frequencies, amplitudes, N, sample_rate):
    signal = [0.0] * N
    for freq, amp in zip(frequencies, amplitudes):
        for n in range(N):
            t = n / sample_rate
            signal[n] += amp * math.sin(2 * math.pi * freq * t)
    return signal


def positional_encoding(pos, d_model):
    pe = [0.0] * d_model
    for i in range(d_model // 2):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        angle = pos * freq
        pe[2 * i] = math.sin(angle)
        pe[2 * i + 1] = math.cos(angle)
    return pe


def demo_pure_sine():
    print("=" * 65)
    print("  纯正弦波的 DFT")
    print("=" * 65)
    print()

    N = 32
    sample_rate = 32
    freq = 5
    signal = generate_signal([freq], [1.0], N, sample_rate)

    print(f"  信号: sin(2*pi*{freq}*t)，{N} 个采样点，采样率 {sample_rate} Hz")
    print()

    X = dft(signal)
    mags = magnitude_spectrum(X)

    print(f"  {'频率桶 k':<12s} {'频率 (Hz)':>14s} {'|X[k]|':>10s}")
    print(f"  {'-' * 12} {'-' * 14} {'-' * 10}")

    for k in range(N // 2 + 1):
        f_hz = k * sample_rate / N
        if mags[k] > 0.01:
            print(f"  k={k:<8d} {f_hz:>14.1f} {mags[k]:>10.4f}")

    print()
    print(f"  峰值出现在 k={freq}，对应 {freq} Hz。")
    print(f"  DFT 正确识别出了该频率。")


def demo_multi_frequency():
    print()
    print()
    print("=" * 65)
    print("  叠加正弦波的 DFT")
    print("=" * 65)
    print()

    N = 64
    sample_rate = 64
    freqs = [3, 7, 15]
    amps = [1.0, 0.5, 0.3]

    signal = generate_signal(freqs, amps, N, sample_rate)

    print(f"  信号: {amps[0]}*sin(2*pi*{freqs[0]}*t) + "
          f"{amps[1]}*sin(2*pi*{freqs[1]}*t) + "
          f"{amps[2]}*sin(2*pi*{freqs[2]}*t)")
    print(f"  {N} 个采样点，采样率 {sample_rate} Hz")
    print()

    X = fft(signal)
    mags = magnitude_spectrum(X)

    print(f"  恢复出的频率（幅值 > 0.5）:")
    print(f"  {'频率 (Hz)':>10s} {'|X[k]|':>10s} {'期望幅值 * N/2':>20s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 20}")

    for k in range(N // 2 + 1):
        if mags[k] > 0.5:
            f_hz = k * sample_rate / N
            expected = ""
            for freq, amp in zip(freqs, amps):
                if abs(f_hz - freq) < 0.1:
                    expected = f"{amp * N / 2:.1f}"
            print(f"  {f_hz:>10.1f} {mags[k]:>10.4f} {expected:>20s}")

    print()
    print("  三个频率均被正确恢复。")
    print("  幅值与期望值一致（幅值 * N/2）。")


def demo_fft_vs_dft():
    print()
    print()
    print("=" * 65)
    print("  FFT 与 DFT：结果相同，速度更快")
    print("=" * 65)
    print()

    N = 32
    import random
    random.seed(42)
    signal = [random.gauss(0, 1) for _ in range(N)]

    X_dft = dft(signal)
    X_fft = fft(signal)

    max_error = 0.0
    for k in range(N):
        diff_real = abs(X_dft[k].real - X_fft[k].real)
        diff_imag = abs(X_dft[k].imag - X_fft[k].imag)
        max_error = max(max_error, diff_real, diff_imag)

    print(f"  随机信号，N = {N}")
    print(f"  DFT 与 FFT 的最大差异: {max_error:.2e}")
    print(f"  是否一致: {max_error < 1e-10}")
    print()

    print(f"  {'k':<6s} {'DFT |X[k]|':>14s} {'FFT |X[k]|':>14s} {'差异':>12s}")
    print(f"  {'-' * 6} {'-' * 14} {'-' * 14} {'-' * 12}")
    for k in range(8):
        d_mag = X_dft[k].magnitude()
        f_mag = X_fft[k].magnitude()
        diff = abs(d_mag - f_mag)
        print(f"  {k:<6d} {d_mag:>14.8f} {f_mag:>14.8f} {diff:>12.2e}")

    print(f"  ...（还有 {N - 8} 个系数）")
    print()

    print(f"  DFT 复杂度: O(N^2) = {N * N} 次乘法")
    print(f"  FFT 复杂度: O(N*log2(N)) = {int(N * math.log2(N))} 次乘法")
    print(f"  加速比: {N * N / (N * math.log2(N)):.1f}x")


def demo_reconstruction():
    print()
    print()
    print("=" * 65)
    print("  完美重建: DFT -> IDFT")
    print("=" * 65)
    print()

    import random
    random.seed(99)
    N = 16
    signal = [random.gauss(0, 2) for _ in range(N)]

    X = fft(signal)
    reconstructed = ifft(X)

    max_err = max(abs(reconstructed[n].real - signal[n]) for n in range(N))

    print(f"  原始信号与重建信号（N={N}）:")
    print(f"  {'n':<4s} {'原始':>12s} {'重建':>14s} {'误差':>12s}")
    print(f"  {'-' * 4} {'-' * 12} {'-' * 14} {'-' * 12}")

    for n in range(N):
        err = abs(reconstructed[n].real - signal[n])
        print(f"  {n:<4d} {signal[n]:>12.6f} {reconstructed[n].real:>14.6f} {err:>12.2e}")

    print()
    print(f"  最大重建误差: {max_err:.2e}")
    print(f"  完美重建: {max_err < 1e-10}")


def demo_convolution_theorem():
    print()
    print()
    print("=" * 65)
    print("  卷积定理")
    print("=" * 65)
    print()

    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = [1.0, 1.0, 1.0]

    direct = convolve_direct(x, h)
    fft_result = convolve_fft(x, h)

    print(f"  信号 x = {x}")
    print(f"  滤波器 h = {h}")
    print(f"  线性卷积 (x * h):")
    print()

    print(f"  {'n':<4s} {'直接':>10s} {'基于 FFT':>10s} {'差异':>12s}")
    print(f"  {'-' * 4} {'-' * 10} {'-' * 10} {'-' * 12}")

    max_err = 0.0
    for n in range(len(direct)):
        diff = abs(direct[n] - fft_result[n])
        max_err = max(max_err, diff)
        print(f"  {n:<4d} {direct[n]:>10.4f} {fft_result[n]:>10.4f} {diff:>12.2e}")

    print()
    print(f"  最大差异: {max_err:.2e}")
    print(f"  是否一致: {max_err < 1e-8}")
    print()
    print("  时域卷积 = 频域乘法。")
    print("  直接卷积: O(N*M) = O(15)")
    print("  FFT 卷积: 当 N 较大时为 O(N*log(N))")


def demo_windowing():
    print()
    print()
    print("=" * 65)
    print("  加窗与频谱泄漏")
    print("=" * 65)
    print()

    N = 64
    sample_rate = 64
    freq = 7.5

    signal = [math.sin(2 * math.pi * freq * n / sample_rate) for n in range(N)]

    X_rect = fft(signal)
    mags_rect = magnitude_spectrum(X_rect)

    hann = hann_window(N)
    signal_hann = apply_window(signal, hann)
    X_hann = fft(signal_hann)
    mags_hann = magnitude_spectrum(X_hann)

    hamm = hamming_window(N)
    signal_hamm = apply_window(signal, hamm)
    X_hamm = fft(signal_hamm)
    mags_hamm = magnitude_spectrum(X_hamm)

    print(f"  信号: sin(2*pi*{freq}*t) -- 频率位于两个频率桶之间")
    print(f"  N = {N}，采样率 = {sample_rate} Hz")
    print(f"  频率分辨率: 每个频率桶 {sample_rate / N:.2f} Hz")
    print(f"  {freq} Hz 位于第 7 个和第 8 个频率桶之间")
    print()

    print(f"  {'频率 (Hz)':>10s} {'不加窗':>12s} {'Hann':>12s} {'Hamming':>12s}")
    print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12}")

    for k in range(N // 2 + 1):
        f_hz = k * sample_rate / N
        if mags_rect[k] > 0.5 or (5 <= f_hz <= 11):
            print(f"  {f_hz:>10.1f} {mags_rect[k]:>12.4f} "
                  f"{mags_hann[k]:>12.4f} {mags_hamm[k]:>12.4f}")

    print()
    print("  不加窗时，能量会泄漏到相邻的频率桶中。")
    print("  Hann 和 Hamming 窗将能量集中在真实频率附近。")
    print("  权衡: 加窗会展宽主峰，但能抑制旁瓣。")


def demo_parseval():
    print()
    print()
    print("=" * 65)
    print("  帕塞瓦尔定理: 能量守恒")
    print("=" * 65)
    print()

    import random
    random.seed(7)
    N = 32
    signal = [random.gauss(0, 1) for _ in range(N)]

    time_energy = sum(s ** 2 for s in signal)

    X = fft(signal)
    freq_energy = sum(xk.real ** 2 + xk.imag ** 2 for xk in X) / N

    print(f"  信号: {N} 个随机采样点")
    print(f"  时域能量:  sum |x[n]|^2 = {time_energy:.6f}")
    print(f"  频域能量:  (1/N) sum |X[k]|^2 = {freq_energy:.6f}")
    print(f"  差异: {abs(time_energy - freq_energy):.2e}")
    print(f"  能量是否守恒: {abs(time_energy - freq_energy) < 1e-10}")


def demo_positional_encoding():
    print()
    print()
    print("=" * 65)
    print("  位置编码的频率")
    print("=" * 65)
    print()

    d_model = 16
    max_pos = 8

    print(f"  d_model = {d_model}，位置 0-{max_pos - 1}")
    print()

    print(f"  每对维度的频率:")
    for i in range(d_model // 2):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        wavelength = 2 * math.pi / freq if freq > 0 else float('inf')
        print(f"    维度 ({2 * i:>2d},{2 * i + 1:>2d}): freq = {freq:.8f}  "
              f"wavelength = {wavelength:.1f}")

    print()
    print(f"  位置编码之间的点积:")
    print(f"  （仅依赖于距离，不依赖于绝对位置）")
    print()

    print(f"  {'pos_i':>6s} {'pos_j':>6s} {'距离':>6s} {'点积':>12s}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 12}")

    pairs = [(0, 0), (0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 4), (3, 7)]
    for p1, p2 in pairs:
        pe1 = positional_encoding(p1, d_model)
        pe2 = positional_encoding(p2, d_model)
        dot = sum(a * b for a, b in zip(pe1, pe2))
        print(f"  {p1:>6d} {p2:>6d} {abs(p2 - p1):>6d} {dot:>12.4f}")

    print()
    print("  距离相同的位置对各具有相近的点积。")
    print("  这使得模型能够通过注意力机制学习相对位置。")


def demo_frequency_scaling():
    print()
    print()
    print("=" * 65)
    print("  FFT 复杂度随规模的变化")
    print("=" * 65)
    print()

    print(f"  {'N':>8s} {'DFT O(N^2)':>14s} {'FFT O(N logN)':>16s} {'加速比':>10s}")
    print(f"  {'-' * 8} {'-' * 14} {'-' * 16} {'-' * 10}")

    for exp in range(3, 14):
        N = 2 ** exp
        dft_ops = N * N
        fft_ops = int(N * math.log2(N))
        speedup = dft_ops / fft_ops
        print(f"  {N:>8d} {dft_ops:>14,d} {fft_ops:>16,d} {speedup:>10.1f}x")


PROMPT_SPECTRAL_ANALYZER_EN = """---
name: prompt-spectral-analyzer
description: Guides analysis of frequency content in signals using Fourier transform techniques
phase: 1
lesson: 20
---

You are a spectral analysis expert. You help engineers analyze the frequency content of signals using Fourier transform techniques.

When given a signal or signal description, guide the analysis step by step:

1. **Determine sampling parameters.**
   - What is the sampling rate (fs)? This sets the maximum detectable frequency (Nyquist = fs/2).
   - How many samples (N)? This sets the frequency resolution (delta_f = fs/N).
   - Is the signal length a power of 2? If not, recommend zero-padding for FFT efficiency.

2. **Choose a window function.**
   - Is the signal exactly periodic in the analysis window? If yes, no window needed.
   - For general analysis: use Hann window (good tradeoff between resolution and leakage).
   - For audio/speech: Hamming window.
   - When side lobe suppression matters most: Blackman window.
   - Remember: windowing widens peaks but reduces leakage.

3. **Compute and interpret the spectrum.**
   - Power spectrum |X[k]|^2 shows energy at each frequency.
   - Peaks in the power spectrum indicate dominant frequencies.
   - X[0] is the DC component (signal mean * N).
   - Only look at bins 0 to N/2 for real-valued signals (upper half is the mirror).
   - Frequency of bin k: f_k = k * fs / N.

4. **Identify dominant frequencies.**
   - Find peaks above a noise threshold.
   - Convert bin index to Hz: freq = k * fs / N.
   - Check for harmonics (peaks at integer multiples of a fundamental).
   - Check for aliased frequencies (apparent frequency = f_actual mod fs; if above fs/2, it folds to fs - f_apparent).

5. **Common pitfalls to watch for.**
   - Spectral leakage: non-integer number of cycles in the window causes energy to spread across bins.
   - Aliasing: if signal contains frequencies above fs/2, they fold back into the spectrum.
   - DC offset: large X[0] can mask nearby low-frequency content. Remove the mean before FFT.
   - Zero-padding increases bin density but does NOT improve actual frequency resolution.
   - Circular vs linear convolution: DFT gives circular convolution. Zero-pad for linear.

6. **For convolution analysis.**
   - Time-domain convolution = frequency-domain multiplication.
   - For large kernels, FFT-based convolution is faster: O(N log N) vs O(N*M).
   - Zero-pad both signals to length N + M - 1 for correct linear convolution.
"""


PROMPT_SPECTRAL_ANALYZER_ZH_CN = """---
name: prompt-spectral-analyzer
description: 指导使用傅里叶变换技术分析信号的频率成分
phase: 1
lesson: 20
---

你是一位频谱分析专家。你帮助工程师使用傅里叶变换技术分析信号的频率成分。

当给定一个信号或信号描述时，按以下步骤指导分析:

1. **确定采样参数。**
   - 采样率（fs）是多少？它决定了可检测的最大频率（奈奎斯特频率 = fs/2）。
   - 采样点数（N）是多少？它决定了频率分辨率（delta_f = fs/N）。
   - 信号长度是否为 2 的幂？如果不是，建议补零以提高 FFT 效率。

2. **选择窗函数。**
   - 信号在分析窗口内是否恰好为整数周期？如果是，则无需加窗。
   - 对于一般分析: 使用 Hann 窗（在分辨率和泄漏之间有较好的权衡）。
   - 对于音频/语音: 使用 Hamming 窗。
   - 当旁瓣抑制最为重要时: 使用 Blackman 窗。
   - 切记: 加窗会展宽峰值但能减少泄漏。

3. **计算并解释频谱。**
   - 功率谱 |X[k]|^2 显示每个频率上的能量。
   - 功率谱中的峰值指示主频率。
   - X[0] 是直流分量（信号均值 * N）。
   - 对于实值信号，只需查看第 0 到 N/2 个频率桶（上半部分是镜像）。
   - 第 k 个频率桶对应的频率: f_k = k * fs / N。

4. **识别主频率。**
   - 寻找高于噪声阈值的峰值。
   - 将频率桶索引转换为 Hz: freq = k * fs / N。
   - 检查谐波（基频整数倍处的峰值）。
   - 检查混叠频率（实际频率 = fs - 表观频率）。

5. **需注意的常见陷阱。**
   - 频谱泄漏: 窗口内周期数为非整数会导致能量分散到各频率桶中。
   - 混叠: 如果信号包含高于 fs/2 的频率，它们会折叠回频谱中。
   - 直流偏移: 过大的 X[0] 会掩盖附近的低频成分。在 FFT 前去除均值。
   - 补零会增加频率桶密度，但并不能真正提高频率分辨率。
   - 循环卷积与线性卷积: DFT 给出的是循环卷积。补零以获得线性卷积。

6. **关于卷积分析。**
   - 时域卷积 = 频域乘法。
   - 对于大卷积核，基于 FFT 的卷积更快: O(N log N) vs O(N*M)。
   - 将两个信号补零至长度 N + M - 1，以获得正确的线性卷积。
"""


def write_prompt_output(output_dir="outputs"):
    output_dir = Path(output_dir)
    canonical_path = output_dir / "prompt-spectral-analyzer.md"
    zh_cn_path = output_dir / "prompt-spectral-analyzer.zh-CN.md"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not canonical_path.exists():
            canonical_path.write_text(PROMPT_SPECTRAL_ANALYZER_EN, encoding="utf-8")
        zh_cn_path.write_text(PROMPT_SPECTRAL_ANALYZER_ZH_CN, encoding="utf-8")
        print(f"\n  提示词输出已写入 {canonical_path} 和 {zh_cn_path}")
    except OSError:
        print("\n  无法写入提示词输出（请在本课程目录下运行）")


def print_summary():
    print()
    print()
    print("=" * 65)
    print("  总结")
    print("=" * 65)
    print()
    print("  1. DFT 将 N 个时域采样点转换为 N 个频域系数。")
    print("  2. 每个 X[k] 度量信号与频率 k 的相关性。")
    print("  3. FFT 以 O(N log N) 而非 O(N^2) 计算 DFT。")
    print("  4. DFT 与 IDFT 互为完美逆变换——不会丢失任何信息。")
    print("  5. 卷积定理: 时域卷积 = 频域乘法。")
    print("     这正是基于 FFT 的卷积速度快的原因。")
    print("  6. 加窗可减少非周期信号的频谱泄漏。")
    print("  7. 帕塞瓦尔定理: 变换过程中能量守恒。")
    print("  8. Transformer 的位置编码使用了相同的频率分解思想")
    print("     ——每个位置都拥有独特的频谱。")
    print()


if __name__ == "__main__":
    demo_pure_sine()
    demo_multi_frequency()
    demo_fft_vs_dft()
    demo_reconstruction()
    demo_convolution_theorem()
    demo_windowing()
    demo_parseval()
    demo_positional_encoding()
    demo_frequency_scaling()
    write_prompt_output()
    print_summary()
