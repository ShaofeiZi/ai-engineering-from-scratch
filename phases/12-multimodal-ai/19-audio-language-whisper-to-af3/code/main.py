"""Audio-LLM 玩具：log-Mel 频谱图、音频 Q-former 与级联/端到端对比。

标准库实现。从合成波形计算基于朴素 DFT 的 log-Mel 频谱，
对结果帧运行玩具 Q-former，并比较级联与端到端流水线的任务覆盖范围。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

random.seed(6)


def synth_waveform(duration_s: float = 1.0, sr: int = 16000) -> list[float]:
    n = int(duration_s * sr)
    freq = 440
    return [0.5 * math.sin(2 * math.pi * freq * i / sr) +
            0.2 * math.sin(2 * math.pi * 880 * i / sr)
            for i in range(n)]


def window_frames(x: list[float], sr: int, win_ms: int = 25, hop_ms: int = 10) -> list[list[float]]:
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames = []
    i = 0
    while i + win <= len(x):
        frames.append(x[i:i + win])
        i += hop
    return frames


def naive_dft_mag(frame: list[float], n_bins: int = 64) -> list[float]:
    """在 n_bins 个频率处使用朴素 DFT 计算幅度谱。"""
    n = len(frame)
    out = []
    for k in range(n_bins):
        re = 0.0
        im = 0.0
        for i, x in enumerate(frame):
            angle = -2 * math.pi * k * i / n
            re += x * math.cos(angle)
            im += x * math.sin(angle)
        out.append(math.sqrt(re * re + im * im))
    return out


def mel_filterbank(n_bins: int = 64, n_mels: int = 20) -> list[list[float]]:
    """三角梅尔滤波器组（简化版，以线性变换作为代理）。"""
    fbank = []
    band = n_bins // n_mels
    for m in range(n_mels):
        row = [0.0] * n_bins
        start = m * band
        end = min(start + band, n_bins)
        for k in range(start, end):
            row[k] = 1.0 / (end - start)
        fbank.append(row)
    return fbank


def apply_mel(spec_mag: list[float], fbank: list[list[float]]) -> list[float]:
    return [sum(w * s for w, s in zip(row, spec_mag)) for row in fbank]


def log_compress(xs: list[float]) -> list[float]:
    return [math.log(1 + x) for x in xs]


def demo_melspec() -> None:
    print("\nLOG-MEL SPECTROGRAM（1s @ 16kHz，25ms 窗，10ms 步长，20 个梅尔 bins）")
    print("-" * 60)
    wave = synth_waveform(1.0, 16000)
    frames = window_frames(wave, 16000, 25, 10)
    print(f"  帧数 : {len(frames)}（1s 时应约为 99）")

    spec = naive_dft_mag(frames[0], n_bins=64)
    fbank = mel_filterbank(n_bins=64, n_mels=20)
    mel = apply_mel(spec, fbank)
    log_mel = log_compress(mel)
    print(f"  每帧梅尔维度：{len(mel)}")
    print(f"  第一帧 log-Mel（已四舍五入）："
          f"{[round(v, 2) for v in log_mel[:10]]}...")


@dataclass
class QFormer:
    n_queries: int
    hidden: int

    def __post_init__(self):
        self.queries = [[random.gauss(0, 0.1) for _ in range(self.hidden)]
                        for _ in range(self.n_queries)]

    def forward(self, frames: list[list[float]]) -> list[list[float]]:
        """朴素 cross-attention：每个查询对所有帧进行注意力计算。"""
        out = []
        for q in self.queries:
            scores = [sum(qi * fi for qi, fi in zip(q, f)) for f in frames]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            z = sum(exps)
            weights = [e / z for e in exps]
            agg = [sum(w * f[k] for w, f in zip(weights, frames))
                   for k in range(self.hidden)]
            out.append(agg)
        return out


def demo_qformer() -> None:
    print("\nAUDIO Q-FORMER（N=8 个查询，对 20 维帧）")
    print("-" * 60)
    frames = [[random.gauss(0, 1) for _ in range(20)] for _ in range(99)]
    qf = QFormer(n_queries=8, hidden=20)
    tokens = qf.forward(frames)
    print(f"  输入帧: {len(frames)}")
    print(f"  输出 token: {len(tokens)} 个，维度为 {len(tokens[0])}")
    print("  每个 token 通过软注意力权重对完整音频进行注意力计算")


def task_coverage_table() -> None:
    print("\n级联（Whisper -> LLM）与端到端 AUDIO-LLM 对比")
    print("-" * 60)
    tasks = [
        ("转录",             "是",   "是"),
        ("关键词提取",       "是",   "是"),
        ("摘要",             "是",   "是"),
        ("说话人分离",       "部分", "是"),
        ("情感推断",         "否",   "是"),
        ("音乐流派分类",     "否",   "是"),
        ("乐器识别",         "否",   "是"),
        ("环境声音识别",     "否",   "是"),
        ("时序事件定位",     "部分", "是"),
        ("深度伪造检测",     "否",   "是"),
    ]
    print(f"  {'任务':<30}{'级联':<14}{'端到端'}")
    for name, cas, e2e in tasks:
        print(f"  {name:<30}{cas:<14}{e2e}")
    print("\n  级联：快速且可靠，适用于可提取文本的信号")
    print("  端到端：纯声学信号所需（约占 MMAU 的 40%）")


def main() -> None:
    print("=" * 60)
    print("音频语言：从 WHISPER 到 AF3（阶段 12，第 19 课）")
    print("=" * 60)

    demo_melspec()
    demo_qformer()
    task_coverage_table()

    print("\n2026 年配方")
    print("-" * 60)
    print("  编码器 : AF-Whisper + BEATs 拼接")
    print("  桥接  : 64 查询 Q-former")
    print("  LLM     : Qwen2.5-7B 携带音频 token")
    print("  训练：AudioCaps + Clotho + MMAU 风格指令")
    print("  选项：按需思维用于复杂推理")


if __name__ == "__main__":
    main()
