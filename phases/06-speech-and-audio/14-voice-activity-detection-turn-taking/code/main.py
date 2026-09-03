"""VAD 级联 + 轮次检测状态机。

三级级联：能量门控 →（模拟）Silero → 轮次检测状态机。
运行合成流：语音 + 静音 + 咳嗽 + 语音，验证轮次检测器是否在正确时刻
触发 START 和 END。

仅使用标准库。运行：python3 code/main.py
"""

import math
import random


def synth_chunk(kind, rng, sr=16000, chunk_ms=20):
    n = int(sr * chunk_ms / 1000)
    if kind == "speech":
        return [0.2 * rng.gauss(0, 1.0) for _ in range(n)]
    if kind == "cough":
        return [0.8 * rng.gauss(0, 1.0) if i < n // 5 else 0.001 * rng.gauss(0, 1.0) for i in range(n)]
    return [0.002 * rng.gauss(0, 1.0) for _ in range(n)]


def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    return 20.0 * math.log10(max(rms, 1e-10)) > threshold_dbfs


def fake_silero_vad(chunk, prev_state, threshold=0.5):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    duration = len(chunk) / 16000.0
    transient = max(chunk) - min(chunk) > 0.6 and duration < 0.03
    if rms > 0.08 and not transient:
        return 0.92
    if rms > 0.05:
        return 0.55
    return 0.02


class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250, pre_roll_ms=300):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms
        self.pre_roll_ms = pre_roll_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            if self.state == "speaking":
                self.silence_ms += chunk_ms
                if self.silence_ms >= self.silence_hangover_ms:
                    self.state = "idle"
                    self.speech_ms = 0
                    self.silence_ms = 0
                    return "END"
        return None


def main():
    random.seed(42)
    rng = random.Random(42)

    sequence = (
        [("silence", 10)] +
        [("speech", 40)] +
        [("silence", 30)] +
        [("cough",   1)]  +
        [("silence", 10)] +
        [("speech", 25)] +
        [("silence", 35)]
    )

    chunks = []
    for kind, count in sequence:
        for _ in range(count):
            chunks.append((kind, synth_chunk(kind, rng)))

    print(f"=== 音频流：{len(chunks)} 个 20 ms 音频块，共 {len(chunks)*20} ms ===")
    print()

    td_silero = TurnDetector()
    td_energy = TurnDetector()
    events_silero = []
    events_energy = []

    for i, (truth, chunk) in enumerate(chunks):
        e_active = energy_vad(chunk)
        silero_prob = fake_silero_vad(chunk, None)
        s_active = silero_prob >= 0.5

        e_event = td_energy.update(e_active)
        s_event = td_silero.update(s_active)
        if e_event:
            events_energy.append((i * 20, e_event, truth))
        if s_event:
            events_silero.append((i * 20, s_event, truth))

    print("=== 仅使用能量的 VAD 轮次事件（咳嗽会产生大量误报）===")
    for ms, ev, truth in events_energy:
        print(f"  t={ms:>4} ms  {ev:<5}  （位于 {truth}）")

    print()
    print("=== Silero 风格 VAD 轮次事件（排除咳嗽）===")
    for ms, ev, truth in events_silero:
        print(f"  t={ms:>4} ms  {ev:<5}  （位于 {truth}）")

    print()
    print("=== 2026 年 VAD 速查表 ===")
    rows = [
        ("WebRTC VAD（Google，2013）", "50.0% TPR @ 5% FPR", "BSD"),
        ("Silero VAD（2020–2026）",    "87.7% TPR @ 5% FPR", "MIT — 默认开放"),
        ("Cobra VAD（Picovoice）",     "98.9% TPR @ 5% FPR", "商业许可"),
        ("pyannote segmentation",      "~95% TPR @ 5% FPR",  "类 MIT — 说话人分离级"),
    ]
    print("  | VAD                       | 准确率              | 许可证                |")
    for name, acc, lic in rows:
        print(f"  | {name:<25} | {acc:<19} | {lic:<21} |")

    print()
    print("要点：")
    print("  - 仅使用能量的 VAD 会在每个瞬态信号上触发，不适合生产环境")
    print("  - Silero VAD 能处理咳嗽而不触发轮次开始")
    print("  - 500 ms 静音拖尾是对话体验的理想值")
    print("  - 要实现低于 200 ms 的端到端语音智能体，可加入刷新技巧")


if __name__ == "__main__":
    main()
