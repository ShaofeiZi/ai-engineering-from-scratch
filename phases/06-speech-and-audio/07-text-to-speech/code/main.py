"""TTS 内部原理演示：音素查找 + 时长估计 + Mel 帧调度。

仅使用标准库。构建简化的英语字素到音素表、估算时长，并输出
FastSpeech 风格模型使用的帧调度。实际合成请安装 kokoro 或 f5-tts
（参见文档）。

运行：python3 code/main.py
"""

import math
import random


# 最小字素到音素表：每个常见字素簇对应一项映射。
# 真正的系统使用 espeak-ng 或 g2p-en（CMU 词典）——此处仅为简化演示。
G2P = {
    " ":    ["_"],
    "a":    ["AH"],  "b":    ["B"],  "c":    ["K"],  "d":    ["D"],  "e":    ["EH"],
    "f":    ["F"],   "g":    ["G"],  "h":    ["HH"], "i":    ["IH"], "j":    ["JH"],
    "k":    ["K"],   "l":    ["L"],  "m":    ["M"],  "n":    ["N"],  "o":    ["AO"],
    "p":    ["P"],   "q":    ["K"],  "r":    ["R"],  "s":    ["S"],  "t":    ["T"],
    "u":    ["UH"],  "v":    ["V"],  "w":    ["W"],  "x":    ["K", "S"],
    "y":    ["Y"],   "z":    ["Z"],
    "the":  ["DH", "AH"],
    "ing":  ["IH", "NG"],
    "er":   ["ER"],
    "sh":   ["SH"],
    "ch":   ["CH"],
    "th":   ["TH"],
    "ee":   ["IY"],
    "oo":   ["UW"],
    "ow":   ["AW"],
    "ay":   ["EY"],
    ".":    ["_PAUSE_"],
    ",":    ["_SHORT_"],
    "?":    ["_PAUSE_"],
    "!":    ["_PAUSE_"],
}

# 典型时长（帧，步长为 12.5 ms）；与 FastSpeech 统计值大致一致
DURATION_FRAMES = {
    "AA": 9, "AE": 7, "AH": 6, "AO": 8, "AW": 9, "AY": 8, "B": 4, "CH": 6,
    "D": 4, "DH": 5, "EH": 6, "ER": 7, "EY": 8, "F": 6, "G": 5, "HH": 4,
    "IH": 5, "IY": 7, "JH": 6, "K": 5, "L": 5, "M": 5, "N": 5, "NG": 6,
    "OW": 8, "OY": 9, "P": 5, "R": 5, "S": 6, "SH": 7, "T": 4, "TH": 5,
    "UH": 6, "UW": 8, "V": 5, "W": 5, "Y": 5, "Z": 6, "ZH": 7,
    "_": 3,           # 单词边界
    "_SHORT_": 6,     # 逗号停顿
    "_PAUSE_": 12,    # 句子停顿
}


def phonemize(text):
    text = text.lower()
    phones = []
    i = 0
    while i < len(text):
        matched = False
        for length in (3, 2, 1):
            if i + length <= len(text):
                chunk = text[i : i + length]
                if chunk in G2P:
                    phones.extend(G2P[chunk])
                    i += length
                    matched = True
                    break
        if not matched:
            i += 1
    return phones


def duration(phones, jitter=0.1, seed=0):
    random.seed(seed)
    out = []
    for p in phones:
        base = DURATION_FRAMES.get(p, 5)
        noise = int(round(base * random.uniform(-jitter, jitter)))
        out.append(max(1, base + noise))
    return out


def mel_schedule(phones, durs, hop_ms=12.5):
    schedule = []
    t = 0.0
    for p, d in zip(phones, durs):
        schedule.append((p, t, t + d * hop_ms))
        t += d * hop_ms
    return schedule, t


def main():
    text = "Please remind me to water the plants at 6 pm."
    print("=== 步骤 1：字素转音素 ===")
    print(f"  文本：{text!r}")
    phones = phonemize(text)
    print(f"  音素（{len(phones)} 个）：{' '.join(phones[:20])}{'...' if len(phones) > 20 else ''}")

    print()
    print("=== 步骤 2：估算各音素时长 ===")
    durs = duration(phones, jitter=0.1, seed=42)
    print(f"  时长（帧）：{durs[:20]}{'...' if len(durs) > 20 else ''}")

    print()
    print("=== 步骤 3：Mel 帧调度（步长 12.5 ms）===")
    sched, total_ms = mel_schedule(phones, durs)
    print(f"  总时长：{total_ms:.1f} ms（{total_ms / 1000:.2f} s）")
    print(f"  前 10 帧：")
    for p, s, e in sched[:10]:
        print(f"    {p:<10} {s:6.1f} – {e:6.1f} ms")

    print()
    print("=== 步骤 4：发送给声码器的 Mel 总帧数 ===")
    total_frames = sum(durs)
    audio_samples = total_frames * 300  # 12.5 ms @ 24 kHz = 300 个样本
    print(f"  Mel 帧数：{total_frames}  24 kHz 音频样本数：{audio_samples}")
    print(f"  流水线内存预算：{total_frames * 80 * 4 / 1024:.1f} KB（Mel，float32）")

    print()
    print("=== 步骤 5：2026 年 TTS 质量榜（UTMOS / CER / 大小）===")
    table = [
        ("真实录音",       4.08, 1.2,   "—"),
        ("F5-TTS",         3.95, 2.1,   "335M"),
        ("Kokoro v0.19",   3.87, 1.8,   "82M"),
        ("XTTS v2",        3.81, 3.5,   "470M"),
        ("Parler-TTS L",   3.76, 2.8,   "2.3B"),
        ("VITS",           3.62, 3.1,   "25M"),
    ]
    print("  | 模型              | UTMOS | CER%  | 大小 |")
    for name, u, c, s in table:
        print(f"  | {name:<17} | {u:.2f}  | {c:.1f}   | {s:<4} |")


if __name__ == "__main__":
    main()
