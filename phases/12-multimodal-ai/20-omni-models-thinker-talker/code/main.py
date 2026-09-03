"""Thinker-Talker 流式管线——TTFAB 计算器与 VAD 回合切换。

标准库。无音频处理；重点关注延迟预算以及 Thinker（文本）与 Talker（语音）之间并行流式的并发。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamConfig:
    thinker_b: int
    talker_m: int
    mic_sr: int = 16000
    include_vision: bool = False


@dataclass
class LatencyComponent:
    name: str
    ms: float


def ttfab(cfg: StreamConfig) -> list[LatencyComponent]:
    components = []
    mic_ms = 40 + (cfg.mic_sr // 8000) * 5
    components.append(LatencyComponent("麦克风 -> 语音 token", mic_ms))

    prefill = 100 * (cfg.thinker_b / 7.0)
    if cfg.include_vision:
        prefill += 80
    components.append(LatencyComponent("Thinker 预填充（提示 + 历史）", prefill))

    first_text = 40 * (cfg.thinker_b / 7.0)
    components.append(LatencyComponent("Thinker 首个文本 token", first_text))

    talker_first = max(15, 20 * (cfg.talker_m / 300.0))
    components.append(LatencyComponent("Talker 首批语音 token", talker_first))

    rvq_decode = 30
    components.append(LatencyComponent("残差 VQ 解码（8 层并行）", rvq_decode))

    wave_decode = 70
    components.append(LatencyComponent("波形解码器（SNAC 级）", wave_decode))
    return components


def print_ttfab(cfg: StreamConfig) -> float:
    vision = "是" if cfg.include_vision else "否"
    print(f"\n配置：Thinker={cfg.thinker_b}B  Talker={cfg.talker_m}M  "
          f"麦克风={cfg.mic_sr}Hz  视觉={vision}")
    print("-" * 60)
    total = 0.0
    for c in ttfab(cfg):
        total += c.ms
        print(f"  {c.name:<40}  +{c.ms:>5.0f} ms  ({total:>6.0f})")
    print(f"  TTFAB = {total:.0f} ms", end=" ")
    if total < 250:
        print("  -> GPT-4o 等级")
    elif total < 400:
        print("  -> 对话级")
    elif total < 700:
        print("  -> 可感知但可用")
    else:
        print("  -> 迟钝，用户注意力分散")
    return total


@dataclass
class VADEvent:
    time_ms: float
    kind: str


def simulate_turn_taking(silence_threshold_ms: int = 200) -> list[VADEvent]:
    """模拟通过静音检测到的用户回合结束。"""
    events = []
    events.append(VADEvent(0, "用户开始说话"))
    events.append(VADEvent(450, "用户音频 token 流式输入"))
    events.append(VADEvent(3800, "用户停止说话"))
    events.append(VADEvent(3800 + silence_threshold_ms, "VAD 触发回合结束"))
    events.append(VADEvent(3800 + silence_threshold_ms + 200, "Thinker 开始预填充"))
    events.append(VADEvent(3800 + silence_threshold_ms + 400, "Talker 输出首段音频"))
    return events


def demo_vad() -> None:
    print("\n半双工回合切换（VAD 静音 200ms）")
    print("-" * 60)
    for e in simulate_turn_taking(200):
        print(f"  t={e.time_ms:>6.0f} ms  {e.kind}")
    print("  用户停止后的净响应延迟：~400ms")


def duplex_modes() -> None:
    print("\n双工模式")
    print("-" * 60)
    modes = [
        ("半双工",  "用户说、模型听，然后交换；回合边界清晰"),
        ("回合切换", "VAD 通过 200-400ms 静音检测回合结束"),
        ("全双工",  "双方可同时说话；需要训练与反馈信号数据"),
    ]
    for mode, note in modes:
        print(f"  {mode:<14}: {note}")


def main() -> None:
    print("=" * 60)
    print("OMNI THINKER-TALKER 流式处理（第12阶段，第20课）")
    print("=" * 60)

    configs = [
        StreamConfig(thinker_b=7,  talker_m=200,  include_vision=False),
        StreamConfig(thinker_b=7,  talker_m=300,  include_vision=True),
        StreamConfig(thinker_b=72, talker_m=300,  include_vision=True),
        StreamConfig(thinker_b=70, talker_m=1000, include_vision=True),
    ]
    for c in configs:
        print_ttfab(c)

    demo_vad()
    duplex_modes()

    print("\n开放流式设计")
    print("-" * 60)
    designs = [
        ("Mini-Omni (2024)",  "首个开放流式模型，文本与语音交错"),
        ("Moshi (2024)",      "单 Transformer 内心独白，TTFAB 为 160ms"),
        ("Qwen2.5-Omni (3/25)", "Thinker-Talker 拆分 + TMRoPE，TTFAB 约 350ms"),
        ("Qwen3-Omni (11/25)", "扩展 Qwen3 基座，延迟接近 GPT-4o"),
    ]
    for name, note in designs:
        print(f"  {name:<22}: {note}")


if __name__ == "__main__":
    main()
