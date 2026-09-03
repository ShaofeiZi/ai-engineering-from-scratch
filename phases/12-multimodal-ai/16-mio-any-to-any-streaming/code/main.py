"""MIO 风格四模态分词器分配与流式解码延迟计算。

标准库实现。打印词表布局，并输出逐步延迟追踪，
对应一次语音对话请求，其中 MIO 消费语音并生成语音。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VocabSlot:
    name: str
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size


def build_vocab() -> list[VocabSlot]:
    slots = []
    cursor = 0
    plan = [
        ("text BPE",      32000),
        ("image SEED",     4096),
        ("speech L0",      4096),
        ("speech L1..L7", 4096),
        ("music",          8192),
        ("<image>",           1),
        ("</image>",          1),
        ("<speech>",          1),
        ("</speech>",         1),
        ("<music>",           1),
        ("</music>",          1),
    ]
    for name, size in plan:
        slots.append(VocabSlot(name=name, start=cursor, size=size))
        cursor += size
    return slots


def print_vocab(slots: list[VocabSlot]) -> None:
    print("\n共享词表布局")
    print("-" * 60)
    print(f"  {'槽位':<18}{'起点':>8}{'终点':>8}{'大小':>8}")
    slot_names = {
        "text BPE": "文本 BPE", "image SEED": "图像 SEED",
        "speech L0": "语音 L0", "speech L1..L7": "语音 L1..L7",
        "music": "音乐",
    }
    for s in slots:
        print(f"  {slot_names.get(s.name, s.name):<18}{s.start:>8}{s.end:>8}{s.size:>8}")
    total = slots[-1].end
    print(f"  {'合计':<18}{total:>8}{'（词表大小）':>16}")


def route_inputs(inputs: list[dict]) -> list[dict]:
    """对每个输入进行分类并分配分词器路径。"""
    routed = []
    for inp in inputs:
        kind = inp["kind"]
        if kind == "text":
            path = "BPE"
        elif kind == "image":
            path = "SEED-Tokenizer"
        elif kind in ("speech", "voice"):
            path = "SpeechTokenizer residual-VQ"
        elif kind == "music":
            path = "Encodec"
        else:
            path = "UNKNOWN"
        routed.append({**inp, "path": path})
    return routed


@dataclass
class LatencyTrace:
    label: str
    ms: float


def streaming_decode_latency(
    prompt_audio_seconds: float = 2.0,
    model_size_b: int = 8,
) -> list[LatencyTrace]:
    trace = []
    trace.append(LatencyTrace("麦克风音频 -> 语音 token",
                              prompt_audio_seconds * 20))
    trace.append(LatencyTrace("预填充提示 token",
                              80 * (model_size_b / 8.0)))
    trace.append(LatencyTrace("首个输出 token",
                              40 * (model_size_b / 8.0)))
    trace.append(LatencyTrace("残差 VQ 第 1..7 层",
                              30))
    trace.append(LatencyTrace("语音解码器（类 Encodec）",
                              80))
    return trace


def print_trace(trace: list[LatencyTrace]) -> None:
    print("\n流式解码延迟（首音频字节延迟）")
    print("-" * 60)
    total = 0.0
    for t in trace:
        total += t.ms
        print(f"  {t.label:<38}  +{t.ms:>5.0f} 毫秒   (累计 {total:>6.0f})")
    print("-" * 60)
    print(f"  总计 TTFAB: {total:.0f} 毫秒")
    if total < 500:
        print("  -> 对话感自然（GPT-4o 级）")
    elif total < 800:
        print("  -> 可接受（第一代任意模态到任意模态）")
    else:
        print(f"  -> 响应迟缓，考虑使用更小的模型或并行解码")


def demo_chain_of_visual_thought() -> None:
    print("\n视觉思维链（MIO）")
    print("-" * 60)
    prompt = "这张照片里的猫正在爬树吗？"
    steps = [
        "用户文本 -> 视觉 token",
        "模型绘制中间图像 <image> ... </image>",
        "模型输出对草图的文本分析",
        "模型给出是/否结论与理由",
    ]
    print(f"  提示词: {prompt}")
    for i, s in enumerate(steps, 1):
        print(f"    步骤 {i}: {s}")
    print("  在空间推理基准测试上胜出，但会增加延迟。")


def main() -> None:
    print("=" * 60)
    print("MIO 任意模态到任意模态流式处理（第12阶段，第16课）")
    print("=" * 60)

    vocab = build_vocab()
    print_vocab(vocab)

    print("\n路由器：四个输入 -> 四个分词器")
    print("-" * 60)
    inputs = [
        {"kind": "text",   "payload": "你好"},
        {"kind": "image",  "payload": "cat.png"},
        {"kind": "voice",  "payload": "user.wav"},
        {"kind": "music",  "payload": "loop.mp3"},
    ]
    kind_names = {"text": "文本", "image": "图像", "voice": "语音", "music": "音乐"}
    for r in route_inputs(inputs):
        print(f"  {kind_names[r['kind']]:<8}  '{r['payload']}'  -> {r['path']}")

    trace = streaming_decode_latency(prompt_audio_seconds=2.0, model_size_b=8)
    print_trace(trace)

    demo_chain_of_visual_thought()


if __name__ == "__main__":
    main()
