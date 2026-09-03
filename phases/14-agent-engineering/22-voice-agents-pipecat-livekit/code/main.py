"""仿 Pipecat 的玩具语音管道：VAD  STT  LLM  TTS  传输。

数据帧沿 DOWNSTREAM（从源到汇）和 UPSTREAM（cancel/control）方向流动。
一段脚本化输入展示了正常流程，以及一个 barge-in 取消操作会停止 TTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Frame:
    kind: str
    payload: Any
    direction: str = "downstream"


class Processor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.next: Processor | None = None
        self.prev: Processor | None = None
        self.trace: list[str] = []

    def process(self, frame: Frame) -> None:
        self.trace.append(f"{self.name} saw {frame.kind}")
        if self.next is not None and frame.direction == "downstream":
            self.next.process(frame)
        elif self.prev is not None and frame.direction == "upstream":
            self.prev.process(frame)


class VAD(Processor):
    def process(self, frame: Frame) -> None:
        if frame.kind == "audio_chunk":
            is_speech = bool(frame.payload)
            self.trace.append(f"VAD: speech={is_speech}")
            if is_speech:
                super().process(Frame("vad_speech", frame.payload))
        else:
            super().process(frame)


class STT(Processor):
    def process(self, frame: Frame) -> None:
        if frame.kind == "vad_speech":
            transcript = str(frame.payload)
            self.trace.append(f"STT: -> {transcript!r}")
            super().process(Frame("transcript", transcript))
        else:
            super().process(frame)


class LLM(Processor):
    def __init__(self, name: str, replies: dict[str, str]) -> None:
        super().__init__(name)
        self.replies = replies

    def process(self, frame: Frame) -> None:
        if frame.kind == "cancel":
            self.trace.append("LLM: cancelled")
            super().process(frame)
            return
        if frame.kind == "transcript":
            text = str(frame.payload)
            reply = self.replies.get(text, "[no canned reply]")
            self.trace.append(f"LLM: {text!r}  -> {reply!r}")
            super().process(Frame("text", reply))
        else:
            super().process(frame)


class TTS(Processor):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cancelled = False

    def process(self, frame: Frame) -> None:
        if frame.kind == "cancel":
            self.cancelled = True
            self.trace.append("TTS: cancel received; drop pending audio")
            super().process(frame)
            return
        if frame.kind == "text":
            self.cancelled = False
            words = str(frame.payload).split()
            emitted: list[str] = []
            for w in words:
                if self.cancelled:
                    self.trace.append(f"TTS: cut mid-word after {emitted}")
                    break
                emitted.append(w)
            self.trace.append(f"TTS: emitted {emitted}")
            super().process(Frame("tts_audio", emitted))
        else:
            super().process(frame)


class Transport(Processor):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.delivered: list[list[str]] = []

    def process(self, frame: Frame) -> None:
        if frame.kind == "tts_audio":
            self.delivered.append(list(frame.payload))
            self.trace.append(f"transport: sent {len(frame.payload)} words")
        else:
            super().process(frame)


def link(*processors: Processor) -> None:
    for a, b in zip(processors, processors[1:]):
        a.next = b
        b.prev = a


def main() -> None:
    print("=" * 70)
    print("语音管道（PIPECAT 风格）— 第 14 阶段，第 22 课")
    print("=" * 70)

    vad = VAD("vad")
    stt = STT("stt")
    llm = LLM("llm", replies={
        "hello": "hi there, how can I help today?",
        "refund please": (
            "sure, I can help with a refund; what order number should I look up?"
        ),
    })
    tts = TTS("tts")
    transport = Transport("transport")
    link(vad, stt, llm, tts, transport)

    print("\n场景 1：正常流程")
    vad.process(Frame("audio_chunk", "hello"))
    print(f"  传输已交付：{transport.delivered[-1]}")

    print("\n场景 2：在话语中途插话")
    tts.cancelled = False
    vad.process(Frame("audio_chunk", "refund please"))
    transport.process(Frame("cancel", None, direction="upstream"))

    print("  跨管道追踪")
    for proc in (vad, stt, llm, tts, transport):
        for line in proc.trace:
            print(f"    {proc.name}：{line}")

    print()
    print("插话需要将取消帧向上游传播回 TTS 和 LLM。")
    print("逐阶段累加延迟；高级栈最终达到 450–600 毫秒端到端延迟。")


if __name__ == "__main__":
    main()
