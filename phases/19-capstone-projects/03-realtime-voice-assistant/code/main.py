"""实时语音流水线——VAD + 轮次检测 + 插话调度器。

2026 年语音智能体最关键的架构原语并非 ASR 或 TTS，而是以有界延迟协调 VAD
事件、ASR 部分结果、轮次完成分数、LLM 流、TTS 流和用户插话的流式调度器。
此脚手架模拟音频帧并完整实现调度器：状态机、插话取消、注入填充语的工具
侧信道，以及延迟统计。

运行：python main.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# 帧流——模拟的 20ms 音频帧
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    t_ms: int              # 自会话开始以来的时间戳（ms）
    is_speech: bool        # VAD 判定（Silero v5 替代实现）
    partial: str = ""      # ASR 累积部分结果（Deepgram Nova-3 替代实现）


def synth_call(script: str, start_ms: int = 0, noise: float = 0.0) -> list[Frame]:
    """为模拟来电者话语生成帧流。"""
    words = script.split()
    frames: list[Frame] = []
    t = start_ms
    # 说话前静音 120ms
    for _ in range(6):
        frames.append(Frame(t_ms=t, is_speech=random.random() < noise))
        t += 20
    partial = ""
    for w in words:
        partial = (partial + " " + w).strip()
        # 每个单词对应约 320ms 语音
        for _ in range(16):
            frames.append(Frame(t_ms=t, is_speech=True, partial=partial))
            t += 20
    # 尾部静音 2200ms（足以覆盖工具 + LLM + TTS）
    for _ in range(110):
        frames.append(Frame(t_ms=t, is_speech=False, partial=partial))
        t += 20
    return frames


# ---------------------------------------------------------------------------
# 轮次检测器——结合 VAD 静音时长与完成分数
# ---------------------------------------------------------------------------

def turn_completion_score(partial: str) -> float:
    """LiveKit 轮次检测模型的小型替代实现。"""
    if not partial:
        return 0.0
    if partial.rstrip().endswith(("?", ".", "!")):
        return 0.95
    # 启发式规则：单词越多，轮次已结束的置信度越高
    n = len(partial.split())
    if n < 3:
        return 0.2
    if n < 6:
        return 0.55
    return 0.75


# ---------------------------------------------------------------------------
# 状态机——IDLE -> LISTENING -> THINKING -> SPEAKING ->（插话）
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    LISTENING = auto()   # 用户正在说话
    WAITING = auto()     # VAD 判定静音，正在检查轮次分数
    THINKING = auto()    # LLM 正在流式输出，但 TTS 尚未开始
    SPEAKING = auto()    # TTS 正在流式输出
    TOOL = auto()        # 侧信道工具正在执行


@dataclass
class Metrics:
    events: list[str] = field(default_factory=list)
    turn_complete_ms: int = 0
    first_llm_token_ms: int = 0
    first_audio_out_ms: int = 0
    false_cutoffs: int = 0
    barge_ins: int = 0

    def log(self, msg: str) -> None:
        self.events.append(msg)

    def latency_ms(self) -> int:
        if self.turn_complete_ms and self.first_audio_out_ms:
            return self.first_audio_out_ms - self.turn_complete_ms
        return -1


# ---------------------------------------------------------------------------
# 工具侧信道——异步天气/日历工具，支持插入填充语
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    latency_ms: int
    result: str


WEATHER = Tool("weather.tokyo_tomorrow", latency_ms=420, result="68/52，局部多云")


# ---------------------------------------------------------------------------
# 调度器——逐帧流式运行完整流水线
# ---------------------------------------------------------------------------

def run_session(frames: list[Frame], use_tool: bool = True,
                barge_in_at_ms: int | None = None) -> Metrics:
    m = Metrics()
    state = State.IDLE
    silence_run_ms = 0
    final_partial = ""
    llm_stream_started_at = -1
    tts_stream_started_at = -1
    tool_started_at = -1
    tool_done_at = -1
    filler_emitted = False

    for f in frames:
        # 插话：智能体处于 SPEAKING 或 THINKING 时用户开始说话
        if (barge_in_at_ms is not None and f.t_ms >= barge_in_at_ms
                and state in (State.SPEAKING, State.THINKING)
                and f.is_speech):
            m.barge_ins += 1
            m.log(f"{f.t_ms}ms 插话：取消 TTS，重新启用 ASR")
            state = State.LISTENING
            tts_stream_started_at = -1
            llm_stream_started_at = -1
            continue

        if state == State.IDLE:
            if f.is_speech:
                state = State.LISTENING
                m.log(f"{f.t_ms}ms LISTENING")

        elif state == State.LISTENING:
            if f.is_speech:
                silence_run_ms = 0
                final_partial = f.partial or final_partial
            else:
                silence_run_ms += 20
                if silence_run_ms >= 500:
                    score = turn_completion_score(final_partial)
                    if score >= 0.6:
                        state = State.WAITING
                        m.turn_complete_ms = f.t_ms
                        m.log(f"{f.t_ms}ms TURN COMPLETE (score={score:.2f})"
                              f" 部分结果='{final_partial}'")
                    else:
                        m.log(f"{f.t_ms}ms 检测到静音，但分数={score:.2f}，继续等待")

        if state == State.WAITING:
            # 启动 LLM
            llm_stream_started_at = f.t_ms + 140  # 模拟首个 token 延迟
            state = State.THINKING
            m.log(f"{f.t_ms}ms 已发起 LLM 调用")
            if use_tool:
                tool_started_at = f.t_ms
                state = State.TOOL

        elif state == State.TOOL:
            if tool_started_at >= 0 and not filler_emitted:
                if f.t_ms - tool_started_at >= 300:
                    filler_emitted = True
                    m.log(f"{f.t_ms}ms 填充语：'稍等，让我查一下'")
            if tool_started_at >= 0 and f.t_ms - tool_started_at >= WEATHER.latency_ms:
                tool_done_at = f.t_ms
                m.log(f"{f.t_ms}ms 工具结果：{WEATHER.result}")
                llm_stream_started_at = f.t_ms + 140
                state = State.THINKING

        elif state == State.THINKING:
            if llm_stream_started_at > 0 and f.t_ms >= llm_stream_started_at:
                if m.first_llm_token_ms == 0:
                    m.first_llm_token_ms = f.t_ms
                    m.log(f"{f.t_ms}ms LLM 首个 token")
                tts_stream_started_at = f.t_ms + 180
                state = State.SPEAKING

        elif state == State.SPEAKING:
            if tts_stream_started_at > 0 and f.t_ms >= tts_stream_started_at:
                if m.first_audio_out_ms == 0:
                    m.first_audio_out_ms = f.t_ms
                    m.log(f"{f.t_ms}ms TTS 首次音频输出")

    return m


# ---------------------------------------------------------------------------
# 演示——运行两个会话：一个正常会话，一个包含插话
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(0)
    print("=== 会话 1：使用天气工具的正常通话 ===")
    frames = synth_call("what is the weather in tokyo tomorrow", start_ms=0)
    m = run_session(frames, use_tool=True, barge_in_at_ms=None)
    for line in m.events:
        print(" ", line)
    print(f"  轮次完成      @ {m.turn_complete_ms}ms")
    print(f"  LLM 首个 token @ {m.first_llm_token_ms}ms")
    print(f"  首次音频输出   @ {m.first_audio_out_ms}ms")
    print(f"  轮次延迟       = {m.latency_ms()}ms")

    print()
    print("=== 会话 2：用户在回复中途插话 ===")
    frames = synth_call("tell me a long story about", start_ms=0)
    # 在尾部静音的后段加入几个合成语音帧
    for i in range(8):
        idx = len(frames) - 20 + i
        if 0 <= idx < len(frames):
            frames[idx] = Frame(t_ms=frames[idx].t_ms, is_speech=True,
                                partial=frames[idx].partial)
    m = run_session(frames, use_tool=False,
                    barge_in_at_ms=frames[-20].t_ms - 60)
    for line in m.events:
        print(" ", line)
    print(f"  插话次数 = {m.barge_ins}")


if __name__ == "__main__":
    main()
