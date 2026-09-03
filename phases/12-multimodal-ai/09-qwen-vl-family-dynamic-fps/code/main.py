"""Qwen-VL 家族：M-RoPE 位置编码 + 动态 FPS 采样器 + JSON 工具调用解析器。

三个示例实现：
  1. M-RoPE 轮换表，覆盖文本、图像和视频 token。
  2. 动态 FPS 采样器，根据目标 token 预算选取每秒帧数。
  3. JSON 输出解析器，用于 Qwen2.5-VL 风格的智能体工具调用。

仅使用标准库。目的是构建可运行的心智模型，而非生产级代码。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass
class MRoPEConfig:
    hidden: int
    temporal_dim: int
    height_dim: int
    width_dim: int
    base: float = 10000.0


def mrope_angles(cfg: MRoPEConfig, t: int, h: int, w: int) -> list[float]:
    """根据 (t, h, w) 位置返回每个频带的 per-pair 旋转角度。"""
    angles = []
    for dim, pos in [(cfg.temporal_dim, t), (cfg.height_dim, h), (cfg.width_dim, w)]:
        band = []
        pairs = dim // 2
        for i in range(pairs):
            theta = cfg.base ** (-2 * i / dim)
            band.append(pos * theta)
        angles.append(band)
    return angles


def mrope_rotate(cfg: MRoPEConfig, vec: list[float], t: int, h: int, w: int) -> list[float]:
    """将 M-RoPE 应用于长度为 cfg.hidden. 的向量"""
    out = list(vec)
    axes = [
        (cfg.temporal_dim, t, 0),
        (cfg.height_dim, h, cfg.temporal_dim),
        (cfg.width_dim, w, cfg.temporal_dim + cfg.height_dim),
    ]
    for dim, pos, start in axes:
        pairs = dim // 2
        for i in range(pairs):
            theta = cfg.base ** (-2 * i / dim)
            angle = pos * theta
            idx0 = start + 2 * i
            idx1 = start + 2 * i + 1
            c, s = math.cos(angle), math.sin(angle)
            v0, v1 = out[idx0], out[idx1]
            out[idx0] = v0 * c - v1 * s
            out[idx1] = v0 * s + v1 * c
    return out


@dataclass
class VideoPlan:
    duration_s: float
    tokens_per_frame: int
    budget: int
    motion: str

    def fps(self) -> float:
        fps_max = self.budget / (self.duration_s * self.tokens_per_frame)
        if self.motion == "high":
            candidates = [8, 4, 2, 1, 0.5, 0.25]
        elif self.motion == "medium":
            candidates = [4, 2, 1, 0.5, 0.25]
        else:
            candidates = [1, 0.5, 0.25, 0.1]
        for f in candidates:
            if f <= fps_max:
                return f
        return candidates[-1]

    def frame_times(self) -> list[float]:
        f = self.fps()
        n_frames = max(1, int(self.duration_s * f))
        step = 1.0 / f
        return [round(i * step, 3) for i in range(n_frames)]

    def total_tokens(self) -> int:
        return len(self.frame_times()) * self.tokens_per_frame


def parse_tool_call(raw: str) -> dict:
    """Qwen2.5-VL 发出 JSON 个工具调用；进行解析并带回退。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"tool": "PARSE_ERROR", "raw": raw}


def demo_mrope() -> None:
    print("\nM-RoPE 位置旋转（hidden=48，每个频带 16）")
    print("-" * 60)
    cfg = MRoPEConfig(hidden=48, temporal_dim=16, height_dim=16, width_dim=16)
    positions = [
        ("文本 token i=0",      0, 0, 0),
        ("文本 token i=12",     12, 0, 0),
        ("图像 patch（h=5, w=7）", 0, 5, 7),
        ("视频帧 t=3（h=5, w=7）", 3, 5, 7),
    ]
    for name, t, h, w in positions:
        angles = mrope_angles(cfg, t, h, w)
        first_pair = [round(a[0], 4) for a in angles]
        print(f"  {name:<30} 第一对角度 (t, h, w) = {first_pair}")


def demo_sampler() -> None:
    print("\n动态 FPS 采样器（3 倍池化后每帧 token 数=81）")
    print("-" * 60)
    videos = [
        ("30 秒网球对打（高速运动）",   30.0, "high"),
        ("30 秒菜谱演示（中速运动）",   30.0, "medium"),
        ("10 分钟安防循环（低速运动）", 600.0, "low"),
        ("1 分钟 UI 智能体回放（中速）", 60.0, "medium"),
    ]
    budget = 32768
    print(f"为每个视频预算 {budget} 个 token：")
    for name, dur, motion in videos:
        plan = VideoPlan(duration_s=dur, tokens_per_frame=81, budget=budget, motion=motion)
        n_frames = len(plan.frame_times())
        print(f"  {name:<38}  fps={plan.fps()}  帧数={n_frames:>4}  token 数={plan.total_tokens():>6}")


def demo_tool_parser() -> None:
    print("\nQWEN2.5-VL 工具调用解析器")
    print("-" * 60)
    examples = [
        '{"tool": "mouse_click", "coords": [1024, 512], "button": "left"}',
        '好的，现在点击 {"tool": "mouse_click", "coords": [800, 400]}。',
        '{"tool": "type_text", "text": "你好"',
        '{"tool": "scroll", "direction": "down", "amount": 300}',
    ]
    for raw in examples:
        parsed = parse_tool_call(raw)
        print(f"  原始    : {raw}")
        print(f"  解析后 : {parsed}")
        print()


def main() -> None:
    print("=" * 60)
    print("QWEN-VL 家族（第 12 阶段，第 09 课）")
    print("=" * 60)

    demo_mrope()
    demo_sampler()
    demo_tool_parser()

    print("=" * 60)
    print("演进摘要")
    print("-" * 60)
    print("  Qwen-VL   (2023) : 448 分辨率，grounding，Q-Former")
    print("  Qwen2-VL  (2024) : M-RoPE，原生分辨率，MLP 投影器")
    print("  Qwen2.5-VL(2025) : 动态 FPS，绝对时间 token，JSON 智能体模式")
    print("  Qwen3-VL  (2025) : Qwen3 基座，思考模式，OCR 规模")


if __name__ == "__main__":
    main()
