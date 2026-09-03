"""长视频 token 预算、大海捞针模拟器与智能体检索。

标准库实现。打印长视频预算表，运行合成 NIH 召回测试，
模拟 VideoAgent 风格的检索循环。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

random.seed(5)


def tokens(duration_s: float, fps: float, per_frame: int) -> int:
    return int(duration_s * fps * per_frame)


def budget_table() -> None:
    print("\n长视频 TOKEN 预算")
    print("-" * 60)
    print(f"{'时长':<14}{'FPS':>5}{'每帧':>12}{'token 数':>12}{'适用上下文':>14}")
    cases = [
        (60, 1, 81,     "32k+"),
        (300, 1, 81,    "32k"),
        (300, 2, 81,    "128k"),
        (1800, 1, 81,   "256k"),
        (3600, 1, 81,   "1M / LongVILA"),
        (7200, 1, 81,   "仅 Gemini 2.5"),
        (7200, 1, 32,   "智能体检索"),
    ]
    for dur, fps, pf, fits in cases:
        t = tokens(dur, fps, pf)
        print(f"{dur//60} 分钟{' ':<7}{fps:>5}{pf:>12}{t:>12,}   {fits}")


@dataclass
class Needle:
    t: float
    marker: str


def nih_trial(duration_s: float, model_recall_curve: list[tuple[float, float]]) -> dict:
    needle_t = random.uniform(0, duration_s)
    needle = Needle(t=needle_t, marker="独特贴纸")
    pct_into_video = needle_t / duration_s
    for thresh, recall in model_recall_curve:
        if pct_into_video <= thresh:
            return {"needle_time": needle_t,
                    "pct_into_video": pct_into_video,
                    "recall_prob": recall}
    return {"needle_time": needle_t,
            "pct_into_video": pct_into_video,
            "recall_prob": model_recall_curve[-1][1]}


def nih_simulation() -> None:
    print("\n大海捞针模拟（每个模型单次试验）")
    print("-" * 60)
    models = [
        ("Qwen2.5-VL-72B @ 15 分钟", 900,  [(0.1, 0.98), (0.5, 0.90), (1.0, 0.85)]),
        ("Qwen2.5-VL-72B @ 30 分钟", 1800, [(0.1, 0.95), (0.5, 0.85), (1.0, 0.75)]),
        ("Gemini 2.5 Pro @ 90 分钟", 5400, [(0.1, 0.99), (0.5, 0.99), (1.0, 0.99)]),
        ("VideoAgent（检索）2 小时", 7200, [(0.1, 0.92), (0.5, 0.92), (1.0, 0.92)]),
    ]
    for name, dur, curve in models:
        r = nih_trial(dur, curve)
        print(f"  {name:<32}  针所在时刻={r['needle_time']:>6.1f} 秒  "
              f"召回概率={r['recall_prob']:.2f}")


def agentic_retrieval_sim(question: str, video_duration: float) -> dict:
    """模拟 VideoAgent：LLM 请求片段，工具返回时间戳，VLM 读取。"""
    trace = []
    trace.append(("LLM  ", f"读取问题：'{question}'"))
    query = question.split()[-1].lower()
    trace.append(("LLM  ", f"调用工具：find_clips(keyword='{query}')"))
    hits = sorted([random.uniform(0, video_duration) for _ in range(3)])
    trace.append(("工具 ", f"返回 3 个片段：{[round(h,1) for h in hits]}"))
    trace.append(("VLM  ", "编码 3 个 30 秒片段（共约 7290 个 token）"))
    trace.append(("LLM  ", "根据片段描述组织答案"))
    tokens_used = 3 * 30 * 81 + 200
    return {"steps": trace, "tokens": tokens_used}


def agentic_demo() -> None:
    print("\nVIDEOAGENT 风格检索（2 小时视频）")
    print("-" * 60)
    r = agentic_retrieval_sim("猫在什么时刻跳跃", 7200)
    for role, msg in r["steps"]:
        print(f"  [{role}] {msg}")
    print(f"\n  使用的总 token 数：~{r['tokens']:,}")
    print(f"  对比 2 小时暴力上下文 @ 1 FPS：约 583,000 tokens")
    print("  -> 单事件查询的推理成本降低 99%")


def main() -> None:
    print("=" * 60)
    print("长视频理解（第12阶段，第18课）")
    print("=" * 60)

    budget_table()
    nih_simulation()
    agentic_demo()

    print("\n策略选择器")
    print("-" * 60)
    print("  <15 分钟           : 暴力上下文（Qwen2.5-VL-72B）")
    print("  15-60 分钟         : LongVILA / Video-XL / Gemini 2.5")
    print("  >1小时 通用 QA     : Gemini 2.5 Pro（闭源前沿模型）")
    print("  >1小时 特定查询    : VideoAgent（智能体检索）")


if __name__ == "__main__":
    main()
