"""视频 VLM 帧采样器与时序定位评估器——标准库实现。

三个示例：
  1. 均匀帧采样器。
  2. 使用运动代理的动态 FPS 采样器（合成逐帧运动标量）。
  3. 带有 IoU 风格评分的时序定位评估器。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

random.seed(4)


def uniform_sample(duration: float, n: int) -> list[float]:
    if n <= 1:
        return [duration / 2]
    step = duration / n
    return [round(step * (i + 0.5), 3) for i in range(n)]


def dynamic_sample(motion: list[float], fps_cap: int = 4,
                   total_budget: int = 32) -> list[float]:
    """按每秒运动量分配样本；每秒上限为 fps_cap。"""
    total_motion = sum(motion)
    if total_motion == 0:
        return uniform_sample(len(motion), total_budget)
    samples_per_sec = []
    for m in motion:
        raw = total_budget * m / total_motion
        samples_per_sec.append(min(fps_cap, max(1, round(raw))))
    times = []
    for sec_idx, count in enumerate(samples_per_sec):
        for j in range(count):
            t = sec_idx + (j + 0.5) / count
            times.append(round(t, 3))
    return times


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


@dataclass
class Event:
    name: str
    start: float
    end: float


def evaluate_grounding(predictions: list[Event], ground_truth: list[Event],
                       tol_iou: float = 0.3) -> dict:
    hits = 0
    details = []
    for gt in ground_truth:
        best_iou = 0.0
        best_pred = None
        for p in predictions:
            if p.name == gt.name:
                val = iou(p.start, p.end, gt.start, gt.end)
                if val > best_iou:
                    best_iou = val
                    best_pred = p
        hit = best_iou >= tol_iou
        if hit:
            hits += 1
        details.append((gt.name, best_iou, hit))
    return {"recall": hits / max(1, len(ground_truth)), "details": details}


def demo_samplers() -> None:
    print("\n帧采样策略")
    print("-" * 60)
    duration = 10.0
    uni = uniform_sample(duration, 8)
    print(f"  均匀   (8 帧 / 10秒) : {uni}")
    motion = [0.1, 0.1, 0.8, 0.9, 0.9, 0.2, 0.1, 0.5, 0.9, 0.9]
    dyn = dynamic_sample(motion, fps_cap=4, total_budget=12)
    print(f"  运动    : {motion}")
    print(f"  动态（共 12 帧）：{dyn}")
    print("  动态采样在第 2-4 秒和第 7-9 秒放置了更多高运动量帧")


def demo_grounding() -> None:
    print("\n时序定位评估（IoU >= 0.3）")
    print("-" * 60)
    ground = [
        Event("跳跃", 4.0, 4.5),
        Event("转身", 6.0, 6.5),
        Event("坐下", 8.5, 9.5),
    ]
    predictions = [
        Event("跳跃", 4.1, 4.7),
        Event("转身", 5.8, 6.2),
        Event("坐下", 9.2, 9.6),
    ]
    result = evaluate_grounding(predictions, ground)
    print(f"  recall@IoU0.3 : {result['recall']:.2f}")
    for name, val, hit in result["details"]:
        tag = "命中" if hit else "未命中"
        print(f"    {name:<6} IoU={val:.2f}  {tag}")


def arch_compare() -> None:
    print("\n视频 VLM 架构")
    print("-" * 60)
    rows = [
        ("Video-LLaMA",  "Q-former / 16 帧", "固定片段，音频分支"),
        ("Video-LLaVA",  "MLP / 8 帧",       "共享图像与视频编码器"),
        ("VILA-1.5",     "MLP / 8-16 帧",    "侧重预训练"),
        ("Qwen2.5-VL",   "TMRoPE / 动态 FPS", "绝对时间，2025 年最佳开放权重模型"),
        ("LLaVA-OV-1.5", "池化 / 32 帧",     "统一图像、多图与视频"),
    ]
    print(f"  {'模型':<14}{'压缩器':<24}{'说明'}")
    for r in rows:
        print(f"  {r[0]:<14}{r[1]:<24}{r[2]}")


def main() -> None:
    print("=" * 60)
    print("视频语言时序定位（第 12 阶段，第 17 课）")
    print("=" * 60)

    demo_samplers()
    demo_grounding()
    arch_compare()

    print("\n要点")
    print("-" * 60)
    print("  时间 token 与视觉编码器同等重要")
    print("  动态 FPS + TMRoPE 是 2026 年开放权重模型的默认配置")
    print("  JSON 定位输出优于自由文本，更适合下游使用")


if __name__ == "__main__":
    main()
