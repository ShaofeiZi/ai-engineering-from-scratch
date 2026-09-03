"""Emu3 token 计数 + CFG 采样玩具示例——标准库实现。

两个小工具：
  1. 面向不同分辨率图像和视频的 token 数与 FPS 计算器。
  2. 带 classifier-free 引导的自回归采样器（CFG）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

random.seed(0)


@dataclass
class TokCost:
    label: str
    resolution: int
    reduction: int
    video_seconds: float = 0.0
    fps: float = 0.0
    time_reduction: int = 1

    def tokens(self) -> int:
        spatial_per_frame = (self.resolution // self.reduction) ** 2
        if self.video_seconds == 0:
            return spatial_per_frame
        frames = int(self.video_seconds * self.fps)
        frames_reduced = max(1, frames // self.time_reduction)
        return spatial_per_frame * frames_reduced


def token_table() -> None:
    print("\nEMU3 TOKEN COUNTS（按推荐的 tokenizer 精简设置）")
    print("-" * 60)
    configs = [
        TokCost("图像 256x256",  256, 8),
        TokCost("图像 512x512",  512, 8),
        TokCost("图像 1024x1024", 1024, 8),
        TokCost("图像 2048x2048", 2048, 8),
        TokCost("视频 4 秒 @8fps 256x256", 256, 4, 4.0, 8, 4),
        TokCost("视频 10 秒 @8fps 256x256", 256, 4, 10.0, 8, 4),
        TokCost("视频 4 秒 @8fps 512x512", 512, 4, 4.0, 8, 4),
    ]
    print(f"{'配置':<32}{'token 数':>12}{'耗时（30 tps）':>18}")
    for c in configs:
        t = c.tokens()
        latency = t / 30.0
        print(f"  {c.label:<30}{t:>12}{latency:>16.1f}s")


def softmax(xs: list[float], temperature: float = 1.0) -> list[float]:
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def cfg_mix(cond_logits: list[float], uncond_logits: list[float],
            gamma: float) -> list[float]:
    """Classifier-free 引导：混合 = 无条件 + gamma *（条件 - 无条件）。"""
    return [u + gamma * (c - u) for c, u in zip(cond_logits, uncond_logits)]


def sample(probs: list[float]) -> int:
    r = random.random()
    acc = 0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def demo_cfg() -> None:
    print("\nCLASSIFIER-FREE GUIDANCE——对 logit 形状的影响")
    print("-" * 60)
    cond = [2.0, 4.0, 1.0, 3.5, 0.5]
    uncond = [1.0, 2.0, 1.5, 1.8, 1.2]
    for gamma in [0.0, 1.0, 3.0, 5.0, 7.0]:
        mixed = cfg_mix(cond, uncond, gamma)
        probs = softmax(mixed)
        top = probs.index(max(probs))
        print(f"  gamma={gamma:>4.1f}  logit={[round(x,2) for x in mixed]}")
        print(f"            概率={[round(p,3) for p in probs]}  最高项={top}")
    print("\n  gamma 越大 -> 分布越尖锐 -> 生成保真度越高")
    print("  Emu3 建议图像生成 gamma = 3.0，强一致性 gamma = 7.0")


def sample_tokens(cond: list[list[float]], uncond: list[list[float]],
                  gamma: float = 3.0, temp: float = 0.8) -> list[int]:
    """用 CFG + 温度采样长度为 len(cond) 的序列。"""
    out = []
    for c, u in zip(cond, uncond):
        mixed = cfg_mix(c, u, gamma)
        probs = softmax(mixed, temperature=temp)
        out.append(sample(probs))
    return out


def demo_sampling() -> None:
    print("\n自回归图像 TOKEN 采样（玩具级，K=16 码本）")
    print("-" * 60)
    K = 16
    steps = 8
    cond = [[random.gauss(0, 2) for _ in range(K)] for _ in range(steps)]
    uncond = [[random.gauss(0, 1) for _ in range(K)] for _ in range(steps)]
    tokens_no_cfg = sample_tokens(cond, uncond, gamma=1.0, temp=1.0)
    tokens_cfg3 = sample_tokens(cond, uncond, gamma=3.0, temp=0.8)
    tokens_cfg7 = sample_tokens(cond, uncond, gamma=7.0, temp=0.8)
    print(f"  无 CFG      ：{tokens_no_cfg}")
    print(f"  CFG gamma=3 ：{tokens_cfg3}")
    print(f"  CFG gamma=7 ：{tokens_cfg7}")
    print("  gamma 越高，越会收敛到条件分布的模态；规模化后规律相同。")


def main() -> None:
    print("=" * 60)
    print("EMU3——图像与视频的下一 TOKEN 预测（第 12 阶段，第 12 课）")
    print("=" * 60)

    token_table()
    demo_cfg()
    demo_sampling()

    print("\n" + "=" * 60)
    print("EMU3 与 SDXL——高层计算开销对比")
    print("-" * 60)
    print("  训练      ：相当（约 300B token / 约 300M 图像步）")
    print("  推理      ：Emu3 慢（30 tps 时每张 512x512 约 2 分钟）")
    print("                SDXL 快（每张 512x512 约 2-5 秒）")
    print("  质量      ：Emu3 在 FID/GenEval 上持平或更优")
    print("  灵活性    ：Emu3 还能做感知 + 视频；SDXL 不行")


if __name__ == "__main__":
    main()
