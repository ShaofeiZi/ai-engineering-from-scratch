"""金丝雀发布模拟器——使用 Python 标准库。

逐步增加候选版本的流量份额，并在每个阶段检查五道门禁。任一门禁被突破
即停止发布。支持注入回归。
"""

from __future__ import annotations

from dataclasses import dataclass
import random


STAGES = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00]

BASELINE = {
    "latency_p99_ms": 900,
    "cost_per_req": 0.02,
    "error_rate": 0.02,
    "output_len_p99": 450,
    "thumbs_down_rate": 0.03,
}

GATES = {
    "latency_p99_ms": 1.5,
    "cost_per_req": 1.2,
    "error_rate": 2.0,
    "output_len_p99": 1.4,
    "thumbs_down_rate": 1.5,
}


@dataclass
class Regression:
    latency_mult: float = 1.0
    cost_mult: float = 1.0
    error_mult: float = 1.0
    output_len_mult: float = 1.0
    thumbs_down_mult: float = 1.0


def measure_stage(stage: float, reg: Regression, seed: int) -> dict:
    rng = random.Random(seed)
    noise = lambda v: v * rng.uniform(0.92, 1.08)
    return {
        "latency_p99_ms": noise(BASELINE["latency_p99_ms"] * reg.latency_mult),
        "cost_per_req": noise(BASELINE["cost_per_req"] * reg.cost_mult),
        "error_rate": noise(BASELINE["error_rate"] * reg.error_mult),
        "output_len_p99": noise(BASELINE["output_len_p99"] * reg.output_len_mult),
        "thumbs_down_rate": noise(BASELINE["thumbs_down_rate"] * reg.thumbs_down_mult),
    }


def check_gates(metrics: dict) -> list[str]:
    breaches = []
    for k, mult in GATES.items():
        if metrics[k] > BASELINE[k] * mult:
            breaches.append(k)
    return breaches


def rollout(name: str, reg: Regression) -> None:
    print(f"\n{name}")
    print(f"回归倍数：延迟={reg.latency_mult}，成本={reg.cost_mult}，错误={reg.error_mult}，长度={reg.output_len_mult}，差评={reg.thumbs_down_mult}")
    for i, stage in enumerate(STAGES):
        metrics = measure_stage(stage, reg, seed=stage_seed(i))
        breaches = check_gates(metrics)
        status = "通过" if not breaches else f"停止（{','.join(breaches)}）"
        pct = int(stage * 100)
        print(f"  阶段 {pct:3}%  "
              f"延迟_p99={metrics['latency_p99_ms']:5.0f}  "
              f"成本=${metrics['cost_per_req']:.4f}  "
              f"错误={metrics['error_rate']*100:4.1f}%  "
              f"差评={metrics['thumbs_down_rate']*100:4.1f}%  "
              f"{status}")
        if breaches:
            print("  → 回滚（切换策略，恢复固定模型）")
            return
    print("  → 已推广至 100%")


def stage_seed(i: int) -> int:
    return 11 + i * 3


def main() -> None:
    print("=" * 95)
    print("金丝雀发布——六个阶段、五道门禁、注入回归")
    print("=" * 95)

    rollout("无回归推广", Regression())
    rollout("轻微成本回归（10%）——未超门禁", Regression(cost_mult=1.10))
    rollout("成本回归 25%", Regression(cost_mult=1.25))
    rollout("延迟回归 80%", Regression(latency_mult=1.80))
    rollout("差评率回归 60%", Regression(thumbs_down_mult=1.60))
    rollout("质量静默下降且成本缓慢上升", Regression(cost_mult=1.15, thumbs_down_mult=1.45))


if __name__ == "__main__":
    main()
