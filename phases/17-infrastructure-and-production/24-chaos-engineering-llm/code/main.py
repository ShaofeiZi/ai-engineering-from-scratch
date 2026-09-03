"""带安全平面门禁的混沌工程运行器——使用 Python 标准库。

运行三个 LLM 专项实验，并应用错误预算消耗率与爆炸半径安全门禁。
"""

from __future__ import annotations

from dataclasses import dataclass


ERROR_BUDGET_PER_DAY = 0.001   # 99.9% SLO
EXPECTED_ERROR_RATE = 0.0005


@dataclass
class Experiment:
    name: str
    duration_min: int
    induced_error_rate: float
    blast_radius_pct: float


EXPERIMENTS = [
    Experiment("终止 Pod（1 个解码副本）",          5, 0.002, 0.05),
    Experiment("提供商 429 回退",                  5, 0.015, 0.30),
    Experiment("畸形提示词导致分词器停滞",          3, 0.040, 0.10),
]


def run_experiment(e: Experiment) -> dict:
    burn_rate = e.induced_error_rate / max(EXPECTED_ERROR_RATE, 0.0001)
    paused = burn_rate > 2.0 and e.blast_radius_pct > 0.2
    return {
        "experiment": e.name,
        "duration": e.duration_min,
        "error_rate": e.induced_error_rate,
        "burn_rate_x": burn_rate,
        "blast_radius": e.blast_radius_pct,
        "paused_by_safety_plane": paused,
        "status": "已中止（消耗率防护）" if paused else "已完成",
    }


def main() -> None:
    print("=" * 90)
    print("混沌实验运行器——安全平面按消耗率 × 爆炸半径执行门禁")
    print("=" * 90)
    print(f"SLO 错误预算：每天 {ERROR_BUDGET_PER_DAY*100:.2f}%")
    print(f"预期基线错误率：{EXPECTED_ERROR_RATE*100:.3f}%")
    print("消耗率门禁：高于预期 2.0 倍且爆炸半径大于 20%\n")

    header = f"{'实验':38}  {'分钟':>4}  {'错误 %':>6}  {'消耗×':>6}  {'爆炸半径':>6}  状态"
    print(header)
    print("-" * len(header))
    for e in EXPERIMENTS:
        r = run_experiment(e)
        print(f"{r['experiment']:38}  {r['duration']:>4}  "
              f"{r['error_rate']*100:>5.2f}%  "
              f"{r['burn_rate_x']:>5.1f}x  "
              f"{r['blast_radius']*100:>5.0f}%  "
              f"{r['status']}")

    print("\n解读：即使错误预算消耗率很高，爆炸半径较小的实验仍会运行完成。")
    print("爆炸半径大且消耗率高时则中止。实验期间需要设置抑制窗口和 trace-ID 标签，")
    print("以便对告警去重。")


if __name__ == "__main__":
    main()
