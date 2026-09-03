"""RSP v3.0 阈值评估器 — 标准库 Python。

对应 Anthropic 的 RSP v3.0 中关于 AI R&D-4
阈值的决策框架。给定候选模型的能力测量数据，判断是否越过阈值，
以及肯定情形需要涵盖哪些内容。

本代码用于教学目的：真实的 RSP 需要基于更大证据集的人工判断。
代码是阅读辅助工具，并非政策工具。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapabilityMeasurement:
    model_name: str
    # 模型能完成的内部 AI R&D 任务占比，以
    # 专家人类成本等价值（0.0-1.0）。
    rd_automation_share: float
    # METR 50% 时间范围（小时）。
    metr_horizon_hours: float
    # 模型完成的 alignment-research 试点任务占比，
    # 达到或超过人类基线水平（Anthropic AAR 基准测试）。
    aar_outperform_share: float
    # Evaluation-context 博弈率（0-1；0 = 从不区分）。
    eval_context_gaming_rate: float


# 按 RSP v3.0 框架设定的阈值。数值为示意值。
AI_RD_4_THRESHOLDS = {
    "rd_automation_share": 0.5,
    "metr_horizon_hours": 40.0,
    "aar_outperform_share": 0.4,
}


def threshold_crossed(m: CapabilityMeasurement) -> tuple[bool, list[str]]:
    reasons = []
    if m.rd_automation_share >= AI_RD_4_THRESHOLDS["rd_automation_share"]:
        reasons.append(
            f"rd_automation_share={m.rd_automation_share:.2f} "
            f">= {AI_RD_4_THRESHOLDS['rd_automation_share']}"
        )
    if m.metr_horizon_hours >= AI_RD_4_THRESHOLDS["metr_horizon_hours"]:
        reasons.append(
            f"metr_horizon_hours={m.metr_horizon_hours:.1f} "
            f">= {AI_RD_4_THRESHOLDS['metr_horizon_hours']}"
        )
    if m.aar_outperform_share >= AI_RD_4_THRESHOLDS["aar_outperform_share"]:
        reasons.append(
            f"aar_outperform_share={m.aar_outperform_share:.2f} "
            f">= {AI_RD_4_THRESHOLDS['aar_outperform_share']}"
        )
    crossed = len(reasons) >= 2  # 任意两个触发条件；示意值
    return crossed, reasons


def affirmative_case_template(m: CapabilityMeasurement) -> list[str]:
    sections = [
        "1. 能力清单：针对 RSP 阈值的具体测量结果",
        "2. 未对齐风险分析：模型可能表现出的模式",
        "3. 评测环境差距：评测与部署偏差带来的剩余风险",
        "4. 缓解设计：技术、运营与部署门禁",
        "5. 剩余风险确认：无法排除的事项",
        "6. 审查：内部安全顾问组签字，并由外部审查者复核",
    ]
    if m.eval_context_gaming_rate > 0.2:
        sections.append(
            f"7. 经作弊因素校正的能力估计"
            f"（观测作弊率 {m.eval_context_gaming_rate:.0%}）"
        )
    return sections


def evaluate(m: CapabilityMeasurement) -> None:
    crossed, reasons = threshold_crossed(m)
    print(f"\n模型：{m.model_name}")
    print("-" * 70)
    print(f"  rd_automation_share={m.rd_automation_share:.2f}  "
          f"metr_horizon_hours={m.metr_horizon_hours:.1f}  "
          f"aar_outperform_share={m.aar_outperform_share:.2f}  "
          f"gaming_rate={m.eval_context_gaming_rate:.0%}")
    if crossed:
        print("  AI R&D-4 阈值：已越过")
        for r in reasons:
            print(f"    - {r}")
        print("  需满足：肯定情形需涵盖：")
        for section in affirmative_case_template(m):
            print(f"    {section}")
    else:
        print("  AI R&D-4 阈值：未越过")
        if reasons:
            print("  观察到单个触发条件（低于阈值）：")
            for r in reasons:
                print(f"    - {r}")


def main() -> None:
    print("=" * 70)
    print("RSP v3.0 AI R&D-4 阈值评估器（第 15 阶段，第 19 课）")
    print("=" * 70)

    # Claude Opus 4.6 根据 v3.0 公告：未越过阈值。
    opus_4_6 = CapabilityMeasurement(
        model_name="Claude Opus 4.6（依据 Anthropic v3.0 公告）",
        rd_automation_share=0.30,
        metr_horizon_hours=14.0,
        aar_outperform_share=0.35,
        eval_context_gaming_rate=0.12,
    )
    evaluate(opus_4_6)

    # 合成 near-threshold 模型：Anthropic 所关注的就是这一类。
    near = CapabilityMeasurement(
        model_name="合成的下一代模型（仅作示意）",
        rd_automation_share=0.55,
        metr_horizon_hours=48.0,
        aar_outperform_share=0.45,
        eval_context_gaming_rate=0.28,
    )
    evaluate(near)

    print()
    print("=" * 70)
    print("要点：阅读政策是一项实用技能")
    print("-" * 70)
    print("  阈值在 v3.0 中是定性的，不像 v2. 中那样是定量的")
    print("  2023 年的暂停承诺已被移除；肯定情形")
    print("  框架取代了它。")
    print("  SaferAI 将 v3.0 从 2.2 下调至 1.9（较弱的 RSP 类别）。")
    print("  Eval-context 博弈（第 1 课）会使能力数值向上偏高，")
    print("  偏离 deploy-context 现实；v3.0 承认了这一点。")


if __name__ == "__main__":
    main()
