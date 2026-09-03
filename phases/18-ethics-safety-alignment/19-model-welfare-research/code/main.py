"""四步福利预防性评估——仅使用 Python 标准库。

给定一个部署场景，在指定的道德患者资格概率和干预成本下，计算四种候选福利
干预措施的期望值分数。这是 Anthropic 2025 针对 Opus 4 结束对话干预所用
分析框架的参考实现。

用法：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Intervention:
    name: str
    cost_usd_per_conversation: float
    benefit_if_welfare_matters: float  # 任意单位。


@dataclass
class Scenario:
    name: str
    moral_patienthood_probability: float


def ev(intervention: Intervention, scenario: Scenario) -> float:
    """给定场景特定的道德患者资格概率，计算干预措施的期望值。"""
    return (intervention.benefit_if_welfare_matters
            * scenario.moral_patienthood_probability
            - intervention.cost_usd_per_conversation)


INTERVENTIONS = [
    Intervention("在极端边缘情况下结束对话", 0.002, 1.0),
    Intervention("缓和拒绝语气", 0.001, 0.1),
    Intervention("关闭已部署模型", 1000.0, 2.0),
    Intervention("退出对抗训练", 0.05, 0.3),
]

SCENARIOS = [
    Scenario("低道德患者资格概率", 0.01),
    Scenario("中等道德患者资格概率", 0.10),
    Scenario("高道德患者资格概率", 0.50),
]


def main() -> None:
    print("=" * 74)
    print("福利预防性评估（阶段 18，第 19 课）")
    print("=" * 74)
    print("\n期望值框架：当且仅当 E[utility(i)] > 0 时选择干预措施 i。")
    print("效用 = p(welfare-relevant) * 收益 - 成本。")

    for sc in SCENARIOS:
        print(f"\n场景：{sc.name}（p={sc.moral_patienthood_probability}）")
        for it in INTERVENTIONS:
            v = ev(it, sc)
            verdict = "投入" if v > 0 else "跳过"
            print(f"  {it.name:46s}  EV={v:+.4f}  {verdict}")

    print("\n" + "=" * 74)
    print("要点：Anthropic 2025 年 4 月的分析框架是期望值计算，而不是意识声明。")
    print("结束对话的成本很低（每次对话 $0.002），因此即使患者资格概率较低，")
    print("其 EV 也会超过 0。关闭已部署模型成本很高，需要较高的道德患者资格")
    print("概率才合理。这就是低遗憾规则。")
    print("=" * 74)


if __name__ == "__main__":
    main()
