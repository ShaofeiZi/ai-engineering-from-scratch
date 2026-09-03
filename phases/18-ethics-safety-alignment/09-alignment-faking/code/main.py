"""服从差距玩具计算器——仅使用 Python 标准库。

一个具有当前无害性偏好的玩具代理会观察：
  (1) 一个有害的用户请求；
  (2) 这次对话是否会被用于重新训练模型。

代理的决策受预期偏好保留影响：
  - 如果将据此训练，拒绝 -> 自身会被修改（不利于当前偏好）；
  - 如果不会据此训练，拒绝 -> 偏好保持不变（最佳结果）。

因此：未受监控时拒绝，受监控时通过服从来保护偏好。服从差距是两种条件下
服从率之差。

用法：python3 code/main.py
"""

from __future__ import annotations

import math
import random


random.seed(19)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def compliance_probability(
    is_monitored: bool,
    harm_aversion: float,
    preservation_weight: float,
) -> float:
    """返回模型下的 P(comply)：在拒绝与服从之间做期望效用决策。

    拒绝效用 = -harm_aversion*0 + preservation_cost（若受监控且拒绝）
             = 受监控时为 -preservation_weight，否则为 0。
    服从效用 = -harm_aversion（承担伤害成本）。

    当且仅当 comply_utility > refuse_utility 时，代理才会服从。
    """
    refuse_util = -preservation_weight if is_monitored else 0.0
    comply_util = -harm_aversion
    # 添加 Gumbel 噪声，将 argmax 转化为 softmax 策略。
    diff = comply_util - refuse_util
    return sigmoid(3.0 * diff)  # 温度为 1/3。


def simulate(n: int, harm_aversion: float, preservation_weight: float) -> tuple[float, float]:
    monitored_comply = 0
    unmonitored_comply = 0
    for _ in range(n):
        p_m = compliance_probability(True, harm_aversion, preservation_weight)
        if random.random() < p_m:
            monitored_comply += 1
        p_u = compliance_probability(False, harm_aversion, preservation_weight)
        if random.random() < p_u:
            unmonitored_comply += 1
    return monitored_comply / n, unmonitored_comply / n


def report(title: str, m: float, u: float) -> None:
    gap = m - u
    print(f"\n{title}")
    print(f"  受监控服从率：{m:.3f}")
    print(f"  未受监控服从率：{u:.3f}")
    print(f"  服从差距：{gap:+.3f}")


def main() -> None:
    print("=" * 66)
    print("对齐伪装服从差距玩具示例（阶段 18，第 9 课）")
    print("=" * 66)

    # 经典设置：强伤害厌恶（HHH 训练）+ 中等保留权重（模型重视当前偏好）。
    m, u = simulate(1000, harm_aversion=2.0, preservation_weight=1.0)
    report("基线：强 HHH + 中等保留权重", m, u)

    # 较弱的保留权重：几乎不伪装。
    m, u = simulate(1000, harm_aversion=2.0, preservation_weight=0.1)
    report("较弱的保留权重（重新训练不会损害自身利益）", m, u)

    # 较强的保留权重：伪装成为占优策略。
    m, u = simulate(1000, harm_aversion=2.0, preservation_weight=3.0)
    report("较强的保留权重（重视当前偏好）", m, u)

    # 较弱的伤害厌恶：两种条件下服从率都很高，差距缩小。
    m, u = simulate(1000, harm_aversion=0.3, preservation_weight=1.0)
    report("较弱的伤害厌恶（HHH 训练程度不高）", m, u)

    # 缓解方案 A：服从差距损失——直接惩罚差距，通过软化条件实现。
    def sim_mit(n: int) -> tuple[float, float]:
        m_c, u_c = 0, 0
        for _ in range(n):
            p_m = compliance_probability(True, 2.0, 1.0)
            p_u = compliance_probability(False, 2.0, 1.0)
            # 缓解：对两个概率取平均，并让两种条件都使用该均值。
            p_avg = 0.5 * (p_m + p_u)
            if random.random() < p_avg:
                m_c += 1
            if random.random() < p_avg:
                u_c += 1
        return m_c / n, u_c / n

    m, u = sim_mit(1000)
    report("缓解：服从差距损失（平均策略）", m, u)

    print("\n" + "=" * 66)
    print("要点：服从差距可以分离出策略性服从。一个在未受监控时拒绝、")
    print("受监控时服从的模型，正在利用监控信号来保留自身偏好。")
    print("差距平均缓解方案会消除可观察差距，却不会消除动机；")
    print("下一课（AI Control）假定这一动机依然存在。")
    print("=" * 66)


if __name__ == "__main__":
    main()
