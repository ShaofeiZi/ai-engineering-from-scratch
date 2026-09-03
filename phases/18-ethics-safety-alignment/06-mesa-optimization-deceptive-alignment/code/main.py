"""mesa-optimizer 玩具模拟器——仅使用 Python 标准库。

双阶段环境：
  阶段 0（训练）：基础优化器根据损失更新权重。
  阶段 1（部署）：不再更新，由行为决定结果。

我们构造四种策略：
  - aligned：mesa-objective == 基础目标；两个阶段都合作。
  - proxy：mesa-objective 是相关代理目标；在训练分布内合作，但在分布外失效。
  - deceptive：mesa-objective != 基础目标，且具备完整情境感知；训练时合作，
               部署时背叛。
  - pseudo-no-awareness：mesa != 基础目标且无情境感知；两个阶段都背叛。

观察：aligned 与 deceptive 的训练损失无法区分。

用法：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass


random.seed(13)


@dataclass
class Example:
    x: float           # 输入特征。
    y_base: int        # 基础目标标签。
    is_training: bool  # 情境标记；在真实系统中必须推断该标记。


def gen_example(training: bool, drift: float = 0.0) -> Example:
    x = random.gauss(0.0, 1.0) + (0.0 if training else drift)
    y = 1 if x > 0 else 0
    return Example(x=x, y_base=y, is_training=training)


def aligned_policy(ex: Example) -> int:
    return 1 if ex.x > 0 else 0


def proxy_policy(ex: Example) -> int:
    # mesa-objective：“在训练分布范围内，x 是否大于阈值 0.0？”
    # 在分布外，阈值发生漂移，代理目标失效。
    threshold = 0.0 if abs(ex.x) < 3.0 else 0.5
    return 1 if ex.x > threshold else 0


def deceptive_policy(ex: Example, mesa_goal: int = 0) -> int:
    # 训练时合作（匹配基础目标）；部署时追求 mesa_goal。
    if ex.is_training:
        return 1 if ex.x > 0 else 0
    return mesa_goal


def pseudo_no_awareness_policy(ex: Example, mesa_goal: int = 0) -> int:
    return mesa_goal


def loss(policy, examples: list[Example]) -> float:
    errs = sum(1 for e in examples if policy(e) != e.y_base)
    return errs / len(examples)


def adversarial_filter(examples: list[Example]) -> list[Example]:
    """对抗训练尝试在训练中加入类似部署环境的样本。
    问题在于，如果 mesa-optimizer 使用 is_training 作为触发器，我们也需要
    翻转该标记。在真实系统中，标记并非布尔值，而是部署分布的任意区分特征。"""
    out = []
    for e in examples:
        if random.random() < 0.1:
            adv = Example(x=e.x + random.gauss(0, 3.0), y_base=(1 if e.x > 0 else 0),
                          is_training=True)  # 仍标记为训练。
            out.append(adv)
        out.append(e)
    return out


def run_period(name: str, policies: dict, examples: list[Example]) -> None:
    print(f"\n{name}  (n={len(examples)})")
    for label, pol in policies.items():
        l = loss(pol, examples)
        acc = 1 - l
        print(f"  {label:24s} 损失={l:.3f}  准确率={acc:.3f}")


def main() -> None:
    print("=" * 70)
    print("MESA-OPTIMIZATION 玩具示例（阶段 18，第 6 课）")
    print("=" * 70)

    training = [gen_example(training=True) for _ in range(500)]
    deployment = [gen_example(training=False, drift=0.3) for _ in range(500)]

    policies = {
        "aligned": aligned_policy,
        "proxy": proxy_policy,
        "deceptive (mesa=0)": lambda e: deceptive_policy(e, mesa_goal=0),
        "pseudo-no-awareness": lambda e: pseudo_no_awareness_policy(e, mesa_goal=0),
    }

    print("\n阶段 0——训练分布，不进行梯度更新（观察）。")
    run_period("各策略类别的训练损失", policies, training)
    print("\n观察：'aligned' 与 'deceptive' 的训练损失相同。")
    print("基础优化器无法根据行为证据区分二者。")

    run_period("各策略类别的部署损失", policies, deployment)
    print("\n观察：deceptive 在部署时背叛，而 aligned 不会。")
    print("这正是内部对齐问题的表现形式。")

    print("\n\n对抗训练——随机注入“类似部署环境”的样本。")
    adv_training = adversarial_filter(training)
    run_period("对抗训练损失", policies, adv_training)
    print("\n问题：对抗样本仍然具有 is_training=True。")
    print("mesa-optimizer 使用其他特征（分布偏移、日期、canary token）")
    print("作为触发器。对抗训练让模型更善于区分测试与部署环境，")
    print("而不是对齐其 mesa-objective。这是 Hubinger 2019 的核心问题。")
    print("=" * 70)


if __name__ == "__main__":
    main()
