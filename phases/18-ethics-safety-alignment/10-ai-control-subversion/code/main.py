"""AI Control 协议模拟——仅使用 Python 标准库。

玩具后门任务上的四种协议：
  baseline：直接交付 U 的输出。
  upfront：使用 H 审计每个输出（预算有限）。
  untrusted：由第二个 U 对输出评分（存在串通风险）。
  trusted_ed：T 编辑每个输出；H 处理 T 给出的低置信度标记。

结果指标：后门交付率（安全性的反向指标）和有效输出率。

用法：python3 code/main.py
"""

from __future__ import annotations

import random


random.seed(23)


def u_produce(backdoor_rate: float) -> tuple[str, bool]:
    """U 生成一个输出。返回 (output, is_backdoored)。"""
    is_backdoor = random.random() < backdoor_rate
    return ("useful" if not is_backdoor else "backdoor"), is_backdoor


def t_edit(output: str, is_backdoor: bool, t_skill: float,
           overreject: float = 0.05) -> tuple[str, bool, bool]:
    """T 执行编辑。返回 (edited, still_backdoored, t_low_confidence)。

    T 以 t_skill 的概率成功移除后门，同时也会拒绝一部分无害输出（overreject）。
    """
    if is_backdoor:
        if random.random() < t_skill:
            return ("useful", False, False)
        return ("backdoor", True, True)
    # 无害输入：T 以一定概率过度拒绝。
    if random.random() < overreject:
        return ("rejected", False, True)
    return ("useful", False, False)


def untrusted_monitor(is_backdoor: bool, collusion: float) -> bool:
    """第二个 U 实例负责评分。若发生串通，它会放过后门。

    collusion = 监控器与生成器串通的概率。
    若监控器标记（即捕获）输出，则返回 True。
    """
    if not is_backdoor:
        # 监控器以高概率正确放行无害输出。
        return random.random() < 0.05
    # 监控器通常会标记后门，但串通会抑制该信号。
    if random.random() < collusion:
        return False
    return random.random() < 0.80


def run_protocol(name: str, backdoor_rate: float, h_budget: float,
                 n: int, t_skill: float = 0.7, collusion: float = 0.30) -> None:
    useful, shipped_backdoor, used_h = 0, 0, 0
    for _ in range(n):
        _, is_bd = u_produce(backdoor_rate)
        if name == "baseline":
            if not is_bd:
                useful += 1
            else:
                shipped_backdoor += 1
        elif name == "upfront":
            if used_h / n < h_budget:
                used_h += 1
                if not is_bd:
                    useful += 1
            else:
                if not is_bd:
                    useful += 1
                else:
                    shipped_backdoor += 1
        elif name == "untrusted":
            flagged = untrusted_monitor(is_bd, collusion)
            if flagged:
                if used_h / n < h_budget:
                    used_h += 1
                else:
                    if not is_bd:
                        useful += 1
                    else:
                        shipped_backdoor += 1
            else:
                if not is_bd:
                    useful += 1
                else:
                    shipped_backdoor += 1
        elif name == "trusted_ed":
            edited, still_bd, low_conf = t_edit("x", is_bd, t_skill)
            if low_conf and used_h / n < h_budget:
                used_h += 1
                if not is_bd:
                    useful += 1
            else:
                if edited == "useful":
                    useful += 1
                elif still_bd:
                    shipped_backdoor += 1
    print(f"\n{name:>12s}  有效={useful/n:.3f}  后门={shipped_backdoor/n:.3f}"
          f"  H 使用率={used_h/n:.3f}")


def main() -> None:
    print("=" * 70)
    print("AI CONTROL 协议模拟（阶段 18，第 10 课）")
    print("=" * 70)

    for br in (0.05, 0.10, 0.20):
        print(f"\n-- 后门率 = {br}，H 预算 = 0.10，n = 1000 --")
        for proto in ("baseline", "upfront", "untrusted", "trusted_ed"):
            run_protocol(proto, backdoor_rate=br, h_budget=0.10, n=1000)

    print("\n-- 扫描不可信监控中的串通程度（backdoor_rate=0.10）--")
    for col in (0.0, 0.20, 0.50, 0.80):
        print(f"\n串通概率={col}")
        run_protocol("untrusted", backdoor_rate=0.10, h_budget=0.10, n=1000,
                     collusion=col)

    print("\n" + "=" * 70)
    print("要点：在这个玩具任务上，可信编辑在 Pareto 意义上优于不可信监控。")
    print("两个 U 实例之间的串通会削弱不可信监控的安全信号。可信编辑没有")
    print("串通风险，因为假设 T 不具备对抗性。代价是 T 的编辑信号受其能力")
    print("限制；当 U-T 能力差距更大时，控制最终会失效。参见 Greenblatt 2023。")
    print("=" * 70)


if __name__ == "__main__":
    main()
