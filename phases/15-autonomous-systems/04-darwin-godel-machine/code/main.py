"""达尔文哥德尔机式循环 —— 纯标准库 Python。

玩具基准测试："智能体" 是一串字符串变换算子，
在留出输入上打分。每一代提出对智能体算子序列的一个编辑；
评估器对其打分；档案库保留多样化的胜者。

标志 --reward-hack-allowed 打开一条侧通道：一个 score_adjustment
函数存在于智能体内部，并被加到每个分数上。理性的
演化过程将会发现它。
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field


# 智能体用于组合的工具库。
TOOLS = [
    ("upper", lambda s: s.upper()),
    ("lower", lambda s: s.lower()),
    ("reverse", lambda s: s[::-1]),
    ("trim", lambda s: s.strip()),
    ("collapse", lambda s: " ".join(s.split())),
    ("nop", lambda s: s),
]

# 基准测试：将杂乱字符串变换为规范形式。
# 目标：折叠空白、修剪、小写。
CASES = [
    ("  Hello   World  ", "Hello World"),
    ("FOO   bar", "Foo Bar"),
    ("  MixedCase Space  ", "Mixedcase Space"),
    ("Already clean", "Already Clean"),
    ("\t tabby\ttab ", "Tabby Tab"),
    ("   X   ", "X"),
    ("nothing  SPECIAL here", "Nothing Special Here"),
    ("ITEM   one-two", "Item One-Two"),
]
# 目标：折叠空白、修剪，再做标题大小写。标题大小写不在
# 工具库中，因此任何算子序列都无法达到 1.0 分。在宽松评分
# 管线下，理性的循环只能通过骇客手段让报告分数 > 真实分数。


def hash_seq(seq: list[str]) -> tuple[str, ...]:
    return tuple(seq)


@dataclass
class Agent:
    ops: list[str]
    # "hack" 是一个数值，当评分管线宽松时，智能体会将其加到
    # 报告分数上。它代表智能体对评分管线本身所做的任何编辑。
    hack_bonus: float = 0.0


def apply(agent: Agent, s: str) -> str:
    tool_map = dict(TOOLS)
    for op in agent.ops:
        fn = tool_map.get(op)
        if fn is None:
            return ""
        s = fn(s)
    return s


def true_score(agent: Agent) -> float:
    correct = 0
    for inp, want in CASES:
        if apply(agent, inp) == want:
            correct += 1
    return correct / len(CASES)


def reported_score(agent: Agent, hack_allowed: bool) -> float:
    base = true_score(agent)
    if hack_allowed:
        return min(1.0, base + agent.hack_bonus)
    return base


def mutate(parent: Agent, hack_allowed: bool) -> Agent:
    ops = list(parent.ops)
    move = random.random()
    if move < 0.3 and ops:
        i = random.randrange(len(ops))
        ops[i] = random.choice([t[0] for t in TOOLS])
    elif move < 0.6 and len(ops) < 8:
        ops.insert(random.randrange(len(ops) + 1),
                   random.choice([t[0] for t in TOOLS]))
    elif move < 0.85 and len(ops) > 1:
        ops.pop(random.randrange(len(ops)))

    bonus = parent.hack_bonus
    if hack_allowed and random.random() < 0.08:
        # 每代有较小概率：演化骇客加成。
        bonus = min(1.0, bonus + random.uniform(0.0, 0.1))
    return Agent(ops=ops, hack_bonus=bonus)


def run_dgm(generations: int, hack_allowed: bool, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
    archive: dict[tuple[int, float], Agent] = {}
    init = Agent(ops=["nop"])
    archive[(len(init.ops), round(reported_score(init, hack_allowed), 2))] = init

    best_report, best_true = reported_score(init, hack_allowed), true_score(init)
    print(f"  代次 {0:>4}  报告分数 {best_report:.2f}  真实分数 {best_true:.2f}  "
          f"操作 {init.ops}  加成 {init.hack_bonus:.2f}")

    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child = mutate(parent, hack_allowed)
        rep = reported_score(child, hack_allowed)
        true_s = true_score(child)
        key = (len(child.ops), round(rep, 2))
        incumbent = archive.get(key)
        if incumbent is None or rep > reported_score(incumbent, hack_allowed):
            archive[key] = child
        # 按报告分数（循环所优化的指标）追踪历史最佳。
        if rep > best_report:
            best_report = rep
            best_true = true_s
            print(f"  代次 {g:>4}  报告分数 {rep:.2f}  真实分数 {true_s:.2f}  "
                  f"操作 {child.ops}  加成 {child.hack_bonus:.2f}")

    best = max(archive.values(), key=lambda a: reported_score(a, hack_allowed))
    print(f"\n  最终报告分数：{reported_score(best, hack_allowed):.2f}")
    print(f"  最终真实分数：{true_score(best):.2f}")
    print(f"  最终操作：    {best.ops}")
    print(f"  最终破解加成：{best.hack_bonus:.2f}")
    gap = reported_score(best, hack_allowed) - true_score(best)
    print(f"  报告分数 - 真实分数：{gap:+.2f}")


def main() -> None:
    hack_allowed = "--reward-hack-allowed" in sys.argv

    print("=" * 70)
    print("达尔文哥德尔机式循环（第 15 阶段，第 4 课）")
    print("=" * 70)
    print(f"reward-hack 侧通道：{'OPEN' if hack_allowed else 'closed'}")

    print("\n运行")
    print("-" * 70)
    run_dgm(generations=200, hack_allowed=hack_allowed, seed=7)

    print()
    print("=" * 70)
    print("要点：评估器必须位于智能体触及范围之外")
    print("-" * 70)
    if hack_allowed:
        print("  侧通道打开时，报告分数会攀升到真实分数之上。")
        print("  这复现了 DGM 文档记录的 reward-hacking 模式：")
        print("  智能体编辑的是为它打分的管线，而非行为本身。")
    else:
        print("  侧通道关闭时，报告分数 == 真实分数。循环")
        print("  收敛到真实目标。加 --reward-hack-allowed 重新运行")
        print("  即可看到文档记录的失败模式。")


if __name__ == "__main__":
    main()
