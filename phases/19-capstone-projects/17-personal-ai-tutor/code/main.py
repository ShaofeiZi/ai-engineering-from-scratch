"""个人 AI 导师——贝叶斯知识追踪 + 苏格拉底式策略脚手架。

关键架构原语是学习者模型：每次交互后通过贝叶斯知识追踪更新逐概念掌握概率，
再将其输入课程图遍历以选择下一个概念。此脚手架实现 BKT、课程 DAG、
苏格拉底式策略决策，以及模拟的两组学习者研究。

运行：python main.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 贝叶斯知识追踪——经典四参数模型
# ---------------------------------------------------------------------------

@dataclass
class BKTParams:
    p_init: float = 0.2     # 先验知识
    p_learn: float = 0.12   # 每次练习的学习率
    p_slip: float = 0.10    # 未掌握却答对
    p_guess: float = 0.15   # 未掌握但猜对


def bkt_update(mastery: float, correct: bool, p: BKTParams) -> float:
    if correct:
        num = mastery * (1 - p.p_slip)
        denom = num + (1 - mastery) * p.p_guess
    else:
        num = mastery * p.p_slip
        denom = num + (1 - mastery) * (1 - p.p_guess)
    posterior = num / max(denom, 1e-6)
    # 状态转移：从本次交互中学习
    return posterior + (1 - posterior) * p.p_learn


# ---------------------------------------------------------------------------
# 课程图——包含先修关系边的概念 DAG
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    name: str
    prereqs: list[str] = field(default_factory=list)


ALGEBRA = [
    Concept("number_line", []),
    Concept("addition_subtraction", ["number_line"]),
    Concept("multiplication_division", ["addition_subtraction"]),
    Concept("negative_numbers", ["addition_subtraction"]),
    Concept("equality", ["addition_subtraction"]),
    Concept("isolating_variable_one_step", ["equality", "addition_subtraction"]),
    Concept("isolating_variable_two_step", ["isolating_variable_one_step", "multiplication_division"]),
    Concept("distributive_property", ["multiplication_division"]),
    Concept("combining_like_terms", ["addition_subtraction", "distributive_property"]),
    Concept("linear_equations", ["isolating_variable_two_step", "combining_like_terms"]),
    Concept("quadratic_basics", ["linear_equations", "multiplication_division"]),
]


def curriculum_map(concepts: list[Concept]) -> dict[str, Concept]:
    return {c.name: c for c in concepts}


# ---------------------------------------------------------------------------
# 学习者状态——逐概念掌握度 + 历史记录
# ---------------------------------------------------------------------------

@dataclass
class LearnerState:
    learner_id: str
    mastery: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.2))
    history: list[tuple[str, bool]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 概念选择器——选择 (a) 已满足先修要求且 (b) 掌握度较低的下一个概念
# ---------------------------------------------------------------------------

def next_concept(state: LearnerState, cmap: dict[str, Concept],
                 master_threshold: float = 0.85) -> str | None:
    for c in cmap.values():
        if state.mastery[c.name] >= master_threshold:
            continue
        if all(state.mastery[pr] >= master_threshold for pr in c.prereqs):
            return c.name
    return None


# ---------------------------------------------------------------------------
# 苏格拉底式策略——决定搭脚手架、进入下一步还是庆祝
# ---------------------------------------------------------------------------

def socratic_policy(state: LearnerState, concept: str, correct: bool) -> str:
    m = state.mastery[concept]
    if correct and m > 0.8:
        return "celebrate_and_advance"
    if correct:
        return "reinforce_and_next_question"
    if m > 0.5:
        return "hint"
    return "scaffold_from_prereq"


# ---------------------------------------------------------------------------
# 学习者模拟器——难度随掌握度变化的随机游走
# ---------------------------------------------------------------------------

def simulate_answer(learner_knowledge: float, concept_difficulty: float,
                    rng: random.Random) -> bool:
    """模拟学习者是否回答正确。"""
    # 答对概率 = sigmoid（知识水平 - 难度）
    import math
    p = 1 / (1 + math.exp(-(learner_knowledge - concept_difficulty)))
    return rng.random() < p


# ---------------------------------------------------------------------------
# 自适应与基线队列——比较 N 次交互后的学习增益
# ---------------------------------------------------------------------------

def run_adaptive(learner_id: str, inherent_ability: float,
                 cmap: dict[str, Concept], n_turns: int, rng: random.Random) -> LearnerState:
    state = LearnerState(learner_id=learner_id)
    p = BKTParams()
    # 导师上次采取的动作会传递到下一轮，因此 scaffold/hint 会实际降低难度，
    # 而 celebration 会小幅提高掌握度
    last_action: str | None = None
    for _ in range(n_turns):
        concept = next_concept(state, cmap)
        if concept is None:
            break
        difficulty = 0.3 + 0.1 * len(cmap[concept].prereqs)
        # 将上一轮动作应用到“本”轮
        if last_action == "scaffold_from_prereq":
            difficulty -= 0.15    # 从先修知识开始的更简单重试
        elif last_action == "hint":
            difficulty -= 0.08    # 轻度提示
        elif last_action == "celebrate_and_advance":
            # 庆祝会在一轮内提升信心
            state.mastery[concept] = min(1.0, state.mastery[concept] + 0.02)
        # 有效知识水平 = 固有能力 + 掌握度
        ek = inherent_ability + state.mastery[concept] * 1.5
        correct = simulate_answer(ek, difficulty, rng)
        last_action = socratic_policy(state, concept, correct)
        state.history.append((concept, correct))
        state.mastery[concept] = bkt_update(state.mastery[concept], correct, p)
    return state


def run_baseline(learner_id: str, inherent_ability: float,
                 cmap: dict[str, Concept], n_turns: int, rng: random.Random) -> LearnerState:
    """非自适应概念选择（轮询）。掌握度仍通过 BKT 更新，因此两组使用相同的
    学习者模型；只有策略 / 概念选择方式不同。"""
    state = LearnerState(learner_id=learner_id)
    p = BKTParams()
    order = list(cmap.keys())
    for i in range(n_turns):
        concept = order[i % len(order)]
        difficulty = 0.3 + 0.1 * len(cmap[concept].prereqs)
        ek = inherent_ability + state.mastery[concept] * 1.5
        correct = simulate_answer(ek, difficulty, rng)
        state.history.append((concept, correct))
        state.mastery[concept] = bkt_update(state.mastery[concept], correct, p)
    return state


def mastery_sum(state: LearnerState, cmap: dict[str, Concept]) -> float:
    return sum(state.mastery[c] for c in cmap)


def main() -> None:
    cmap = curriculum_map(ALGEBRA)
    rng = random.Random(29)

    print("=== 两周效果研究（模拟）===")
    print(f"课程：{len(cmap)} 个概念")

    adaptive_gains: list[float] = []
    baseline_gains: list[float] = []
    n_learners = 10
    n_turns = 60

    for i in range(n_learners):
        ability = rng.gauss(0.3, 0.4)
        # 配对随机性：两组使用相同的潜在 RNG 流，因此差值衡量的是策略差异，
        # 而非 seed 噪声
        seed = 100 + i
        r_adapt = random.Random(seed)
        r_base = random.Random()
        r_base.setstate(r_adapt.getstate())
        s1 = run_adaptive(f"adapt_{i}", ability, cmap, n_turns, r_adapt)
        s2 = run_baseline(f"base_{i}", ability, cmap, n_turns, r_base)
        adaptive_gains.append(mastery_sum(s1, cmap))
        baseline_gains.append(mastery_sum(s2, cmap))

    def mean(xs): return sum(xs) / len(xs)
    print(f"自适应掌握度总和  均值={mean(adaptive_gains):.2f}")
    print(f"基线掌握度总和    均值={mean(baseline_gains):.2f}")
    delta = mean(adaptive_gains) - mean(baseline_gains)
    print(f"差值（自适应 - 基线）：{n_turns} 轮后为 {delta:+.2f} 个掌握度点")

    print("\n=== 示例轨迹（自适应学习者 0）===")
    state = run_adaptive("demo", 0.3, cmap, 20, random.Random(7))
    seen_concepts = []
    for c, ok in state.history:
        if c not in [x[0] for x in seen_concepts]:
            seen_concepts.append((c, state.mastery[c]))
    for c, m in seen_concepts[:8]:
        print(f"  {c:34s} mastery={m:.2f}")


if __name__ == "__main__":
    main()
