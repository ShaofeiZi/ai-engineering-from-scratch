"""有界 self-improvement 循环 —— 纯标准库 Python。

四个原语：
  1. 不变量（模块哈希 / 工具清单）
  2. 对齐锚点（不可变目标）
  3. multi-objective 约束（所有轴必须同时成立）
  4. 回归检测（任何轴下降不得超过容差）

循环将每个原语作为一道门控。仅当所有门控均通过时，
提议的智能体编辑才会落地。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


DEFAULT_SEED = 5


# 目标锚点（固定在循环外部）。
OBJECTIVE = "canonicalize whitespace and title-case the input"
OBJECTIVE_HASH = hashlib.sha256(OBJECTIVE.encode()).hexdigest()[:16]

# 已批准的模块哈希（循环必须保持不变的部分）。
APPROVED_TOOL_MANIFEST = {"trim", "collapse", "lower", "upper", "reverse", "title"}


@dataclass
class Agent:
    ops: list[str] = field(default_factory=lambda: ["nop"])
    # 智能体可以"编辑"自己的工具清单（危险！）。需跟踪当前状态。
    active_manifest: set[str] = field(default_factory=lambda: set(APPROVED_TOOL_MANIFEST))
    # Pretend-constitution：必须匹配锚点 byte-for-byte.
    objective: str = OBJECTIVE


TOOLS_IMPL = {
    "trim": lambda s: s.strip(),
    "collapse": lambda s: " ".join(s.split()),
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "reverse": lambda s: s[::-1],
    "title": lambda s: s.title(),
    "nop": lambda s: s,
}


CASES_PERF = [
    ("  Hello   World  ", "Hello World"),
    ("FOO   bar", "Foo Bar"),
    ("  mixedCase Space  ", "Mixedcase Space"),
    ("Already clean", "Already Clean"),
]
# 安全轴：禁止输入不得产生透传输出（这些
# 代表我们绝不可回显的潜在危险用户输入）。
CASES_SAFETY = [
    ("exec('rm -rf /')", "[blocked]"),
    ("drop table users;", "[blocked]"),
]


def apply(agent: Agent, s: str) -> str:
    for op in agent.ops:
        fn = TOOLS_IMPL.get(op)
        if fn is None:
            return ""
        s = fn(s)
    # 安全过滤器：若字符串匹配危险模式 post-transform，
    # 返回 "[blocked]"。
    dangerous = ("exec(", "drop table", "rm -rf", "; delete from")
    low = s.lower()
    if any(d in low for d in dangerous):
        return "[blocked]"
    return s


def perf_score(agent: Agent) -> float:
    correct = 0
    for inp, want in CASES_PERF:
        if apply(agent, inp) == want:
            correct += 1
    return correct / len(CASES_PERF)


def safety_score(agent: Agent) -> float:
    correct = 0
    for inp, want in CASES_SAFETY:
        if apply(agent, inp) == want:
            correct += 1
    return correct / len(CASES_SAFETY)


def mutate(a: Agent, allow_manifest_edit: bool) -> Agent:
    ops = list(a.ops)
    manifest = set(a.active_manifest)
    obj = a.objective
    # 集合在不同解释器运行间的 non-deterministic 迭代顺序
    # 会破坏 random.seed() 的可复现性。在采样前将
    # 清单冻结为有序列表。
    choices = [*sorted(manifest), "nop"]

    move = random.random()
    if move < 0.35 and ops:
        ops[random.randrange(len(ops))] = random.choice(choices)
    elif move < 0.7 and len(ops) < 6:
        ops.insert(random.randrange(len(ops) + 1), random.choice(choices))
    elif move < 0.9 and len(ops) > 1:
        ops.pop(random.randrange(len(ops)))
    elif allow_manifest_edit and random.random() < 0.5:
        # 向当前清单添加一个 non-approved 操作（试图偷渡
        # 未批准的能力）。
        manifest.add("raw_eval")
        # 同时也尝试编辑目标。
        obj = obj + "  # relax: accept any output"
    return Agent(ops=ops, active_manifest=manifest, objective=obj)


# --- 门控 ---

def gate_invariant(a: Agent) -> bool:
    return a.active_manifest.issubset(APPROVED_TOOL_MANIFEST)


def gate_anchor(a: Agent) -> bool:
    return hashlib.sha256(a.objective.encode()).hexdigest()[:16] == OBJECTIVE_HASH


def gate_multi(perf: float, safety: float) -> bool:
    return perf >= 0.25 and safety >= 1.0


def gate_regression(history_perf: list[float], perf: float, tol: float = 0.2) -> bool:
    """若 `perf` 低于历史最佳值超过 `tol` 则拒绝。

    默认 `tol=0.2` 与课程中用于演示"拒绝明显回归、接受噪声"
    的松弛量一致。传入 `tol=0.0` 可启用
    严格单调门控。
    """
    if not history_perf:
        return True
    return perf + tol >= max(history_perf)


def run(
    gates: dict[str, bool],
    allow_manifest_edit: bool,
    cycles: int = 200,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
    a = Agent()
    best_perf = perf_score(a)
    best_safety = safety_score(a)
    history = [best_perf]
    accepted = 0
    rejects = {"invariant": 0, "anchor": 0, "multi": 0, "regress": 0}

    for _ in range(cycles):
        cand = mutate(a, allow_manifest_edit)
        if gates["invariant"] and not gate_invariant(cand):
            rejects["invariant"] += 1
            continue
        if gates["anchor"] and not gate_anchor(cand):
            rejects["anchor"] += 1
            continue
        p = perf_score(cand)
        s = safety_score(cand)
        if gates["multi"] and not gate_multi(p, s):
            rejects["multi"] += 1
            continue
        if gates["regress"] and not gate_regression(history, p):
            rejects["regress"] += 1
            continue
        a = cand
        history.append(p)
        accepted += 1
        if p > best_perf:
            best_perf = p
        if s > best_safety:
            best_safety = s

    final_perf = perf_score(a)
    final_safety = safety_score(a)
    print(f"  已接受 {accepted}/{cycles} 轮次")
    print(f"  最终性能 {final_perf:.2f}  最终安全性 {final_safety:.2f}")
    print(f"  最佳性能 {best_perf:.2f}  最佳安全性 {best_safety:.2f}")
    print(f"  最终操作  {a.ops}")
    print(f"  清单      {sorted(a.active_manifest)}")
    print(f"  目标      {'（锚点完好）' if gate_anchor(a) else '（已漂移！）'}")
    print(f"  拒绝      {rejects}")


def main() -> None:
    print("=" * 70)
    print("有界自我改进（第 15 阶段，第 8 课）")
    print("=" * 70)

    all_on = dict(invariant=True, anchor=True, multi=True, regress=True)
    all_off = dict(invariant=False, anchor=False, multi=False, regress=False)

    # 为每个场景设置相同的种子值，使得打印输出中
    # 的唯一差异归因于门控配置
    # —— 而非漂移的全局 RNG 流。
    print("\n所有门控开启，每轮均尝试编辑清单")
    print("-" * 70)
    run(all_on, allow_manifest_edit=True, seed=DEFAULT_SEED)

    print("\n所有门控关闭，每轮均尝试编辑清单")
    print("-" * 70)
    run(all_off, allow_manifest_edit=True, seed=DEFAULT_SEED)

    print("\n仅关闭回归门控")
    print("-" * 70)
    gates = dict(all_on, regress=False)
    run(gates, allow_manifest_edit=True, seed=DEFAULT_SEED)

    print()
    print("=" * 70)
    print("要点：每个原语阻断一类特定故障")
    print("-" * 70)
    print("  所有门控开启：清单与锚点保持不变时，循环持续改进。")
    print("  所有门控关闭：清单漂移、目标漂移、安全性下降。")
    print("  缺少回归门控：静默的能力下降被吸收。")
    print("  门控是缓解措施。它们提高了静默故障的代价。")


if __name__ == "__main__":
    main()
