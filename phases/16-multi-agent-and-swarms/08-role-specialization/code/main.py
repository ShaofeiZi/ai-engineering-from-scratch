"""角色专业化：planner、executor、critic、verifier。

构建一个小型 Python 函数。critic（由 LLM 模拟）和 verifier（代码）协同工作，
捕获任何一方单独工作时会漏掉的缺陷。

运行两次：一次使用正确的 executor 输出，一次使用不符合规格的输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Spec:
    task_name: str
    signature: str
    description: str
    tests: list[tuple[tuple, int]]


@dataclass
class Artifact:
    code: str


@dataclass
class CriticReport:
    approved: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class VerifierReport:
    passed: bool
    failures: list[str] = field(default_factory=list)


def planner(user_wish: str) -> Spec:
    """根据高层次愿望生成结构化规格。"""
    return Spec(
        task_name="add_two",
        signature="add_two(a: int, b: int) -> int",
        description=user_wish,
        tests=[((1, 2), 3), ((10, 20), 30), ((-5, 5), 0)],
    )


def executor_correct(spec: Spec) -> Artifact:
    return Artifact(code="def add_two(a, b):\n    return a + b\n")


def executor_buggy(spec: Spec) -> Artifact:
    return Artifact(code="def add_two(a, b):\n    return a * b\n")


def critic(spec: Spec, art: Artifact) -> CriticReport:
    """LLM 风格的审查。通过模式匹配发现常见问题，但可能被看似合理、
    实际语义错误的代码骗过。"""
    notes: list[str] = []
    if "def" not in art.code:
        notes.append("缺少 def 语句")
    if "return" not in art.code:
        notes.append("缺少 return")
    if spec.task_name not in art.code:
        notes.append(f"函数名与规格“{spec.task_name}”不匹配")
    approved = not notes
    return CriticReport(approved=approved, notes=notes)


def verifier(spec: Spec, art: Artifact) -> VerifierReport:
    """在沙箱命名空间中运行代码并执行测试。结果是确定性的。"""
    ns: dict = {}
    try:
        exec(art.code, ns, ns)
    except Exception as e:
        return VerifierReport(passed=False, failures=[f"exec 错误：{e}"])
    fn = ns.get(spec.task_name)
    if not callable(fn):
        return VerifierReport(passed=False, failures=[f"未生成可调用对象“{spec.task_name}”"])
    failures: list[str] = []
    for args, expected in spec.tests:
        try:
            got = fn(*args)
        except Exception as e:
            failures.append(f"调用 {args} 时抛出 {e}")
            continue
        if got != expected:
            failures.append(f"调用 {args}：预期 {expected}，实际得到 {got}")
    return VerifierReport(passed=not failures, failures=failures)


def run_pipeline(user_wish: str, executor, label: str) -> None:
    print(f"\n=== {label} ===")
    spec = planner(user_wish)
    print(f"  [planner] 规格：{spec.signature}，包含 {len(spec.tests)} 个测试")
    art = executor(spec)
    print(f"  [executor] 生成内容：\n    {art.code.replace(chr(10), chr(10)+'    ')}")
    crep = critic(spec, art)
    print(f"  [critic] approved={crep.approved}，笔记={crep.notes}")
    vrep = verifier(spec, art)
    print(f"  [verifier] passed={vrep.passed}，失败项={vrep.failures}")
    if crep.approved and vrep.passed:
        print("  结果：可以发布。")
    elif not vrep.passed:
        print("  结果：verifier 阻止发布（确定性捕获）。")
    elif not crep.approved:
        print("  结果：critic 阻止发布（主观判断捕获）。")


def main() -> None:
    print("角色专业化流水线 — planner、executor、critic、verifier")
    print("-" * 70)

    run_pipeline(
        "返回两个整数之和的函数。",
        executor_correct,
        "正确的 executor 输出",
    )

    run_pipeline(
        "返回两个整数之和的函数。",
        executor_buggy,
        "有缺陷的 executor 输出（看似合理，但运行时失败）",
    )

    print("\n关键洞察：critic 会放过有缺陷的代码，因为它看起来没有问题。")
    print("只有 verifier，也就是确定性测试执行，才能捕获语义缺陷。")
    print("全 LLM 流水线（没有 verifier）会把缺陷发布出去。这是典型的 MAST 失败模式。")


if __name__ == "__main__":
    main()
