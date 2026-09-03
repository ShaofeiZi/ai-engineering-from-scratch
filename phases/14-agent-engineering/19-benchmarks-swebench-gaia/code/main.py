"""仿 SWE-bench 的玩具评测框架，附带一个仿 GAIA 的难度分类器。

SWE-bench：带 FAIL_TO_PASS 和 PASS_TO_PASS 门控的 bug-fix 任务。
GAIA：simple-for-humans，hard-for-AI 道按分解深度评分的问题。
两者均为合成数据；目的是让评估器规则具体化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Task:
    tid: str
    description: str
    state_before: dict[str, int]
    patch: Callable[[dict[str, int]], dict[str, int]]
    fail_to_pass: list[tuple[str, Callable[[dict[str, int]], bool]]]
    pass_to_pass: list[tuple[str, Callable[[dict[str, int]], bool]]]


@dataclass
class TaskResult:
    tid: str
    ftp_passed: int
    ftp_total: int
    ptp_passed: int
    ptp_total: int
    resolved: bool


def run_task(task: Task) -> TaskResult:
    state = dict(task.state_before)
    ftp_pre = sum(1 for _, check in task.fail_to_pass if check(state))
    ptp_pre = sum(1 for _, check in task.pass_to_pass if check(state))

    new_state = task.patch(dict(state))

    ftp_post = sum(1 for _, check in task.fail_to_pass if check(new_state))
    ptp_post = sum(1 for _, check in task.pass_to_pass if check(new_state))

    ftp_fixed = ftp_post - ftp_pre
    ptp_broke = ptp_pre - ptp_post
    resolved = (ftp_post == len(task.fail_to_pass)) and (ptp_broke == 0)

    return TaskResult(
        tid=task.tid,
        ftp_passed=ftp_post, ftp_total=len(task.fail_to_pass),
        ptp_passed=ptp_post, ptp_total=len(task.pass_to_pass),
        resolved=resolved,
    )


def gaia_level(question: str) -> int:
    steps = sum(1 for w in question.lower().split()
                if w in {"then", "after", "finally", "next", "and"}) + 1
    modalities = sum(word in question.lower() for word in
                     ("image", "video", "audio", "pdf", "chart", "graph"))
    tools = sum(word in question.lower() for word in
                ("search", "look up", "find", "visit", "extract"))
    score = steps + modalities + tools
    if score <= 2:
        return 1
    if score <= 5:
        return 2
    return 3


def swe_demo() -> None:
    print("-" * 70)
    print("SWE-bench-style 评测框架（FAIL_TO_PASS + PASS_TO_PASS）")
    print("-" * 70)

    tasks = [
        Task(
            tid="t001",
            description="fix off-by-one in counter",
            state_before={"counter": 0, "multiplier": 2},
            patch=lambda s: {**s, "counter": s["counter"] + 1},
            fail_to_pass=[("counter > 0", lambda s: s["counter"] > 0)],
            pass_to_pass=[("multiplier unchanged", lambda s: s["multiplier"] == 2)],
        ),
        Task(
            tid="t002",
            description="fix multiplier regression",
            state_before={"counter": 1, "multiplier": 0},
            patch=lambda s: {**s, "multiplier": 2},
            fail_to_pass=[("multiplier > 0", lambda s: s["multiplier"] > 0)],
            pass_to_pass=[("counter unchanged", lambda s: s["counter"] == 1)],
        ),
        Task(
            tid="t003",
            description="agent overreaches and breaks a passing test",
            state_before={"counter": 1, "multiplier": 2, "flag": True},
            patch=lambda s: {**s, "counter": 2, "flag": False},
            fail_to_pass=[("counter > 1", lambda s: s["counter"] > 1)],
            pass_to_pass=[("flag stays true", lambda s: s["flag"]),
                          ("multiplier unchanged", lambda s: s["multiplier"] == 2)],
        ),
    ]

    resolved_count = 0
    for task in tasks:
        result = run_task(task)
        print(f"  {result.tid}：{task.description}")
        print(f"    FAIL_TO_PASS：{result.ftp_passed}/{result.ftp_total}")
        print(f"    PASS_TO_PASS：{result.ptp_passed}/{result.ptp_total}")
        print(f"    已解决：     {result.resolved}")
        if result.resolved:
            resolved_count += 1
    print(f"\n解决率：{resolved_count}/{len(tasks)}")


def gaia_demo() -> None:
    print("\n" + "-" * 70)
    print("GAIA-style 难度分类器")
    print("-" * 70)
    questions = [
        "What is the capital of France?",
        "Search for the Wikipedia article on ReAct and extract the first author.",
        "Visit the arXiv listing for ReAct, find the GitHub linked in the PDF, "
        "then count the open issues with label 'bug' and return the ratio "
        "of bugs to total issues as a decimal.",
    ]
    for q in questions:
        level = gaia_level(q)
        print(f"  [级别 {level}] {q[:70]}")


def main() -> None:
    print("=" * 70)
    print("基准测试：SWE-bench、GAIA — 第 14 阶段，第 19 课")
    print("=" * 70)
    swe_demo()
    gaia_demo()
    print()
    print("SWE-bench：基于补丁并由单元测试门禁。已验证，消除歧义。")
    print("GAIA：深度 + 模态 + 工具 -> 难度级别。")
    print("同时报告基准测试分数以及 Verified/人工审核分数。")


if __name__ == "__main__":
    main()
