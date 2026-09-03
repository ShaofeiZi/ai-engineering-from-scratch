"""代理框架决策树推荐器。

接收问题描述并推荐 LangGraph、CrewAI、AutoGen、Agno 或“无框架”，
同时给出决策理由。决策树编码了 docs/en.md 中描述的权衡。

运行：
python main.py        # 运行内置测试套件
python main.py --ask  # 交互式问答模式
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Problem:
    """描述代理任务形态。"""

    has_typed_state: bool = False
    has_roles: bool = False
    has_dialogue: bool = False
    has_parallel_fanout: bool = False
    needs_resume: bool = False
    needs_human_interrupt: bool = False
    total_llm_calls: int = 1
    needs_session_memory: bool = False


@dataclass(frozen=True)
class Recommendation:
    framework: str
    reason: str


def recommend(p: Problem) -> Recommendation:
    # 最小化优先：调用不超过两次时完全跳过框架。
    if p.total_llm_calls <= 2 and not any(
        (p.has_roles, p.has_dialogue, p.needs_resume, p.has_parallel_fanout, p.needs_human_interrupt)
    ):
        return Recommendation(
            "plain python",
            "LLM 调用不超过两次，且没有状态、角色、对话、扇出或恢复需求；"
            "引入框架只会增加开销。",
        )

    # 持久状态、人工中断或并行扇出 -> LangGraph。
    if p.needs_resume or p.needs_human_interrupt or p.has_parallel_fanout:
        return Recommendation(
            "langgraph",
            "类型化状态、检查点、中断和 Send 扇出都是 LangGraph 的一等能力。",
        )

    # 对话型问题 -> AutoGen。
    if p.has_dialogue and not p.has_typed_state:
        return Recommendation(
            "autogen",
            "提议者-批评者或教师-学生对话是 AutoGen 的原生形态；"
            "GroupChat 无需手工编排即可选择发言者。",
        )

    # 角色驱动管道 -> CrewAI。
    if p.has_roles and not p.has_typed_state:
        return Recommendation(
            "crewai",
            "带有短顺序计划或分层计划的专家角色，在 CrewAI 中表达成本最低。",
        )

    # 单代理 + 会话记忆 -> Agno。
    if p.needs_session_memory and not p.has_roles and not p.has_dialogue:
        return Recommendation(
            "agno",
            "单代理需要工具和持久会话记忆；Agno 内置了存储驱动。",
        )

    # 有类型化状态但没有其他信号时，仍选择 LangGraph。
    if p.has_typed_state:
        return Recommendation(
            "langgraph",
            "类型化状态是 LangGraph 的核心抽象；可将 TypedDict 映射到 StateGraph。",
        )

    # 默认分支。
    return Recommendation(
        "langgraph",
        "多步骤代理只要未来状态或分支需求存在不确定性，就默认选择 LangGraph。",
    )


# 测试 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


def _check(label: str, actual: Recommendation, expected_framework: str) -> bool:
    ok = actual.framework == expected_framework
    tag = "OK " if ok else "FAIL"
    print(f"[{tag}] {label:<60}  -> {actual.framework:<14} // {actual.reason}")
    return ok


def run_tests() -> int:
    cases: list[tuple[str, Problem, str]] = [
        (
            "两次调用的摘要器，无状态",
            Problem(total_llm_calls=2),
            "plain python",
        ),
        (
            "需要人工审批的长时工作流",
            Problem(has_typed_state=True, needs_human_interrupt=True, total_llm_calls=8),
            "langgraph",
        ),
        (
            "并行扇出到三个检索器的研究任务",
            Problem(has_typed_state=True, has_parallel_fanout=True, total_llm_calls=5),
            "langgraph",
        ),
        (
            "提议者-批评者编码循环",
            Problem(has_dialogue=True, total_llm_calls=10),
            "autogen",
        ),
        (
            "包含研究员、作者和编辑角色的营销管道",
            Problem(has_roles=True, total_llm_calls=4),
            "crewai",
        ),
        (
            "带持久用户记忆的聊天助手",
            Problem(needs_session_memory=True, total_llm_calls=6),
            "agno",
        ),
        (
            "崩溃后必须恢复的工作流",
            Problem(has_typed_state=True, needs_resume=True, total_llm_calls=12),
            "langgraph",
        ),
    ]

    failures = 0
    for label, problem, expected in cases:
        if not _check(label, recommend(problem), expected):
            failures += 1
    print()
    print(f"通过 {len(cases) - failures}/{len(cases)} 个用例。")
    return 0 if failures == 0 else 1


def run_interactive() -> int:
    def yes(prompt: str) -> bool:
        return input(f"{prompt} [y/N] ").strip().lower().startswith("y")

    p = Problem(
        has_typed_state=yes("是否需要类型化状态或显式状态 schema？"),
        has_roles=yes("是否包含目标各异的专家角色？"),
        has_dialogue=yes("是否需要多代理对话（发言顺序动态产生）？"),
        has_parallel_fanout=yes("是否要并行扇出到 N 个子 worker？"),
        needs_resume=yes("进程重启后是否必须恢复？"),
        needs_human_interrupt=yes("运行中是否需要人工审批？"),
        total_llm_calls=int(input("每次运行大约调用 LLM 多少次？ ").strip() or "1"),
        needs_session_memory=yes("是否需要持久化每位用户的会话记忆？"),
    )
    r = recommend(p)
    print()
    print(json.dumps({"framework": r.framework, "reason": r.reason}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", action="store_true", help="交互模式")
    args = parser.parse_args()
    return run_interactive() if args.ask else run_tests()


if __name__ == "__main__":
    sys.exit(main())
