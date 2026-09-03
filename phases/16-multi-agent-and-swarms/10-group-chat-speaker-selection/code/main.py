"""带发言者选择的群聊：AutoGen GroupChat 的微型实现。

包含三个 Agent（coder、reviewer、manager）、两种 selector 变体
（轮询、LLM 模拟），并以 TERMINATE token 作为停止条件。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Msg:
    speaker: str
    content: str


@dataclass
class Agent:
    name: str
    role: str
    policy: Callable[[list[Msg]], str]


def coder_policy(pool: list[Msg]) -> str:
    recent = [m for m in pool[-3:] if m.speaker != "coder"]
    last = recent[-1].content if recent else ""
    if _contains_any(last, "review", "fix", "审查", "修复"):
        return "修改后的代码：return a + b"
    if not any(m.speaker == "coder" for m in pool):
        return "初始代码：return a - b（有缺陷）"
    return "TERMINATE"


def reviewer_policy(pool: list[Msg]) -> str:
    last_coder = next((m for m in reversed(pool) if m.speaker == "coder"), None)
    if last_coder is None:
        return "等待代码"
    if "a - b" in last_coder.content:
        return "审查：发现缺陷，求和必须是 a+b，请修复"
    if "a + b" in last_coder.content:
        return "审查：已批准"
    return "审查：不明确"


def manager_policy(pool: list[Msg]) -> str:
    approvals = [
        m
        for m in pool
        if m.speaker == "reviewer" and _is_approval(m.content)
    ]
    if approvals:
        return "TERMINATE"
    return "manager：继续工作"


AGENTS: dict[str, Agent] = {
    "coder": Agent("coder", "编写代码", coder_policy),
    "reviewer": Agent("reviewer", "审查代码", reviewer_policy),
    "manager": Agent("manager", "推动工作进展", manager_policy),
}


def _contains_any(text: str, *markers: str) -> bool:
    normalized = text.casefold()
    return any(marker.casefold() in normalized for marker in markers)


def _is_approval(text: str) -> bool:
    if _contains_any(text, "not approved", "unapproved", "未批准", "不批准", "未通过", "不通过"):
        return False
    return _contains_any(text, "approved", "批准", "通过")


def round_robin_selector(pool: list[Msg], team: dict[str, Agent]) -> Optional[str]:
    names = list(team.keys())
    if not pool:
        return names[0]
    idx = (names.index(pool[-1].speaker) + 1) % len(names)
    return names[idx]


def llm_style_selector(pool: list[Msg], team: dict[str, Agent]) -> Optional[str]:
    """模拟的 LLM selector：根据近期上下文关键词进行选择。
    真实实现会调用 LLM 并传入近期消息池。"""
    if not pool:
        return "manager"
    last = pool[-1]
    if last.speaker == "coder":
        return "reviewer"
    if last.speaker == "reviewer":
        if _is_approval(last.content):
            return "manager"
        return "coder"
    if last.speaker == "manager":
        return "coder"
    return None


def run_groupchat(
    team: dict[str, Agent],
    selector: Callable[[list[Msg], dict[str, Agent]], Optional[str]],
    max_rounds: int,
    label: str,
) -> list[Msg]:
    print(f"\n=== {label} ===")
    pool: list[Msg] = []
    trace: list[str] = []
    for _ in range(max_rounds):
        nxt = selector(pool, team)
        if nxt is None:
            break
        trace.append(nxt)
        agent = team[nxt]
        content = agent.policy(pool)
        pool.append(Msg(speaker=nxt, content=content))
        print(f"  [{nxt:8s}]: {content}")
        if content.strip().endswith("TERMINATE"):
            break
    print(f"  Selector 轨迹：{trace}")
    print(f"  使用轮数：{len(pool)}")
    return pool


def speaker_counts(pool: list[Msg]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in pool:
        counts[m.speaker] = counts.get(m.speaker, 0) + 1
    return counts


def main() -> None:
    print("带发言者选择的群聊 — AutoGen GroupChat 结构")
    print("-" * 62)

    p_rr = run_groupchat(AGENTS, round_robin_selector, max_rounds=8, label="轮询")
    print(f"  发言次数：{speaker_counts(p_rr)}")

    p_llm = run_groupchat(AGENTS, llm_style_selector, max_rounds=8, label="LLM 风格（上下文感知）")
    print(f"  发言次数：{speaker_counts(p_llm)}")

    print("\n观察结果：")
    print("  - 轮询不考虑上下文，让每个 Agent 获得同等的发言机会。")
    print("  - LLM 风格按上下文路由，例如 reviewer 只会在 coder 之后发言。")
    print("  - 两者都在遇到 TERMINATE token 或达到 max_rounds 时终止。")


if __name__ == "__main__":
    main()
