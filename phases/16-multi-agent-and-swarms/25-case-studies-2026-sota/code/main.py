"""案例映射器：为拟议设计选择最接近的 2026 年参考案例。

仅使用 stdlib。根据设计属性，以脚本方式映射到三个案例之一（Anthropic Research、
MetaGPT/ChatDev、OpenClaw/Moltbook），并推荐框架。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Design:
    name: str
    task_type: str          # "research" | "engineering" | "population" | "automation"
    n_agents_expected: int
    verification_required: bool
    runtime_duration_hours: float
    roles_distinct: bool
    user_facing_network: bool


CASES = {
    "anthropic_research": {
        "name": "Anthropic Research（supervisor-worker）",
        "patterns": ["全新上下文的子 Agent", "orchestrator 综合",
                     "rainbow 部署", "验证角色"],
        "framework": "Anthropic Claude Agent SDK 或 LangGraph",
        "citation": "https://www.anthropic.com/engineering/multi-agent-research-system",
    },
    "metagpt_chatdev": {
        "name": "MetaGPT / ChatDev（SOP 角色分解）",
        "patterns": ["角色 prompt 编码 SOP", "结构化产物 handoff",
                     "通过通信消除幻觉", "大规模 DAG 路由"],
        "framework": "CrewAI 或 MetaGPT 参考实现",
        "citation": "arXiv:2308.00352 (MetaGPT), arXiv:2307.07924 (ChatDev), arXiv:2406.07155 (MacNet)",
    },
    "openclaw_moltbook": {
        "name": "OpenClaw / Moltbook（群体规模底座）",
        "patterns": ["本地 ReAct 循环", "Agent 间社交图",
                     "涌现经济", "prompt 注入威胁模型"],
        "framework": "自定义底座 + MCP + A2A",
        "citation": "https://en.wikipedia.org/wiki/OpenClaw",
    },
}

FRAMEWORK_LANDSCAPE = [
    ("LangGraph", "生产可用", "结构化图 + checkpoint + HITL"),
    ("CrewAI", "生产可用", "基于角色的团队；顺序/分层"),
    ("AG2", "社区维护", "GroupChat + 发言者选择"),
    ("Microsoft Agent Framework", "RC（2026 年 2 月）", "编排模式 + 企业能力"),
    ("OpenAI Agents SDK", "生产可用", "Swarm 后继者；工具返回 handoff"),
    ("Google ADK", "生产可用（2025 年 4 月）", "A2A 原生；Google Cloud"),
    ("Anthropic Claude Agent SDK", "生产可用", "Agent + Research 扩展"),
]


def map_to_case(d: Design) -> str:
    if d.task_type == "population" or d.user_facing_network:
        return "openclaw_moltbook"
    if d.task_type == "engineering" or d.roles_distinct:
        return "metagpt_chatdev"
    if d.task_type == "research":
        return "anthropic_research"
    if d.verification_required and d.runtime_duration_hours >= 1:
        return "anthropic_research"
    return "anthropic_research"


def print_case(key: str) -> None:
    case = CASES[key]
    print(f"\n  最接近的案例：{case['name']}")
    print("  可借鉴的模式：")
    for p in case["patterns"]:
        print(f"    - {p}")
    print(f"  推荐框架：{case['framework']}")
    print(f"  引用：{case['citation']}")


def print_landscape() -> None:
    print("\n" + "=" * 78)
    print("框架概览 — 2026 年 4 月")
    print("=" * 78)
    print(f"  {'框架':30s} {'状态':22s} {'最适合':30s}")
    for name, status, best_for in FRAMEWORK_LANDSCAPE:
        print(f"  {name:30s} {status:22s} {best_for:30s}")
    print("\n  每个主流框架都支持 MCP；大多数支持 A2A。")
    print("  协议兼容性已不再是差异点，handoff 语义才是。")


def main() -> None:
    designs = [
        Design("research-assistant", "research", 6, True, 2.0, False, False),
        Design("codegen-team", "engineering", 5, True, 1.0, True, False),
        Design("agent-marketplace", "population", 1000, False, 24.0, False, True),
        Design("internal-automation", "automation", 3, True, 0.5, True, False),
    ]

    print("=" * 78)
    print("案例映射器 — 拟议设计 → 最接近的 2026 年参考案例")
    print("=" * 78)

    for d in designs:
        print(f"\n设计：{d.name!r}")
        print(f"  类型={d.task_type}  Agent 数={d.n_agents_expected}  "
              f"需要验证={d.verification_required}  运行小时数={d.runtime_duration_hours}")
        case_key = map_to_case(d)
        print_case(case_key)

    print_landscape()

    print("\n要点：")
    print("  先选择案例，再调整设计以匹配其已知权衡。")
    print("  2026 年的每个框架都支持 MCP，大多数支持 A2A；应根据 handoff 语义选择。")
    print("  生产级多 Agent 需要：验证、成本核算与 rainbow 部署。")


if __name__ == "__main__":
    main()
