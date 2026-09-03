"""多智能体软件团队——类型化任务板 + 交接统计脚手架。

关键架构原语是类型化消息任务板，它协调一名架构师、N 名并行编码者、一名审查者
和一名测试者，并在每个角色边界生成 trace span。此脚手架使用 stub LLM 调用
运行完整消息流，使交接逻辑和 token 统计可以端到端观察。

运行：python main.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 类型化消息任务板——A2A 风格的类型化消息
# ---------------------------------------------------------------------------

class MsgKind(Enum):
    PLAN_REQUEST = "plan_request"
    SUBTASK = "subtask"
    DIFF_READY = "diff_ready"
    REVIEW_NEEDED = "review_needed"
    REVIEW_FEEDBACK = "review_feedback"
    APPROVED = "approved"
    TEST_NEEDED = "test_needed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"


@dataclass
class Msg:
    kind: MsgKind
    by: str
    to: str
    payload: dict = field(default_factory=dict)
    tokens: int = 0


@dataclass
class Board:
    messages: list[Msg] = field(default_factory=list)
    tokens_by_role: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def post(self, m: Msg) -> None:
        self.messages.append(m)
        self.tokens_by_role[m.by] += m.tokens

    def inbox(self, role: str) -> list[Msg]:
        return [m for m in self.messages if m.to == role]


# ---------------------------------------------------------------------------
# 角色 stub——架构师、编码者、审查者、测试者
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    name: str
    files: list[str]
    lines_changed: int = 0
    has_bug: bool = False  # 用于注入缺陷的探测


def architect_plan(issue: str, rng: random.Random) -> list[Subtask]:
    """架构师计划的 stub 实现。"""
    subs = [
        Subtask("parser", ["src/parser.py"]),
        Subtask("cache", ["src/cache.py", "src/cache_test.py"]),
        Subtask("api", ["src/api.py"]),
        Subtask("migration", ["src/migrate.py"]),
    ]
    # 随机注入一个缺陷，用于探测审查者
    subs[rng.randrange(len(subs))].has_bug = rng.random() < 0.3
    return subs


def coder_implement(sub: Subtask, rng: random.Random) -> dict:
    sub.lines_changed = rng.randint(15, 95)
    return {"subtask": sub.name, "lines": sub.lines_changed,
            "has_bug": sub.has_bug}


def reviewer_check(diffs: list[dict], rng: random.Random) -> tuple[bool, str]:
    """审查者 stub。约 85% 的概率发现缺陷，误批准率为 15%。"""
    buggy = [d for d in diffs if d["has_bug"]]
    if not buggy:
        return True, "lgtm"
    if rng.random() < 0.85:
        return False, f"在 {buggy[0]['subtask']} 中发现缺陷，请重新检查"
    return True, "lgtm（误批准）"


def tester_run(diffs: list[dict], rng: random.Random) -> tuple[bool, str]:
    """测试者 stub。发现所有残留缺陷，约有 3% 的 flaky 概率。"""
    buggy = [d for d in diffs if d["has_bug"]]
    if buggy:
        return False, f"{buggy[0]['subtask']} 模块中的测试失败"
    if rng.random() < 0.03:
        return False, "flaky 测试"
    return True, "412/412 通过"


# ---------------------------------------------------------------------------
# 编排器——运行完整流程并计算 token 放大率
# ---------------------------------------------------------------------------

def run_team(issue: str, n_coders: int = 4, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random(0)
    board = Board()

    # 架构师
    plan = architect_plan(issue, rng)
    board.post(Msg(MsgKind.PLAN_REQUEST, by="architect", to="board",
                   payload={"issue": issue, "subtasks": [s.name for s in plan]},
                   tokens=4500))

    # 将子任务分派给编码者
    for i, sub in enumerate(plan[:n_coders]):
        coder = f"coder-{chr(65 + i)}"
        board.post(Msg(MsgKind.SUBTASK, by="architect", to=coder,
                       payload={"subtask": sub.name, "files": sub.files},
                       tokens=1200))

    # 编码者并行实现
    diffs: list[dict] = []
    for i, sub in enumerate(plan[:n_coders]):
        coder = f"coder-{chr(65 + i)}"
        result = coder_implement(sub, rng)
        diffs.append(result)
        board.post(Msg(MsgKind.DIFF_READY, by=coder, to="merge_coord",
                       payload=result, tokens=3200 + result["lines"] * 30))

    # 合并（此脚手架的构造保证不会发生冲突）
    board.post(Msg(MsgKind.REVIEW_NEEDED, by="merge_coord", to="reviewer",
                   payload={"diffs": diffs}, tokens=2000))

    # 审查者
    approved, comment = reviewer_check(diffs, rng)
    if approved:
        board.post(Msg(MsgKind.APPROVED, by="reviewer", to="tester",
                       payload={"comment": comment}, tokens=1800))
    else:
        # 路由回负责该子任务的编码者（简化为第一名编码者）
        board.post(Msg(MsgKind.REVIEW_FEEDBACK, by="reviewer", to="coder-A",
                       payload={"comment": comment}, tokens=1800))
        # 编码者修订
        board.post(Msg(MsgKind.DIFF_READY, by="coder-A", to="merge_coord",
                       payload={"subtask": "parser", "lines": 52, "has_bug": False},
                       tokens=3100))
        # 审查者重新批准
        board.post(Msg(MsgKind.APPROVED, by="reviewer", to="tester",
                       payload={"comment": "now lgtm"}, tokens=1500))
        # 更新 diff：移除缺陷
        diffs = [{"subtask": d["subtask"], "lines": d["lines"], "has_bug": False}
                 for d in diffs]

    # 测试者
    passed, testmsg = tester_run(diffs, rng)
    if passed:
        board.post(Msg(MsgKind.TEST_PASSED, by="tester", to="pr_opener",
                       payload={"msg": testmsg}, tokens=1200))
    else:
        board.post(Msg(MsgKind.TEST_FAILED, by="tester", to="coder-A",
                       payload={"msg": testmsg}, tokens=1400))

    return {
        "approved": approved,
        "review_comment": comment,
        "tested_passed": passed,
        "test_msg": testmsg,
        "total_tokens": sum(board.tokens_by_role.values()),
        "tokens_by_role": dict(board.tokens_by_role),
        "handoffs": sum(1 for m in board.messages if m.to != m.by),
    }


# ---------------------------------------------------------------------------
# 运行多组配对试验，并与单智能体基线比较
# ---------------------------------------------------------------------------

def single_agent_baseline(issue: str, rng: random.Random) -> dict:
    """Stub：由单个工作树中的一个 Sonnet 4.7 完成全部工作。"""
    # 速度较慢但交接更少；token 数约为总预算减去角色开销
    return {
        "passed": rng.random() < 0.68,
        "total_tokens": 18_000 + rng.randint(0, 6_000),
    }


def main() -> None:
    rng = random.Random(11)
    print("=== 多智能体团队运行 ===")
    result = run_team("fix widget parser race", n_coders=4, rng=rng)
    print(f"已批准      ：{result['approved']}  ({result['review_comment']})")
    print(f"测试已通过  ：{result['tested_passed']}  ({result['test_msg']})")
    print(f"交接次数    ：{result['handoffs']}")
    print(f"token 总数  ：{result['total_tokens']:,}")
    print("按角色统计 token：")
    for role, n in sorted(result['tokens_by_role'].items(), key=lambda x: -x[1]):
        print(f"  {role:14s} {n:>6,}")

    print("\n=== 10 组配对试验与单智能体基线对比 ===")
    team_pass = 0
    baseline_pass = 0
    team_tok_sum = 0
    base_tok_sum = 0
    rng2 = random.Random(17)
    for i in range(10):
        r_team = run_team(f"issue-{i}", n_coders=4, rng=rng2)
        r_base = single_agent_baseline(f"issue-{i}", rng2)
        if r_team['tested_passed']:
            team_pass += 1
        if r_base['passed']:
            baseline_pass += 1
        team_tok_sum += r_team['total_tokens']
        base_tok_sum += r_base['total_tokens']

    print(f"团队通过    ：{team_pass}/10   每次运行 token：{team_tok_sum/10:,.0f}")
    print(f"基线通过    ：{baseline_pass}/10   每次运行 token：{base_tok_sum/10:,.0f}")
    print(f"token 放大率：{team_tok_sum / max(1, base_tok_sum):.2f}x")


if __name__ == "__main__":
    main()
