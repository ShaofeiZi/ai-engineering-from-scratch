"""GitHub issue-to-PR 异步云端智能体——分发器 + 预算 + 安全门禁。

关键架构原语是分发器，它强制执行逐仓库预算、限定范围的 GitHub App 凭据，以及
绝不允许智能体强制推送或逃离仓库范围的沙箱生命周期。此脚手架实现分发器、
预算台账、沙箱状态机和验证门禁。

运行：python main.py
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# webhook -> 任务入队——标签触发器与队列契约
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: int
    repo: str
    issue_num: int
    title: str
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 预算台账——逐仓库每日美元和 PR 数量上限
# ---------------------------------------------------------------------------

@dataclass
class BudgetLedger:
    daily_dollar_cap: float = 50.0
    daily_pr_cap: int = 5
    per_task_dollar_cap: float = 20.0
    spent_today: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prs_today: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def permit(self, repo: str, estimated_cost: float) -> tuple[bool, str]:
        if estimated_cost > self.per_task_dollar_cap:
            return False, f"task estimate ${estimated_cost:.2f} > cap ${self.per_task_dollar_cap}"
        # 按每个任务的最坏情况支出预留，而不是按估算值预留。``run_agent`` 中的
        # 智能体循环在触发 ``dollar_cap`` 前最多可使用 ``per_task_dollar_cap``，
        # 因此若按 ``estimated`` 准入，一批达到上限的运行会突破每日上限。``record``
        # 仍会写入实际支出，因此未使用的预留会自动核销。
        worst_case = self.per_task_dollar_cap
        if self.spent_today[repo] + worst_case > self.daily_dollar_cap:
            return False, f"daily $ cap for {repo} would be exceeded"
        if self.prs_today[repo] >= self.daily_pr_cap:
            return False, f"daily PR cap ({self.daily_pr_cap}) for {repo} reached"
        return True, "ok"

    def record(self, repo: str, spent: float, opened_pr: bool) -> None:
        self.spent_today[repo] += spent
        if opened_pr:
            self.prs_today[repo] += 1


# ---------------------------------------------------------------------------
# GitHub App 身份——短期 installation token、限定范围的权限
# ---------------------------------------------------------------------------

@dataclass
class InstallationToken:
    repo: str
    expires_at: float
    permissions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def mint(cls, repo: str) -> "InstallationToken":
        return cls(repo=repo,
                   expires_at=time.time() + 3600,
                   permissions={"issues": "rw", "pull_requests": "rw",
                                "contents": "rw", "workflows": "r"})

    def can(self, action: str) -> bool:
        # 硬策略：绝不强制推送
        if action == "force_push":
            return False
        if action.startswith("write:main"):
            return False
        return True


# ---------------------------------------------------------------------------
# 沙箱状态机——CLONE -> INFER -> AGENT -> VERIFY -> PR
# ---------------------------------------------------------------------------

class SState(Enum):
    CLONE = auto()
    INFER = auto()
    AGENT = auto()
    VERIFY = auto()
    PR = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class SandboxRun:
    task: Task
    state: SState = SState.CLONE
    turns: int = 0
    dollars: float = 0.0
    wall_min: float = 0.0
    coverage_delta: float = 0.0
    ci_green: bool = False
    pr_opened: bool = False
    failure: str | None = None
    trace: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 智能体循环 stub——使用按难度加权的逐轮概率
# ---------------------------------------------------------------------------

def run_agent(run: SandboxRun, difficulty: float, rng: random.Random,
              turn_cap: int = 20, dollar_cap: float = 20.0,
              minute_cap: float = 30.0) -> None:
    run.state = SState.AGENT
    per_turn_p = max(0.05, 0.35 * (1 - difficulty))
    per_turn_min = 0.9 + difficulty * 0.6
    per_turn_usd = 0.25 + difficulty * 0.45

    while True:
        run.turns += 1
        run.wall_min += per_turn_min
        run.dollars += per_turn_usd
        run.trace.append(f"turn {run.turns}: $={run.dollars:.2f}")

        if run.turns >= turn_cap:
            run.failure = "turn_cap"
            run.state = SState.FAILED
            return
        if run.dollars >= dollar_cap:
            run.failure = "dollar_cap"
            run.state = SState.FAILED
            return
        if run.wall_min >= minute_cap:
            run.failure = "minute_cap"
            run.state = SState.FAILED
            return

        if rng.random() < per_turn_p:
            run.state = SState.VERIFY
            return


def run_verify(run: SandboxRun, difficulty: float, rng: random.Random) -> None:
    flake = rng.random() < 0.05
    if flake:
        run.ci_green = False
        run.failure = "flaky_test"
        run.state = SState.FAILED
        return
    run.ci_green = True
    run.coverage_delta = rng.gauss(0.0, 0.6)
    if run.coverage_delta < -2.0:
        run.failure = "coverage_regression"
        run.state = SState.FAILED
        return
    run.state = SState.PR


def open_pr(run: SandboxRun, token: InstallationToken) -> None:
    # 使用显式运行时检查——安全门禁绝不能使用 `assert`。`python -O` 会移除断言，
    # 从而让被拒绝或已过期的 token 仍可创建 PR。
    if time.time() >= token.expires_at:
        run.failure = "token_expired"
        run.state = SState.FAILED
        return
    if not token.can("pull_request.open"):
        run.failure = "policy_denied"
        run.state = SState.FAILED
        return
    run.pr_opened = True
    run.state = SState.DONE


# ---------------------------------------------------------------------------
# 分发器——拉取任务、强制预算并运行沙箱流程
# ---------------------------------------------------------------------------

def dispatch(task: Task, ledger: BudgetLedger, rng: random.Random) -> SandboxRun:
    difficulty = rng.uniform(0.3, 0.92)
    estimated = 2.0 + difficulty * 8.0
    allowed, reason = ledger.permit(task.repo, estimated)
    if not allowed:
        run = SandboxRun(task)
        run.failure = f"dispatcher: {reason}"
        run.state = SState.FAILED
        return run

    token = InstallationToken.mint(task.repo)
    run = SandboxRun(task)
    run.trace.append("state: CLONE")
    run.state = SState.INFER
    run.trace.append("state: INFER (dockerfile synthesized)")
    run_agent(run, difficulty, rng)
    if run.state == SState.VERIFY:
        run_verify(run, difficulty, rng)
    if run.state == SState.PR:
        open_pr(run, token)
    ledger.record(task.repo, run.dollars, run.pr_opened)
    return run


# ---------------------------------------------------------------------------
# 演示——在 3 个仓库运行 20 个 issue；部分任务会触及预算上限
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(9)
    ledger = BudgetLedger()
    repos = ["acme/widget", "acme/service", "acme/library"]
    runs: list[SandboxRun] = []

    for i in range(20):
        task = Task(task_id=i, repo=rng.choice(repos), issue_num=800 + i,
                    title=f"fix NPE in module {i}")
        run = dispatch(task, ledger, rng)
        runs.append(run)

    opened = sum(1 for r in runs if r.pr_opened)
    failed = sum(1 for r in runs if r.state == SState.FAILED)
    print(f"=== 分发结果（{len(runs)} 个任务）===")
    print(f"已创建 PR：{opened}")
    print(f"失败      ：{failed}")

    print("\n失败原因：")
    reasons = defaultdict(int)
    for r in runs:
        if r.failure:
            reasons[r.failure] += 1
    for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:24s} {n}")

    print("\n预算摘要：")
    for repo in repos:
        print(f"  {repo:20s} 已花费=${ledger.spent_today[repo]:.2f}  "
              f"PRs={ledger.prs_today[repo]}")

    if opened:
        mean_cost = sum(r.dollars for r in runs if r.pr_opened) / opened
        mean_turns = sum(r.turns for r in runs if r.pr_opened) / opened
        print(f"\n通过集合：每个 PR 平均成本 = ${mean_cost:.2f}  平均轮次 = {mean_turns:.1f}")


if __name__ == "__main__":
    main()
