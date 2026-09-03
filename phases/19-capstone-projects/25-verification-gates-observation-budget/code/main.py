"""
智能体运行框架的验证门禁与观察预算。

See: phases/19-capstone-projects/25-verification-gates-observation-budget/docs/en.md
概念参考：
  - 门禁链模式（优先执行开销最低的拒绝检查，最后才放行）。
  - 将观察预算用作确定性的停止条件。
文件末尾的演示会运行一个合成的三轮循环，并以状态码 0 退出。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Union


# ---------------------------------------------------------------------------
# 传输结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """模型发出的工具调用请求。"""

    turn: int
    tool: str
    argv: tuple[str, ...]
    payload: str = ""

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "tool": self.tool,
            "argv": list(self.argv),
            "payload": self.payload,
        }


@dataclass(frozen=True)
class Observation:
    """工具调用后展示给模型的文本。"""

    turn: int
    tool: str
    text: str
    tokens: int

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "tool": self.tool,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class ToolResult:
    """工具的展示文本，以及用于预算核算的稳定语义夹具。"""

    text: str
    budget_text: str


@dataclass(frozen=True)
class GateDecision:
    """单个门禁的判定结果。"""

    allow: bool
    gate: str
    reason: str

    def to_dict(self) -> dict:
        return {"allow": self.allow, "gate": self.gate, "reason": self.reason}


# ---------------------------------------------------------------------------
# Token 估算器
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """真实 tokenizer 的确定性保守替代实现。

    真实运行框架会接入 tiktoken 或模型自带的 tokenizer。
    门禁链只要求计数器具有单调性和确定性。
    """

    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 观察账本
# ---------------------------------------------------------------------------


@dataclass
class ObservationLedger:
    """记录所有已展示给模型的观察结果，只允许追加。"""

    rows: list[Observation] = field(default_factory=list)

    def record(self, obs: Observation) -> None:
        self.rows.append(obs)

    def cumulative(self) -> int:
        return sum(row.tokens for row in self.rows)

    def per_tool(self, name: str) -> int:
        return sum(row.tokens for row in self.rows if row.tool == name)

    def turns_seen(self) -> list[int]:
        return sorted({row.turn for row in self.rows})

    def latest_turn(self) -> int:
        return self.rows[-1].turn if self.rows else -1

    def snapshot(self) -> list[dict]:
        return [row.to_dict() for row in self.rows]


# ---------------------------------------------------------------------------
# 门禁协议
# ---------------------------------------------------------------------------


@dataclass
class GateContext:
    """传给每个门禁的只读上下文。"""

    ledger: ObservationLedger
    current_turn: int
    history: tuple[ToolCall, ...] = ()


class VerificationGate(Protocol):
    name: str

    def evaluate(self, call: ToolCall, ctx: GateContext) -> GateDecision: ...


# ---------------------------------------------------------------------------
# 具体门禁
# ---------------------------------------------------------------------------


@dataclass
class WhitelistGate:
    """拒绝显式允许集合之外的所有工具；这是开销最低的门禁。"""

    allowed: frozenset[str]
    name: str = "whitelist"

    def evaluate(self, call: ToolCall, ctx: GateContext) -> GateDecision:
        if call.tool in self.allowed:
            return GateDecision(True, self.name, "tool in allow-set")
        return GateDecision(
            False,
            self.name,
            f"tool {call.tool!r} not in allow-set {sorted(self.allowed)}",
        )


@dataclass
class RegexGate:
    """如果 argv 拼接后的字符串匹配任一拒绝模式，则拒绝调用。"""

    refuse_patterns: tuple[re.Pattern[str], ...]
    name: str = "regex"

    @classmethod
    def from_strings(cls, patterns: Iterable[str], name: str = "regex") -> "RegexGate":
        compiled = tuple(re.compile(p) for p in patterns)
        return cls(refuse_patterns=compiled, name=name)

    def evaluate(self, call: ToolCall, ctx: GateContext) -> GateDecision:
        haystack = " ".join(call.argv) + " " + call.payload
        for pat in self.refuse_patterns:
            if pat.search(haystack):
                return GateDecision(
                    False, self.name, f"argv matched refuse pattern {pat.pattern!r}"
                )
        return GateDecision(True, self.name, "no refuse pattern matched")


@dataclass
class RecencyGate:
    """如果上一次观察距今超过 window 轮，则拒绝调用。

    这样可以强制执行新的读取，而不是依赖过时状态。
    会话中的第一次调用始终放行。
    """

    window: int
    name: str = "recency"

    def evaluate(self, call: ToolCall, ctx: GateContext) -> GateDecision:
        last = ctx.ledger.latest_turn()
        if last < 0:
            return GateDecision(True, self.name, "no prior observations")
        gap = call.turn - last
        if gap > self.window:
            return GateDecision(
                False,
                self.name,
                f"observation gap {gap} turns exceeds window {self.window}",
            )
        return GateDecision(True, self.name, f"gap {gap} within window {self.window}")


@dataclass
class BudgetGate:
    """累计观察预算耗尽后拒绝调用。

    单次调用无法提前获知其结果会包含多少 token。因此，门禁会根据调用前的
    账本状态进行评估；记录观察结果后，运行框架会针对新的账本状态再次执行
    累计检查。
    """

    max_tokens: int
    name: str = "budget"

    def evaluate(self, call: ToolCall, ctx: GateContext) -> GateDecision:
        used = ctx.ledger.cumulative()
        if used >= self.max_tokens:
            return GateDecision(
                False,
                self.name,
                f"observation budget exhausted: {used}/{self.max_tokens} tokens",
            )
        remaining = self.max_tokens - used
        return GateDecision(
            True, self.name, f"{remaining} tokens of budget remaining"
        )


@dataclass
class PerToolBudgetGate:
    """可选门禁：单个工具消耗超过其配额时拒绝调用。"""

    limits: dict[str, int]
    name: str = "per-tool-budget"

    def evaluate(self, call: ToolCall, ctx: GateContext) -> GateDecision:
        limit = self.limits.get(call.tool)
        if limit is None:
            return GateDecision(True, self.name, "tool has no per-tool budget")
        used = ctx.ledger.per_tool(call.tool)
        if used >= limit:
            return GateDecision(
                False,
                self.name,
                f"per-tool budget for {call.tool} exhausted: {used}/{limit}",
            )
        return GateDecision(
            True, self.name, f"per-tool {call.tool}: {limit - used} tokens remaining"
        )


# ---------------------------------------------------------------------------
# 门禁链
# ---------------------------------------------------------------------------


@dataclass
class ChainOutcome:
    """门禁链评估的完整结果：各门禁的判定及最终结论。"""

    decisions: list[GateDecision]

    @property
    def allow(self) -> bool:
        return all(d.allow for d in self.decisions)

    @property
    def deny_reason(self) -> str | None:
        for d in self.decisions:
            if not d.allow:
                return f"[{d.gate}] {d.reason}"
        return None

    def to_dict(self) -> dict:
        return {
            "allow": self.allow,
            "deny_reason": self.deny_reason,
            "decisions": [d.to_dict() for d in self.decisions],
        }


@dataclass
class GateChain:
    """按顺序评估的门禁列表，遇到首次拒绝时短路。"""

    gates: tuple[VerificationGate, ...]

    def evaluate(self, call: ToolCall, ctx: GateContext) -> ChainOutcome:
        decisions: list[GateDecision] = []
        for gate in self.gates:
            decision = gate.evaluate(call, ctx)
            decisions.append(decision)
            if not decision.allow:
                return ChainOutcome(decisions=decisions)
        return ChainOutcome(decisions=decisions)


# ---------------------------------------------------------------------------
# 用于演示的微型合成智能体循环
# ---------------------------------------------------------------------------


ToolFn = Callable[[ToolCall], Union[str, ToolResult]]


@dataclass
class LoopReport:
    """合成循环运行的审计记录。"""

    turns: int
    allowed: int
    refused: int
    observations: list[Observation]
    decisions: list[ChainOutcome]

    def to_dict(self) -> dict:
        return {
            "turns": self.turns,
            "allowed": self.allowed,
            "refused": self.refused,
            "observations": [o.to_dict() for o in self.observations],
            "decisions": [d.to_dict() for d in self.decisions],
        }


def run_synthetic_loop(
    calls: list[ToolCall],
    chain: GateChain,
    tool_fns: dict[str, ToolFn],
) -> LoopReport:
    """让固定的工具调用序列依次通过门禁链。

    这是运行框架骨架的微缩版本。真实框架会向模型询问下一次工具调用，
    但门禁链的契约完全相同。
    """

    ledger = ObservationLedger()
    decisions: list[ChainOutcome] = []
    observations: list[Observation] = []
    allowed = 0
    refused = 0

    history: list[ToolCall] = []

    for call in calls:
        ctx = GateContext(
            ledger=ledger, current_turn=call.turn, history=tuple(history)
        )
        outcome = chain.evaluate(call, ctx)
        decisions.append(outcome)
        history.append(call)
        if not outcome.allow:
            refused += 1
            continue
        fn = tool_fns.get(call.tool)
        if fn is None:
            refused += 1
            continue
        result = fn(call)
        if isinstance(result, ToolResult):
            text = result.text
            budget_text = result.budget_text
        else:
            text = result
            budget_text = result
        obs = Observation(
            turn=call.turn,
            tool=call.tool,
            text=text,
            tokens=estimate_tokens(budget_text),
        )
        ledger.record(obs)
        observations.append(obs)
        allowed += 1

    return LoopReport(
        turns=len(calls),
        allowed=allowed,
        refused=refused,
        observations=observations,
        decisions=decisions,
    )


# ---------------------------------------------------------------------------
# 演示装配
# ---------------------------------------------------------------------------


def _demo_tools() -> dict[str, ToolFn]:
    """三个合成工具：read_file 输出较多，list_dir 较短，run_tests 返回结构化结果。"""

    def read_file(call: ToolCall) -> ToolResult:
        target = call.argv[0] if call.argv else "<missing>"
        return ToolResult(
            text=(
                f"# {target} 的模拟内容\n"
                + ("一行长度约为六十字节的模拟源代码 " * 12)
            ),
            # 预算代表工具结果的语义载荷，不随展示语言变化。
            budget_text=(
                f"# fake contents of {target}\n"
                + ("line of fake source code that is sixty bytes long " * 12)
            ),
        )

    def list_dir(call: ToolCall) -> str:
        return "main.py\nREADME.md\ntests/test_main.py\n"

    def run_tests(call: ToolCall) -> str:
        return json.dumps(
            {"status": "passed", "tests": 4, "duration_ms": 42}, indent=2
        )

    return {"read_file": read_file, "list_dir": list_dir, "run_tests": run_tests}


def build_default_chain(budget: int = 200) -> GateChain:
    """按照 en.md 记录的顺序装配标准四门禁链。"""

    return GateChain(
        gates=(
            WhitelistGate(
                allowed=frozenset({"read_file", "list_dir", "run_tests"})
            ),
            RegexGate.from_strings(
                patterns=(
                    r"\brm\s+-rf\b",
                    r"\bsudo\b",
                    r"^/etc/",
                )
            ),
            RecencyGate(window=3),
            BudgetGate(max_tokens=budget),
        )
    )


def run_demo() -> int:
    """可自行终止的演示：打印 JSON 追踪，并以状态码 0 退出。"""

    chain = build_default_chain(budget=200)
    tools = _demo_tools()

    calls = [
        ToolCall(turn=1, tool="list_dir", argv=("./",)),
        ToolCall(turn=2, tool="read_file", argv=("main.py",)),
        ToolCall(turn=3, tool="read_file", argv=("README.md",)),
        ToolCall(turn=4, tool="run_tests", argv=("./",)),
        ToolCall(turn=5, tool="shell", argv=("rm", "-rf", "/")),
    ]

    report = run_synthetic_loop(calls, chain, tools)

    print("验证门禁演示")
    print(f"轮数={report.turns} 已放行={report.allowed} 已拒绝={report.refused}")
    print("")
    for idx, (call, outcome) in enumerate(zip(calls, report.decisions)):
        verdict = "ALLOW" if outcome.allow else "DENY"
        print(f"  [{idx}] turn={call.turn} tool={call.tool} -> {verdict}")
        if not outcome.allow:
            print(f"        原因：{outcome.deny_reason}")
    print("")
    print(f"累计观察 token 数：{sum(o.tokens for o in report.observations)}")
    print(f"已记录观察数：{len(report.observations)}")

    if report.refused < 1:
        print("错误：演示预期至少出现一次拒绝", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
