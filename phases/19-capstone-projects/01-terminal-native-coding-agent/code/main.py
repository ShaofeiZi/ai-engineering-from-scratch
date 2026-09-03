"""终端原生编码智能体——最小化 plan/act/observe 循环脚手架。

2026 年编码智能体最关键的架构原语并非模型调用或某个单独工具，而是具有
上下文边界、结构化计划状态、沙箱化工具分发器，以及覆盖每个生命周期节点
hook 回调的 plan-act-observe-recover 循环。本文件使用 Python 标准库端到端
实现该循环。LLM 由确定性脚本替代，因此无需网络调用也能观察并测试循环逻辑。

运行：python main.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 计划状态——采用 TodoWrite 结构，每轮整体重写
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    id: int
    description: str
    status: str  # "pending" | "in_progress" | "done" | "failed"
    note: str = ""


@dataclass
class PlanState:
    goal: str
    items: list[TodoItem] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"GOAL: {self.goal}"]
        for it in self.items:
            mark = {"pending": " ", "in_progress": ">", "done": "x", "failed": "!"}[it.status]
            lines.append(f"  [{mark}] {it.id}. {it.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 预算——轮次、token 和美元成本的硬上限
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    max_turns: int = 50
    max_tokens: int = 200_000
    max_dollars: float = 5.00
    turns_used: int = 0
    tokens_used: int = 0
    dollars_used: float = 0.0

    def step(self, tokens: int, dollars: float) -> None:
        self.turns_used += 1
        self.tokens_used += tokens
        self.dollars_used += dollars

    def exceeded(self) -> str | None:
        if self.turns_used >= self.max_turns:
            return "turn_limit"
        if self.tokens_used >= self.max_tokens:
            return "token_limit"
        if self.dollars_used >= self.max_dollars:
            return "dollar_limit"
        return None


# ---------------------------------------------------------------------------
# hook——2026 年的八事件接口（Pre/PostToolUse、SessionStart/End 等）
# ---------------------------------------------------------------------------

HookFn = Callable[[dict[str, Any]], dict[str, Any]]


class HookBus:
    EVENTS = ("SessionStart", "SessionEnd", "PreToolUse", "PostToolUse",
              "UserPromptSubmit", "Notification", "Stop", "PreCompact")

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFn]] = {e: [] for e in self.EVENTS}

    def on(self, event: str, fn: HookFn) -> None:
        self._hooks[event].append(fn)

    def fire(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        for fn in self._hooks[event]:
            payload = fn(payload) or payload
        return payload


# ---------------------------------------------------------------------------
# 工具接口——六个沙箱化工具，各自返回截断后的文本
# ---------------------------------------------------------------------------

TRUNCATE_BYTES = 4096


def tool_read_file(sandbox: str, path: str) -> str:
    full = os.path.join(sandbox, path)
    if not os.path.realpath(full).startswith(os.path.realpath(sandbox)):
        raise RuntimeError("路径越出沙箱范围")
    with open(full, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()[:TRUNCATE_BYTES]


def tool_run_shell(sandbox: str, cmd: str, timeout: int = 30) -> str:
    proc = subprocess.run(cmd, cwd=sandbox, shell=True, capture_output=True,
                          text=True, timeout=timeout)
    out = (proc.stdout + proc.stderr)[:TRUNCATE_BYTES]
    return f"exit={proc.returncode}\n{out}"


TOOLS: dict[str, Callable[..., str]] = {
    "read_file": tool_read_file,
    "run_shell": tool_run_shell,
}


# ---------------------------------------------------------------------------
# stub 模型——使用确定性脚本，无需 LLM 即可测试循环
# ---------------------------------------------------------------------------

SCRIPT = [
    {"plan": [("定位目标文件", "in_progress"),
              ("读取并诊断", "pending"),
              ("应用修复并验证", "pending")],
     "tool": ("run_shell", {"cmd": "ls"}),
     "tokens": 1200, "cost": 0.02},
    {"plan": [("定位目标文件", "done"),
              ("读取并诊断", "in_progress"),
              ("应用修复并验证", "pending")],
     "tool": ("read_file", {"path": "README.md"}),
     "tokens": 900, "cost": 0.02},
    {"plan": [("定位目标文件", "done"),
              ("读取并诊断", "done"),
              ("应用修复并验证", "done")],
     "tool": None,  # 终止轮次
     "tokens": 600, "cost": 0.01},
]


def model_step(plan: PlanState, turn: int) -> dict[str, Any]:
    """Stub 模型：返回重写后的计划，以及可选的工具调用。"""
    if turn >= len(SCRIPT):
        return {"plan": plan.items, "tool": None, "tokens": 200, "cost": 0.005}
    s = SCRIPT[turn]
    items = [TodoItem(i + 1, desc, status) for i, (desc, status) in enumerate(s["plan"])]
    return {"plan": items, "tool": s["tool"], "tokens": s["tokens"], "cost": s["cost"]}


# ---------------------------------------------------------------------------
# 主循环——完整集成 hook 的 plan / act / observe / recover 流程
# ---------------------------------------------------------------------------

def destructive_guard(payload: dict[str, Any]) -> dict[str, Any]:
    cmd = payload.get("args", {}).get("cmd", "")
    if "rm -rf" in cmd or "shutdown" in cmd:
        payload["blocked"] = True
        payload["reason"] = "破坏性命令已被 PreToolUse hook 阻止"
    return payload


def run_agent(task: str, sandbox: str) -> dict[str, Any]:
    plan = PlanState(goal=task, items=[])
    budget = Budget()
    hooks = HookBus()
    trace: list[dict[str, Any]] = []

    hooks.on("PreToolUse", destructive_guard)
    hooks.on("PostToolUse", lambda p: (trace.append({"event": "tool", **p}), p)[1])
    hooks.on("SessionStart", lambda p: (trace.append({"event": "start", **p}), p)[1])
    hooks.on("SessionEnd", lambda p: (trace.append({"event": "end", **p}), p)[1])

    hooks.fire("SessionStart", {"task": task, "sandbox": sandbox,
                                "started_at": time.time()})

    turn = 0
    while True:
        stop = budget.exceeded()
        if stop:
            hooks.fire("Stop", {"reason": stop, "turn": turn})
            break

        step = model_step(plan, turn)
        plan.items = step["plan"]
        budget.step(step["tokens"], step["cost"])

        call = step["tool"]
        if call is None:
            hooks.fire("Stop", {"reason": "complete", "turn": turn})
            break

        name, args = call
        pre = hooks.fire("PreToolUse", {"tool": name, "args": args})
        if pre.get("blocked"):
            hooks.fire("PostToolUse", {"tool": name, "blocked": True,
                                       "reason": pre.get("reason", "")})
            turn += 1
            continue

        try:
            result = TOOLS[name](sandbox, **args)
            hooks.fire("PostToolUse", {"tool": name, "ok": True,
                                       "bytes": len(result)})
        except Exception as exc:
            hooks.fire("PostToolUse", {"tool": name, "ok": False,
                                       "error": str(exc)})

        turn += 1

    hooks.fire("SessionEnd", {"turns": budget.turns_used,
                              "tokens": budget.tokens_used,
                              "dollars": budget.dollars_used})

    return {"plan": plan.summary(), "budget": asdict(budget), "trace": trace}


def main() -> None:
    task = "演示无需网络调用的 plan-act-observe 循环"
    sandbox = os.path.dirname(os.path.abspath(__file__))
    result = run_agent(task, sandbox)
    print(result["plan"])
    print("---")
    print(f"turns={result['budget']['turns_used']} "
          f"tokens={result['budget']['tokens_used']} "
          f"dollars=${result['budget']['dollars_used']:.3f}")
    print("---")
    print(f"追踪事件数：{len(result['trace'])}")
    for ev in result["trace"]:
        print(" ", json.dumps(ev, default=str))


if __name__ == "__main__":
    main()
