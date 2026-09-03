"""最小 durable-execution 引擎 — 纯标准库 Python。

建模 Temporal、LangGraph 检查点机制、Microsoft Agent Framework 和 Claude Code Routines 所使用的工作流 / 活动 / event-log 模式。

活动在执行前记录输入，执行后记录输出。重放工作流时会重新运行工作流代码，但对于事件日志中已存在的活动直接返回缓存的输出。运行中途崩溃只会丢失未完成的活动。
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from dataclasses import dataclass


# ---------- 事件日志 ----------

@dataclass
class EventLog:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def events(self) -> list[dict]:
        with open(self.path) as f:
            return json.load(f)

    def append(self, ev: dict) -> None:
        evs = self.events()
        evs.append(ev)
        with open(self.path, "w") as f:
            json.dump(evs, f)

    def lookup(self, name: str, args: tuple) -> dict | None:
        for ev in self.events():
            if ev["name"] == name and ev["args"] == list(args) and ev["status"] == "done":
                return ev
        return None


# ---------- 活动装饰器 ----------

def activity(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(log: EventLog, *args):
            hit = log.lookup(name, args)
            if hit:
                print(f"    [重放] {name}({args}) -> {hit['result']}（来自日志）")
                return hit["result"]
            log.append({"name": name, "args": list(args), "status": "started"})
            result = fn(*args)
            log.append({"name": name, "args": list(args),
                        "status": "done", "result": result})
            print(f"    [运行]   {name}({args}) -> {result}")
            return result
        return wrapper
    return deco


# ---------- 示例活动 ----------

@activity("fetch_docs")
def fetch_docs(query: str) -> int:
    # 模拟调用 API;返回文档数量。
    return len(query) * 3


@activity("call_llm")
def call_llm(doc_count: int) -> str:
    # 模拟 LLM 调用;此处为教学目的设为确定性。
    return f"summary({doc_count}_docs)"


@activity("write_report")
def write_report(summary: str) -> str:
    # 模拟带有副作用的工具调用。
    return f"report://{summary}"


# ---------- 工作流 ----------

def workflow(log: EventLog, query: str, crash_after: int = -1) -> str:
    """包含三个活动的工作流，可选择触发教学用崩溃。"""
    doc_count = fetch_docs(log, query)
    if crash_after == 1:
        raise RuntimeError("simulated crash after fetch_docs")
    summary = call_llm(log, doc_count)
    if crash_after == 2:
        raise RuntimeError("simulated crash after call_llm")
    report = write_report(log, summary)
    return report


# ---------- 驱动程序 ----------

def reset_log(path: str) -> EventLog:
    if os.path.exists(path):
        os.remove(path)
    return EventLog(path)


def count_runs(log: EventLog) -> int:
    return sum(1 for ev in log.events() if ev["status"] == "started")


def main() -> None:
    print("=" * 70)
    print("DURABLE EXECUTION(阶段 15,第 12 课)")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()

    # 朴素重试:崩溃时丢失事件日志。每次重启都会重新运行
    # 所有内容。
    print("\n朴素重试(事件日志未持久化)")
    print("-" * 70)
    for attempt in range(1, 4):
        log = reset_log(os.path.join(tmpdir, "naive.json"))
        print(f"  第 {attempt} 次尝试:")
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> 结果 {r}")
            print(f"    -> {count_runs(log)} 活动启动于本次尝试")
            break
        except RuntimeError as e:
            print(f"    -> 崩溃:{e};{count_runs(log)} 活动启动白费了")

    # 持久化重试:跨尝试保留事件日志;重放不会
    # re-execute 已完成的活动。
    print("\n持久化重试(事件日志跨尝试保留)")
    print("-" * 70)
    durable_path = os.path.join(tmpdir, "durable.json")
    if os.path.exists(durable_path):
        os.remove(durable_path)

    for attempt in range(1, 4):
        log = EventLog(durable_path)
        print(f"  第 {attempt} 次尝试:")
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> 结果 {r}")
            print(f"    -> 跨所有尝试累计 {count_runs(log)} 次活动启动")
            break
        except RuntimeError as e:
            print(f"    -> 崩溃:{e}")

    print()
    print("=" * 70)
    print("要点：持久化使长视野运行失败的成本可控")
    print("-" * 70)
    print("  朴素重试会在每次尝试中重新执行所有活动。")
    print("  持久化重试从日志中重放已完成的活动;")
    print("  只有缺失的活动真正执行。同样的设计被")
    print("  Temporal、LangGraph 检查点机制、Microsoft Agent")
    print("  Framework 和 Claude Code Routines 采用。LLM 调用")
    print("  只是日志中又一个非确定性活动。")


if __name__ == "__main__":
    main()
