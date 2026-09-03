"""Supervisor / Orchestrator-Worker 模式（Anthropic Research 风格）。

Lead Agent 分解查询，通过并行线程启动 worker，再综合结果。
这里不进行真实的 LLM 调用，worker 是脚本化的抓取和总结模拟。

重点是并行子 Agent 缩短实际耗时的收益，以及这一模式本身。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class WorkerResult:
    sub_question: str
    summary: str
    tokens_spent: int
    wall_time: float


@dataclass
class TraceEntry:
    worker_id: int
    event: str
    t: float
    sub_question: str = ""


@dataclass
class Trace:
    entries: list[TraceEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, worker_id: int, event: str, sub_question: str = "") -> None:
        with self._lock:
            self.entries.append(
                TraceEntry(worker_id=worker_id, event=event, t=time.time(), sub_question=sub_question)
            )


def fake_web_fetch(query: str) -> str:
    """模拟网页抓取与总结的延迟。"""
    time.sleep(0.3)
    return f"关于“{query}”的总结：从 5 个来源中得到 3 项关键发现。"


class Worker:
    def __init__(self, worker_id: int, trace: Trace) -> None:
        self.worker_id = worker_id
        self.trace = trace

    def run(self, sub_question: str, results: list[WorkerResult | None], idx: int) -> None:
        start = time.time()
        self.trace.log(self.worker_id, "start", sub_question)
        summary = fake_web_fetch(sub_question)
        elapsed = time.time() - start
        results[idx] = WorkerResult(
            sub_question=sub_question,
            summary=summary,
            tokens_spent=800,
            wall_time=elapsed,
        )
        self.trace.log(self.worker_id, "done", sub_question)


class Lead:
    """Supervisor：规划、并行启动 worker 并综合结果。"""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def plan(self, query: str) -> list[str]:
        """分解任务。真实的 lead 会使用 LLM；此处按启发式规则拆分。"""
        return [
            f"{query} -- 历史起源",
            f"{query} -- 2026 年技术前沿",
            f"{query} -- 开放问题",
        ]

    def synthesize(self, query: str, results: list[WorkerResult]) -> str:
        ok = [r for r in results if r is not None]
        parts = [f"- {r.sub_question}: {r.summary}" for r in ok]
        return f"对“{query}”的回答：\n" + "\n".join(parts)

    def run(self, query: str) -> tuple[str, dict]:
        t0 = time.time()
        sub_questions = self.plan(query)
        self.trace.log(worker_id=-1, event="plan", sub_question=str(len(sub_questions)))

        results: list[WorkerResult | None] = [None] * len(sub_questions)
        threads: list[threading.Thread] = []
        for i, sq in enumerate(sub_questions):
            w = Worker(worker_id=i, trace=self.trace)
            th = threading.Thread(target=w.run, args=(sq, results, i))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        self.trace.log(worker_id=-1, event="synthesize")
        synthesis = self.synthesize(query, [r for r in results if r is not None])
        total_wall = time.time() - t0
        total_tokens = sum((r.tokens_spent for r in results if r is not None)) + 1200
        return synthesis, {
            "wall_clock_seconds": round(total_wall, 3),
            "total_tokens": total_tokens,
            "worker_count": len(sub_questions),
        }


def render_trace(trace: Trace, t0: float) -> None:
    for e in trace.entries:
        rel = round(e.t - t0, 3)
        sq = f" | {e.sub_question}" if e.sub_question else ""
        tag = "LEAD" if e.worker_id == -1 else f"W{e.worker_id}"
        print(f"  +{rel:>5}s  {tag:>4}  {e.event}{sq}")


def main() -> None:
    print("Supervisor / Orchestrator-Worker 演示")
    print("-" * 42)

    trace = Trace()
    t0 = time.time()
    lead = Lead(trace=trace)
    answer, stats = lead.run("多 Agent 系统在 2023 至 2026 年间发生了哪些变化？")

    print("\n轨迹（相对于规划开始时间的秒数）：")
    render_trace(trace, t0)

    print("\n最终综合结果：")
    print("  " + answer.replace("\n", "\n  "))

    print("\n统计：")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n顺序执行的基线约为 0.9 秒（3 * 0.3 秒）。")
    print("并行执行实际约为 0.35 秒。这就是 supervisor 带来的收益。")


if __name__ == "__main__":
    main()
