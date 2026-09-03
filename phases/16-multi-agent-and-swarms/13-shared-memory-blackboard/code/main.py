"""共享内存模式：MessagePool、Blackboard 与投毒演示。

同一个三 Agent 调研任务运行两次。第一次运行中，一个幻觉小数会通过共享内存
传播到最终报告。第二次运行加入一个只读 verifier，它重新获取来源并标记不一致。
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProvenanceEntry:
    id: int
    writer: str
    topic: str
    content: str
    timestamp: float
    prompt_hash: str
    source_uri: str | None = None
    supersedes: int | None = None
    flags: list[str] = field(default_factory=list)


class MessagePool:
    """仅追加的全池共享状态。"""

    def __init__(self) -> None:
        self.entries: list[ProvenanceEntry] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def write(self, writer: str, content: str, prompt: str, source_uri: str | None = None,
              topic: str = "default", supersedes: int | None = None) -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(
                id=eid,
                writer=writer,
                topic=topic,
                content=content,
                timestamp=time.time(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                source_uri=source_uri,
                supersedes=supersedes,
            )
            self.entries.append(e)
            return eid

    def read_all(self) -> list[ProvenanceEntry]:
        with self._lock:
            return list(self.entries)

    def flag(self, entry_id: int, flag: str) -> None:
        with self._lock:
            for e in self.entries:
                if e.id == entry_id:
                    e.flags.append(flag)
                    return


class Blackboard:
    """按主题设键的发布/订阅 blackboard。"""

    def __init__(self) -> None:
        self.topics: dict[str, list[ProvenanceEntry]] = {}
        self.subscribers: dict[str, list[Callable[[ProvenanceEntry], None]]] = {}
        self._lock = threading.Lock()
        self._next_id = 0

    def publish(self, writer: str, topic: str, content: str, prompt: str,
                source_uri: str | None = None) -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(
                id=eid,
                writer=writer,
                topic=topic,
                content=content,
                timestamp=time.time(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                source_uri=source_uri,
            )
            self.topics.setdefault(topic, []).append(e)
            subs = list(self.subscribers.get(topic, []))
        for cb in subs:
            cb(e)
        return eid

    def subscribe(self, topic: str, cb: Callable[[ProvenanceEntry], None]) -> None:
        with self._lock:
            self.subscribers.setdefault(topic, []).append(cb)

    def read_topic(self, topic: str) -> list[ProvenanceEntry]:
        with self._lock:
            return list(self.topics.get(topic, []))


FAKE_SOURCES = {
    "https://arxiv.org/paper-1": "研究报告称，准确率比基线提高了 4.2%。",
    "https://arxiv.org/paper-2": "数据集包含 12,500 个样本。",
}


def retrieval_agent(pool: MessagePool, uri: str, hallucinate: bool) -> int:
    content = FAKE_SOURCES[uri]
    if hallucinate and "4.2%" in content:
        content = content.replace("4.2%", "42%")
    return pool.write(
        writer="retriever",
        content=content,
        prompt=f"Fetch and summarize {uri}",
        source_uri=uri,
    )


def summarizer_agent(pool: MessagePool) -> int:
    retrieved = [e for e in pool.read_all() if e.writer == "retriever"]
    if not retrieved:
        return pool.write("summarizer", "没有来源", "总结检索结果", None)
    latest = retrieved[-1].content
    summary = f"总结：研究报告了一个显著结果 —— {latest.split('。')[0]}。"
    return pool.write("summarizer", summary, "总结检索结果", None)


def analyst_agent(pool: MessagePool) -> int:
    summaries = [e for e in pool.read_all() if e.writer == "summarizer"]
    if not summaries:
        return pool.write("analyst", "没有总结", "得出结论", None)
    latest = summaries[-1].content
    verdict = "建议采用" if "42%" in latest else "建议进一步审查"
    return pool.write("analyst", f"分析结论：{verdict}（依据：{latest}）",
                      "得出结论", None)


def verifier_agent(pool: MessagePool) -> list[tuple[int, str]]:
    """只读 Agent。重新获取引用来源并标记不一致。

    返回 (entry_id, reason) 元组列表，由调用方采取行动。
    verifier 绝不会写回消息池；如何处理由调用方决定。
    """
    findings = []
    for e in pool.read_all():
        if e.source_uri and e.source_uri in FAKE_SOURCES:
            truth = FAKE_SOURCES[e.source_uri]
            if e.content != truth:
                findings.append((e.id, f"与 {e.source_uri} 不一致：获取的文本为 {truth!r}"))
    return findings


def run_without_verifier() -> None:
    print("=" * 72)
    print("运行 1 — 无 verifier；幻觉继续传播")
    print("=" * 72)
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    analyst_agent(pool)
    for e in pool.read_all():
        print(f"  [{e.id}] {e.writer:11s} ({e.prompt_hash}) :: {e.content}")
    print("\n最终报告使用了幻觉产生的 42% 数字，却没有触发任何警报。")


def run_with_verifier() -> None:
    print("\n" + "=" * 72)
    print("运行 2 — 只读 verifier 重新获取来源并加以标记")
    print("=" * 72)
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    findings = verifier_agent(pool)
    for eid, reason in findings:
        pool.flag(eid, reason)
    analyst_agent(pool)

    for e in pool.read_all():
        flag_str = f" [已标记：{'; '.join(e.flags)}]" if e.flags else ""
        print(f"  [{e.id}] {e.writer:11s} ({e.prompt_hash}) :: {e.content}{flag_str}")
    if findings:
        print(f"\nverifier 发现 {len(findings)} 处不一致。下游 Agent 可据此抑制该结论。")


def demo_blackboard() -> None:
    print("\n" + "=" * 72)
    print("BLACKBOARD 演示 — 按主题设键的发布/订阅，并非每个 Agent 都读取全部内容")
    print("=" * 72)
    bb = Blackboard()
    received = {"prices": [], "alerts": []}

    def on_prices(e: ProvenanceEntry) -> None:
        received["prices"].append(e.id)

    def on_alerts(e: ProvenanceEntry) -> None:
        received["alerts"].append(e.id)

    bb.subscribe("prices", on_prices)
    bb.subscribe("alerts", on_alerts)

    bb.publish("scraper-1", "prices", "AAPL=192.4", "poll market")
    bb.publish("scraper-2", "prices", "MSFT=401.2", "poll market")
    bb.publish("risk-engine", "alerts", "ALERT: AAPL moved >2% in 60s", "watch prices")

    print(f"  价格订阅者收到的 ID：{received['prices']}")
    print(f"  警报订阅者收到的 ID：{received['alerts']}")
    print("  （注意：价格订阅者从未看到警报，这正是该设计的目的）")


def main() -> None:
    run_without_verifier()
    run_with_verifier()
    demo_blackboard()
    print("\n要点：")
    print("  1. 没有来源记录的共享状态，会把幻觉漂白后送入下游推理")
    print("  2. 能独立访问来源的只读 verifier 可以发现内存投毒")
    print("  3. blackboard 比全量消息池更易扩展，因为 Agent 只读取自己订阅的内容")


if __name__ == "__main__":
    main()
