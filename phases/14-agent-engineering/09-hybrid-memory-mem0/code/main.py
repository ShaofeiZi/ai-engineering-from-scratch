"""仿 Mem0 的混合记忆：向量 + KV + 图，带融合评分。

仅使用标准库。向量存储使用 token-overlap 作为嵌入 stand-in.
范围分类：user / session / agent。融合：相关性 + 重要性 + 时效性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    rid: str
    text: str
    scope: str
    user_id: str
    session_id: str
    importance: float = 0.5
    ts: float = field(default_factory=time.time)
    tags: tuple[str, ...] = ()


class VectorStore:
    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def add(self, record: Record) -> None:
        self._records[record.rid] = record

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Record]]:
        q_tokens = set(query.lower().split())
        scored: list[tuple[float, Record]] = []
        for record in self._records.values():
            r_tokens = set(record.text.lower().split())
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens)
            if overlap == 0:
                continue
            score = overlap / (len(q_tokens | r_tokens))
            scored.append((score, record))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


@dataclass(frozen=True)
class KVKey:
    user_id: str
    fact_type: str
    entity: str


class KVStore:
    def __init__(self) -> None:
        self._map: dict[KVKey, Record] = {}

    def put(self, key: KVKey, record: Record) -> None:
        self._map[key] = record

    def get(self, key: KVKey) -> Record | None:
        return self._map.get(key)

    def by_user(self, user_id: str) -> list[Record]:
        return [r for k, r in self._map.items() if k.user_id == user_id]


@dataclass
class Edge:
    subject: str
    relation: str
    obj: str
    valid: bool = True
    ts: float = field(default_factory=time.time)


class GraphStore:
    def __init__(self) -> None:
        self._edges: list[Edge] = []

    def add_edge(self, subject: str, relation: str, obj: str) -> None:
        for edge in self._edges:
            if edge.valid and edge.subject == subject and edge.relation == relation:
                edge.valid = False
        self._edges.append(Edge(subject=subject, relation=relation, obj=obj))

    def neighbors(self, subject: str, valid_only: bool = True) -> list[Edge]:
        return [e for e in self._edges
                if e.subject == subject and (e.valid or not valid_only)]

    def all_edges(self) -> list[Edge]:
        return list(self._edges)


@dataclass
class Mem0Config:
    w_relevance: float = 0.6
    w_importance: float = 0.2
    w_recency: float = 0.2
    recency_halflife_s: float = 86400.0


class Mem0:
    def __init__(self, config: Mem0Config | None = None) -> None:
        self.vector = VectorStore()
        self.kv = KVStore()
        self.graph = GraphStore()
        self.config = config or Mem0Config()
        self._counter = 0

    def add(self, text: str, *, user_id: str, session_id: str = "s0",
            scope: str = "user", importance: float = 0.5,
            tags: tuple[str, ...] = (),
            kv_triples: tuple[tuple[str, str], ...] = (),
            graph_triples: tuple[tuple[str, str, str], ...] = ()) -> str:
        self._counter += 1
        rid = f"m{self._counter:03d}"
        record = Record(rid=rid, text=text, scope=scope, user_id=user_id,
                        session_id=session_id, importance=importance, tags=tags)
        self.vector.add(record)
        for fact_type, entity in kv_triples:
            self.kv.put(KVKey(user_id=user_id, fact_type=fact_type, entity=entity), record)
        for subject, relation, obj in graph_triples:
            self.graph.add_edge(subject, relation, obj)
        return rid

    def _recency_score(self, record: Record, now: float) -> float:
        elapsed = max(0.0, now - record.ts)
        half = self.config.recency_halflife_s
        return 0.5 ** (elapsed / half) if half > 0 else 1.0

    def search(self, query: str, *, user_id: str,
               scope: str | None = None, top_k: int = 5) -> list[tuple[float, Record]]:
        now = time.time()
        vector_hits = self.vector.search(query, top_k=top_k * 3)
        fused: dict[str, tuple[float, Record]] = {}
        for rel, record in vector_hits:
            if scope is not None and record.scope != scope:
                continue
            if record.user_id != user_id and record.scope == "user":
                continue
            recency = self._recency_score(record, now)
            score = (self.config.w_relevance * rel
                     + self.config.w_importance * record.importance
                     + self.config.w_recency * recency)
            fused[record.rid] = (score, record)
        for record in self.kv.by_user(user_id):
            if record.rid in fused:
                continue
            recency = self._recency_score(record, now)
            score = (self.config.w_relevance * 0.4
                     + self.config.w_importance * record.importance
                     + self.config.w_recency * recency)
            fused[record.rid] = (score, record)
        ordered = sorted(fused.values(), key=lambda x: -x[0])
        return ordered[:top_k]


def main() -> None:
    print("=" * 70)
    print("MEM0 混合记忆 — 第 14 阶段，第 09 课")
    print("=" * 70)

    mem = Mem0()

    mem.add(
        "ava prefers citation-heavy, terse writing over tutorial style",
        user_id="ava", session_id="s001",
        importance=0.7, tags=("preference", "writing"),
        kv_triples=(("writing_style", "terse_citation_heavy"),),
    )
    mem.add(
        "ava is building a 30-lesson curriculum on agent engineering",
        user_id="ava", session_id="s001",
        importance=0.9, tags=("project",),
        kv_triples=(("project", "agent_curriculum"),),
        graph_triples=(("ava", "owns_project", "agent_curriculum"),),
    )
    mem.add(
        "ava lives in Berlin",
        user_id="ava", session_id="s001",
        importance=0.6, tags=("profile",),
        kv_triples=(("city", "Berlin"),),
        graph_triples=(("ava", "lives_in", "Berlin"),),
    )
    mem.add(
        "ava moved to Lisbon last month",
        user_id="ava", session_id="s002",
        importance=0.8, tags=("profile", "update"),
        kv_triples=(("city", "Lisbon"),),
        graph_triples=(("ava", "lives_in", "Lisbon"),),
    )
    mem.add(
        "bob requested a refund for invoice 4711",
        user_id="bob", session_id="s010",
        importance=0.9, tags=("billing",),
        kv_triples=(("refund_request", "4711"),),
    )

    print("\nvector-only 召回'写作风格偏好'")
    for score, record in mem.vector.search("writing style preferences", top_k=3):
        print(f"  {score:.3f}  {record.rid}  {record.text}")

    print("\n图召回与 'ava' 关联的实体")
    for edge in mem.graph.neighbors("ava", valid_only=False):
        status = "VALID  " if edge.valid else "INVALID"
        print(f"  [{status}] {edge.subject} --{edge.relation}--> {edge.obj}")

    print("\nKV 召回 ava 的所有事实")
    for record in mem.kv.by_user("ava"):
        print(f"  {record.rid}  {record.text}")

    print("\nava 的融合 top-3，查询 'ava 住在哪里'")
    for score, record in mem.search("where does ava live", user_id="ava", top_k=3):
        print(f"  {score:.3f}  {record.rid}  {record.text}")

    print("\nava 的融合 top-3，查询 '她在构建什么'")
    for score, record in mem.search("what is ava building", user_id="ava", top_k=3):
        print(f"  {score:.3f}  {record.rid}  {record.text}")

    print("\n范围隔离：bob 的退款信息不会泄露到 ava 的搜索中")
    hits = mem.search("refund invoice", user_id="ava", top_k=5)
    print(f"  ava 结果：{len(hits)} （预期来自 bob 的 user-scoped 命中数为 0）")
    for score, record in hits:
        print(f"    {score:.3f}  {record.user_id}  {record.text}")

    print()
    print("融合：相关性 + 重要性 + 时效性。per-product 权重调优。")


if __name__ == "__main__":
    main()
