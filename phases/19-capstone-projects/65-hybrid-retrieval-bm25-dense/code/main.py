"""混合检索：BM25 + 稠密向量 + 倒数排名融合。

纯 Python 实现。BM25 来自 Robertson/Sparck Jones 论文。
RRF 来自 2009 年 Cormack/Clarke/Buettcher 的 SIGIR 论文。

参考：
- ./docs/en.md
- 第 19 阶段第 64 课（切分器向该检索器输入）
- 第 19 阶段第 66 课（重排器消费融合后的 top-k）
- 第 19 阶段第 68 课（基于该检索器的评估框架）

运行：python3 code/main.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Doc:
    doc_id: str
    title: str
    body: str

    def field_text(self, field_name: str) -> str:
        return {"title": self.title, "body": self.body}.get(field_name, "")


# ---------------------------------------------------------------------------
# 分词器
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ---------------------------------------------------------------------------
# 从零实现 BM25
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    field_weights: dict[str, int] = field(default_factory=lambda: {"title": 3, "body": 1})
    docs: list[Doc] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    tf: list[Counter] = field(default_factory=list)
    avgdl: float = 0.0

    def _doc_tokens(self, doc: Doc) -> list[str]:
        out: list[str] = []
        for field_name, weight in self.field_weights.items():
            tokens = tokenize(doc.field_text(field_name))
            out.extend(tokens * weight)
        return out

    def add(self, doc: Doc) -> None:
        tokens = self._doc_tokens(doc)
        counts = Counter(tokens)
        self.docs.append(doc)
        self.doc_lens.append(len(tokens))
        self.tf.append(counts)
        for term in counts:
            self.df[term] += 1
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

    def search(self, query: str, k: int = 10) -> list[tuple[Doc, float]]:
        if not self.docs:
            return []
        q_terms = tokenize(query)
        n = len(self.docs)
        scores: list[float] = [0.0] * n
        for term in q_terms:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(zip(self.docs, scores), key=lambda x: -x[1])
        return [(d, s) for d, s in ranked[:k] if s > 0]


# ---------------------------------------------------------------------------
# 确定性模拟 embedding + 稠密检索器
# ---------------------------------------------------------------------------

def mock_embed(text: str, dim: int = 96) -> list[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = 0
        for ch in tok:
            h = (h * 1315423911) ^ ord(ch)
            h &= 0xFFFFFFFF
        vec[h % dim] += 1.0
        vec[(h >> 7) % dim] += 0.5
        # 加入 bigram 风格的混合，使同义词的分布与 BM25 有所不同。
        for i in range(len(tok) - 1):
            bg = (ord(tok[i]) * 31 + ord(tok[i + 1])) & 0xFFFFFFFF
            vec[bg % dim] += 0.25
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class DenseIndex:
    vectors: list[tuple[Doc, list[float]]] = field(default_factory=list)

    def add(self, doc: Doc) -> None:
        text = f"{doc.title}\n{doc.body}"
        self.vectors.append((doc, mock_embed(text)))

    def search(self, query: str, k: int = 10) -> list[tuple[Doc, float]]:
        qv = mock_embed(query)
        scored = [(d, cosine(qv, v)) for d, v in self.vectors]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# 倒数排名融合（Reciprocal Rank Fusion）
# ---------------------------------------------------------------------------

def rrf(
    rankings: list[list[tuple[Doc, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[Doc, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 长度必须与 rankings 长度一致")
    score: dict[str, float] = defaultdict(float)
    by_id: dict[str, Doc] = {}
    for w, ranks in zip(weights, rankings):
        for rank, (doc, _) in enumerate(ranks):
            score[doc.doc_id] += w * (1.0 / (k + rank + 1))
            by_id[doc.doc_id] = doc
    fused = sorted(score.items(), key=lambda x: -x[1])
    return [(by_id[did], s) for did, s in fused]


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------

@dataclass
class HybridRetriever:
    bm25: BM25Index = field(default_factory=BM25Index)
    dense: DenseIndex = field(default_factory=DenseIndex)
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0

    def add(self, doc: Doc) -> None:
        self.bm25.add(doc)
        self.dense.add(doc)

    def search(self, query: str, k_each: int = 10, k_out: int = 5) -> dict[str, list]:
        bm25_hits = self.bm25.search(query, k=k_each)
        dense_hits = self.dense.search(query, k=k_each)
        fused = rrf(
            [bm25_hits, dense_hits],
            k=self.rrf_k,
            weights=[self.bm25_weight, self.dense_weight],
        )
        return {
            "bm25": bm25_hits,
            "dense": dense_hits,
            "fused": fused[:k_out],
        }


# ---------------------------------------------------------------------------
# 固定语料库与示例查询
# ---------------------------------------------------------------------------

CORPUS = [
    Doc("d1", "AbortMultipartOnFail",
        "Aborts an in-flight S3 multipart upload and decrements the per-bucket retry budget when "
        "the upload fails. Wired into the central retry budget configuration."),
    Doc("d2", "Uploading large files",
        "When you upload a large file the storage service splits it into parts. The client must "
        "track each part. If the network drops the partial upload can be resumed or cancelled "
        "depending on the resume window. Cancelled uploads do not block subsequent attempts."),
    Doc("d3", "Per-bucket budgets",
        "Each storage bucket carries a retry budget that limits how often a failed operation can "
        "be retried within a window. Budget exhaustion triggers a cooldown period."),
    Doc("d4", "check_permission",
        "Authorization is centralized in the check_permission function which evaluates a policy "
        "against the principal, the resource, and the action. Both human users and service "
        "accounts pass through the same function."),
    Doc("d5", "Policy engine",
        "The policy engine wraps an Open Policy Agent runtime and exposes evaluate. Cached for "
        "a configured TTL to amortize repeated lookups."),
    Doc("d6", "Search ranking",
        "Production search engines combine lexical and semantic retrieval through a rank fusion "
        "step. The fusion is rank-based, not score-based, so the two modalities can be combined "
        "without per-corpus calibration."),
    Doc("d7", "Index sizing",
        "The vector index sits in memory. Plan for 1 KB per vector at 256 dimensions in float32 "
        "and add a small overhead for the graph structure."),
]


def print_ranking(label: str, hits: Iterable[tuple[Doc, float]], top: int = 5) -> None:
    print(f"  {label}:")
    for i, (doc, score) in enumerate(list(hits)[:top]):
        print(f"    {i + 1}. {doc.doc_id} ({doc.title})  score={score:.4f}")


def main() -> None:
    retriever = HybridRetriever()
    for d in CORPUS:
        retriever.add(d)

    queries = [
        ("AbortMultipartOnFail",
         "字面符号；BM25 轻松胜出，稠密检索也应通过哈希 token 把 d1 排在前面"),
        ("how do we handle cancelled uploads",
         "改写表述；稠密检索应找到上传文档；BM25 则不够直接"),
        ("centralized authorization for service accounts",
         "混合情形；两种模态都应命中授权文档"),
    ]

    for q, note in queries:
        print(f"\nquery: {q}\nnote:  {note}")
        result = retriever.search(q, k_each=5, k_out=5)
        print_ranking("bm25 ", result["bm25"])
        print_ranking("dense", result["dense"])
        print_ranking("fused", result["fused"])


if __name__ == "__main__":
    main()
