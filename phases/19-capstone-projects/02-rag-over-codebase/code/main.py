"""代码 RAG——感知 AST 的分块与混合检索脚手架。

这里关键的架构原语是带排名融合的混合检索：两个索引结构（稠密向量、BM25）
并行运行，结果通过倒数排名融合合并，再由重排器选出最终 top-k。此脚手架使用
标准库实现两部分：朴素稠密索引（采用基于哈希的伪 embedding，使循环可在离线
环境中确定性运行），以及从零实现的真实 BM25。融合与重排逻辑才是重点。

运行：python main.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 分块结构——感知 AST 的函数级分块
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    repo: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    body: str
    summary: str = ""

    def anchor(self) -> str:
        return f"{self.repo}/{self.path}:{self.start_line}-{self.end_line}"


SAMPLE_CORPUS = [
    Chunk("uploader", "services/retry.go", 122, 148, "AbortMultipartOnFail",
          "if ctx.Err() != nil { return abort() }; decrement bucket budget; retry with backoff",
          "aborts an in-flight S3 multipart upload and decrements the per-bucket retry budget"),
    Chunk("uploader", "config/budgets.yaml", 34, 51, "bucket_budget",
          "per_bucket_budget: 64; backoff_ms: [100, 500, 2500]; abort_threshold: 3",
          "declares the retry budget and exponential backoff schedule per S3 bucket"),
    Chunk("client", "libs/s3client/multipart.ts", 44, 61, "abortUpload",
          "await s3.abortMultipartUpload({Bucket, Key, UploadId}); metrics.inc('s3.abort')",
          "client-side S3 multipart abort with metrics instrumentation"),
    Chunk("auth", "services/authz/check.py", 12, 38, "check_permission",
          "def check_permission(user, resource, action): return policy.evaluate(user, resource, action)",
          "central authorization gateway evaluating an OPA policy for user-resource-action"),
    Chunk("auth", "libs/policy/opa.py", 88, 110, "evaluate",
          "def evaluate(user, resource, action): return self.engine.query('authz', input=...)",
          "OPA policy engine query wrapper for authorization checks"),
    Chunk("catalog", "services/search/query.rs", 200, 240, "rank_fusion",
          "pub fn rank_fusion(dense: Vec<Hit>, sparse: Vec<Hit>) -> Vec<Hit>",
          "reciprocal rank fusion of dense and sparse retrieval results"),
]


# ---------------------------------------------------------------------------
# 朴素稠密索引——用于测试脚手架的确定性伪 embedding
# ---------------------------------------------------------------------------

def fake_embed(text: str, dim: int = 64) -> list[float]:
    """基于哈希的确定性 embedding，用作 Voyage-code-3 的替代实现。"""
    vec = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        h = hash(tok)
        vec[h % dim] += 1.0
        vec[(h >> 8) % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class DenseIndex:
    vectors: list[tuple[Chunk, list[float]]] = field(default_factory=list)

    def add(self, chunk: Chunk) -> None:
        text = f"{chunk.symbol}\n{chunk.summary}\n{chunk.body}"
        self.vectors.append((chunk, fake_embed(text)))

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        qv = fake_embed(query)
        scored = [(c, cosine(qv, v)) for c, v in self.vectors]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# 从零实现 BM25——真实算法，文档类型为 Chunk
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    docs: list[Chunk] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    tf: list[Counter] = field(default_factory=list)
    avgdl: float = 0.0

    def add(self, chunk: Chunk) -> None:
        # 字段加权：symbol x4、summary x2、body x1
        tokens = (tokenize(chunk.symbol) * 4 +
                  tokenize(chunk.summary) * 2 +
                  tokenize(chunk.body))
        counts = Counter(tokens)
        self.docs.append(chunk)
        self.doc_lens.append(len(tokens))
        self.tf.append(counts)
        for term in counts:
            self.df[term] += 1
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        q_terms = tokenize(query)
        n = len(self.docs)
        scores: list[float] = [0.0] * n
        for term in q_terms:
            if term not in self.df:
                continue
            idf = math.log((n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
            for i, counts in enumerate(self.tf):
                if term not in counts:
                    continue
                f = counts[term]
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(zip(self.docs, scores), key=lambda x: -x[1])
        return [(c, s) for c, s in ranked[:k] if s > 0]


# ---------------------------------------------------------------------------
# 倒数排名融合——混合检索的合并步骤
# ---------------------------------------------------------------------------

def rrf(dense: list[tuple[Chunk, float]], sparse: list[tuple[Chunk, float]],
        k_rrf: int = 60) -> list[tuple[Chunk, float]]:
    score: dict[str, float] = defaultdict(float)
    by_anchor: dict[str, Chunk] = {}
    for rank, (c, _) in enumerate(dense):
        score[c.anchor()] += 1.0 / (k_rrf + rank + 1)
        by_anchor[c.anchor()] = c
    for rank, (c, _) in enumerate(sparse):
        score[c.anchor()] += 1.0 / (k_rrf + rank + 1)
        by_anchor[c.anchor()] = c
    fused = sorted(score.items(), key=lambda x: -x[1])
    return [(by_anchor[a], s) for a, s in fused]


# ---------------------------------------------------------------------------
# stub 重排器——cross-encoder 替代实现；按查询与符号的重叠度重排
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[tuple[Chunk, float]],
           top_k: int = 5) -> list[tuple[Chunk, float]]:
    q_toks = set(tokenize(query))
    out: list[tuple[Chunk, float]] = []
    for c, prior in candidates:
        symbol_overlap = len(q_toks & set(tokenize(c.symbol))) * 3
        summary_overlap = len(q_toks & set(tokenize(c.summary)))
        out.append((c, prior + 0.3 * symbol_overlap + 0.1 * summary_overlap))
    out.sort(key=lambda x: -x[1])
    return out[:top_k]


# ---------------------------------------------------------------------------
# 编排器——完整的检索 -> 融合 -> 重排流程
# ---------------------------------------------------------------------------

def answer(query: str, dense: DenseIndex, bm25: BM25Index) -> dict[str, object]:
    dense_hits = dense.search(query, k=10)
    sparse_hits = bm25.search(query, k=10)
    fused = rrf(dense_hits, sparse_hits)
    top = rerank(query, fused, top_k=5)
    citations = [c.anchor() for c, _ in top]
    return {
        "query": query,
        "dense_top": [c.anchor() for c, _ in dense_hits[:3]],
        "sparse_top": [c.anchor() for c, _ in sparse_hits[:3]],
        "fused_top": [c.anchor() for c, _ in fused[:5]],
        "rerank_top": citations,
    }


def main() -> None:
    dense = DenseIndex()
    bm25 = BM25Index()
    for ch in SAMPLE_CORPUS:
        dense.add(ch)
        bm25.add(ch)

    for q in ("how is S3 multipart abort wired into retry budget",
              "where is authorization centralized",
              "how does rank fusion work"):
        result = answer(q, dense, bm25)
        print(f"Q: {result['query']}")
        print(f"  稠密检索：{result['dense_top']}")
        print(f"  稀疏检索：{result['sparse_top']}")
        print(f"  融合结果：{result['fused_top']}")
        print(f"  重排结果：{result['rerank_top']}")
        print()


if __name__ == "__main__":
    main()
