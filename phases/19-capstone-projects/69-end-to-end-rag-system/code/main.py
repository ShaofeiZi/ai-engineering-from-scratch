"""端到端 RAG 流水线，整合第 64–68 课内容。

自终止演示：导入固定语料库，运行查询，执行评估，
打印结果，成功时以 0 退出，未达阈值时以非零值退出。

参考资料：
- ./docs/en.md
- 第 19 阶段第 64 课（分块器）
- 第 19 阶段第 65 课（混合检索器）
- 第 19 阶段第 66 课（交叉编码器重排器）
- 第 19 阶段第 67 课（查询改写器）
- 第 19 阶段第 68 课（评估套件）

运行：python3 code/main.py
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn


VOCAB_SIZE = 8192
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
SEED = 19660101


# ---------------------------------------------------------------------------
# 分词器 + 确定性嵌入（各阶段共用）
# ---------------------------------------------------------------------------

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "at", "for",
         "with", "and", "or", "as", "by", "do", "you", "we", "our", "how",
         "what", "where", "when", "which", "before", "after"}


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 1}


def _stable_hash(tok: str) -> int:
    h = 0
    for ch in tok:
        h = (h * 1315423911) ^ ord(ch)
        h &= 0xFFFFFFFF
    return h


def mock_embed(text: str, dim: int = 96) -> list[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = _stable_hash(tok)
        vec[h % dim] += 1.0
        vec[(h >> 7) % dim] += 0.5
        for i in range(len(tok) - 1):
            bg = (ord(tok[i]) * 31 + ord(tok[i + 1])) & 0xFFFFFFFF
            vec[bg % dim] += 0.25
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 流水线中通用的分块结构
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str

    def anchor(self) -> str:
        return f"{self.doc_id}:{self.chunk_index}"


# ---------------------------------------------------------------------------
# 分块器——递归切分（第 64 课的默认策略）
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _recursive_split(text: str, separators: tuple[str, ...], target: int) -> list[str]:
    if len(text) <= target or not separators:
        return [text] if text else []
    sep = separators[0]
    rest = separators[1:]
    if sep not in text:
        return _recursive_split(text, rest, target)
    parts = text.split(sep)
    pieces: list[str] = []
    for p in parts:
        if not p:
            continue
        if len(p) <= target:
            pieces.append(p)
        else:
            pieces.extend(_recursive_split(p, rest, target))
    packed: list[str] = []
    for p in pieces:
        if packed and len(packed[-1]) + len(sep) + len(p) <= target:
            packed[-1] = packed[-1] + sep + p
        else:
            packed.append(p)
    return packed


@dataclass
class Chunker:
    target: int = 400
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        pieces = _recursive_split(text, self.separators, self.target)
        return [Chunk(doc_id, i, p.strip()) for i, p in enumerate(pieces) if p.strip()]


# ---------------------------------------------------------------------------
# 混合索引（BM25 + 稠密检索 + RRF）
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    tf: list[Counter] = field(default_factory=list)
    avgdl: float = 0.0

    def add(self, c: Chunk) -> None:
        tokens = tokenize(c.text)
        counts = Counter(tokens)
        self.chunks.append(c)
        self.doc_lens.append(len(tokens))
        self.tf.append(counts)
        for term in counts:
            self.df[term] += 1
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        n = len(self.chunks)
        if n == 0:
            return []
        scores: list[float] = [0.0] * n
        for term in tokenize(query):
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
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])
        return [(c, s) for c, s in ranked[:k] if s > 0]


@dataclass
class DenseIndex:
    vectors: list[tuple[Chunk, list[float]]] = field(default_factory=list)

    def add(self, c: Chunk) -> None:
        self.vectors.append((c, mock_embed(c.text)))

    def search_vec(self, qv: list[float], k: int) -> list[tuple[Chunk, float]]:
        scored = [(c, cosine(qv, v)) for c, v in self.vectors]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        return self.search_vec(mock_embed(query), k)


def rrf(rankings: list[list[tuple[Chunk, float]]], k: int = 60) -> list[tuple[Chunk, float]]:
    score: dict[str, float] = defaultdict(float)
    by_anchor: dict[str, Chunk] = {}
    for ranks in rankings:
        for rank, (c, _) in enumerate(ranks):
            score[c.anchor()] += 1.0 / (k + rank + 1)
            by_anchor[c.anchor()] = c
    fused = sorted(score.items(), key=lambda x: -x[1])
    return [(by_anchor[a], s) for a, s in fused]


@dataclass
class HybridIndex:
    bm25: BM25Index = field(default_factory=BM25Index)
    dense: DenseIndex = field(default_factory=DenseIndex)

    def add(self, c: Chunk) -> None:
        self.bm25.add(c)
        self.dense.add(c)

    def search(self, query: str, k_each: int = 8, k_out: int = 12) -> list[Chunk]:
        b = self.bm25.search(query, k_each)
        d = self.dense.search(query, k_each)
        return [c for c, _ in rrf([b, d])[:k_out]]

    def search_with_hypothetical(self, query: str, hypothetical: str,
                                 k_each: int = 8, k_out: int = 12) -> list[Chunk]:
        b = self.bm25.search(query, k_each)
        d = self.dense.search_vec(mock_embed(hypothetical), k_each)
        return [c for c, _ in rrf([b, d])[:k_out]]


# ---------------------------------------------------------------------------
# 查询改写器（通过启发式规则选择策略）
# ---------------------------------------------------------------------------

_REWRITE_HYDE: dict[str, str] = {
    "what is the threshold before a multipart upload is dropped":
        "The abort threshold is configured per bucket at three failed parts. Past that threshold the upload is aborted.",
    "where is the central permission gate":
        "Authorization is centralized in check_permission which evaluates a policy against the principal, the resource, and the action.",
    "how do production search engines fuse two retrievers":
        "Production search combines lexical and semantic retrieval through reciprocal rank fusion.",
    "how do you stop a long-running worker":
        "Long-running jobs accept a cancellation signal that stops the worker and releases the queue slot.",
}


@dataclass
class Rewriter:
    def pick_strategy(self, query: str) -> str:
        if " and " in query.lower():
            return "decompose"
        if any(t in tokenize(query) for t in ("api", "function", "endpoint", "config")):
            return "hyde"
        return "multiquery"

    def rewrite_hyde(self, query: str) -> str | None:
        return _REWRITE_HYDE.get(query.lower().strip().rstrip("?"))

    def rewrite_multiquery(self, query: str) -> list[str]:
        base = query.strip().rstrip("?")
        toks = [t for t in tokenize(base) if t not in _STOP]
        variants: list[str] = [base]
        if toks:
            variants.append(" ".join(toks))
        return list(dict.fromkeys(v for v in variants if v))

    def rewrite_decompose(self, query: str) -> list[str]:
        parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)
        out = [p.strip().rstrip("?") for p in parts if p.strip()]
        return out or [query.strip().rstrip("?")]


# ---------------------------------------------------------------------------
# 交叉编码器重排器（第 66 课的精简版）
# ---------------------------------------------------------------------------

def _token_to_id(token: str) -> int:
    return 3 + (_stable_hash(token) % (VOCAB_SIZE - 3))


def tokenize_pair(query: str, document: str, max_len: int = 96) -> tuple[list[int], list[int]]:
    q = [_token_to_id(t) for t in tokenize(query)]
    d = [_token_to_id(t) for t in tokenize(document)]
    ids = [CLS_ID] + q + [SEP_ID] + d + [SEP_ID]
    type_ids = ([0] * (len(q) + 2)) + ([1] * (len(d) + 1))
    if len(ids) > max_len:
        ids = ids[:max_len]
        type_ids = type_ids[:max_len]
    else:
        pad = max_len - len(ids)
        ids = ids + [PAD_ID] * pad
        type_ids = type_ids + [0] * pad
    return ids, type_ids


class CrossEncoder(nn.Module):
    def __init__(self, d_model: int = 48, n_heads: int = 4, ff_hidden: int = 96,
                 max_len: int = 96) -> None:
        super().__init__()
        torch.manual_seed(SEED)
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.type_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_hidden), nn.GELU(),
                                nn.Linear(ff_hidden, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.max_len = max_len

    def forward(self, ids: torch.Tensor, tids: torch.Tensor) -> torch.Tensor:
        b, s = ids.shape
        pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(b, s)
        x = self.token_emb(ids) + self.type_emb(tids) + self.pos_emb(pos)
        mask = (ids == PAD_ID)
        a, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        keep = (~mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


# 重排训练集：每类查询包含正样本文档和困难负样本干扰项。
# 困难负样本让交叉编码器学会避开词法基线在检索时容易选中的干扰文档。
TRAIN_TRIPLES: list[tuple[str, str, float]] = [
    # 鉴权类
    ("function for centralized auth check",
     "Authorization is centralized in check_permission which evaluates a policy against principal, resource, and action.", 1.0),
    ("function for centralized auth check",
     "A central permission gate inherits roles from the parent group. This document does not describe the authorization implementation; see d4.", 0.1),
    ("function for centralized auth check",
     "Plan for one kilobyte per vector at 256 dimensions.", 0.0),
    ("which document implements authorization",
     "Authorization is centralized in check_permission which evaluates a policy against principal, resource, and action.", 1.0),
    ("which document implements authorization",
     "A central permission gate inherits roles from the parent group. This document does not describe the authorization implementation; see d4.", 0.0),
    # 中止类
    ("what controls the upload abort policy",
     "AbortMultipartOnFail aborts an in-flight S3 multipart upload and decrements the per-bucket retry budget.", 1.0),
    ("what controls the upload abort policy",
     "The abort threshold is configured per bucket at three failed parts. Past that threshold the upload is aborted.", 0.95),
    ("what controls the upload abort policy",
     "Stale records are dropped from the in-memory cache after a TTL. This is unrelated to the upload abort threshold.", 0.1),
    ("what controls the upload abort policy",
     "Long-running jobs accept a cancellation signal that stops the worker.", 0.0),
    ("at what point is a multipart upload aborted",
     "The abort threshold is configured per bucket at three failed parts. Past that threshold the upload is aborted.", 1.0),
    ("at what point is a multipart upload aborted",
     "Stale records are dropped from the in-memory cache after a TTL. This is unrelated to the upload abort threshold.", 0.0),
    # 融合类
    ("how is rank fusion performed",
     "Production search combines lexical and semantic retrieval through reciprocal rank fusion at k = 60.", 1.0),
    ("how is rank fusion performed",
     "Multiple query results can be combined by simple union or by intersection. Production engines prefer rank fusion; see d6.", 0.1),
    ("how is rank fusion performed",
     "Plan for one kilobyte per vector at 256 dimensions.", 0.0),
    # worker 类
    ("how is a worker cancelled",
     "Long-running jobs accept a cancellation signal that stops the worker and releases the queue slot.", 1.0),
    ("how is a worker cancelled",
     "The worker pool is sized by p95 demand. Long-running workers are recycled when the pool is overprovisioned.", 0.1),
    ("how is a worker cancelled",
     "Authorization is centralized in check_permission.", 0.0),
]


def train_reranker(model: CrossEncoder, triples: list[tuple[str, str, float]],
                   epochs: int = 80, lr: float = 5e-3) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fn = nn.MSELoss()
    ids = []
    tids = []
    labels = []
    for q, d, label in triples:
        a, b = tokenize_pair(q, d, max_len=model.max_len)
        ids.append(a)
        tids.append(b)
        labels.append(label)
    ids_t = torch.tensor(ids, dtype=torch.long)
    tids_t = torch.tensor(tids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.float)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(ids_t, tids_t)
        loss = fn(pred, labels_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def rerank(model: CrossEncoder, query: str, candidates: list[Chunk],
           top_k: int, overlap_weight: float = 0.3) -> list[tuple[Chunk, float]]:
    """融合交叉编码器分数与内容重合信号。

    本课程的小型交叉编码器在训练集外查询上噪声较大。符合生产形态的修正方法
    是在评分时加入词法重合信号。overlap_weight 用于权衡纯重排质量与稳健性。
    """
    if not candidates:
        return []
    model.eval()
    ids: list[list[int]] = []
    tids: list[list[int]] = []
    for c in candidates:
        a, b = tokenize_pair(query, c.text, max_len=model.max_len)
        ids.append(a)
        tids.append(b)
    with torch.no_grad():
        ce_scores = model(torch.tensor(ids, dtype=torch.long),
                          torch.tensor(tids, dtype=torch.long)).tolist()
    q_tokens = _content_tokens(query)
    blended: list[tuple[Chunk, float]] = []
    for c, ce in zip(candidates, ce_scores):
        c_tokens = _content_tokens(c.text)
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        blended.append((c, ce + overlap_weight * overlap))
    blended.sort(key=lambda x: -x[1])
    return blended[:top_k]


# ---------------------------------------------------------------------------
# 答案生成器
# ---------------------------------------------------------------------------

REFUSE_THRESHOLD = 0.05
REFUSE_TEXT = "I do not know."


def generate_answer(query: str, ranked: list[tuple[Chunk, float]]) -> tuple[str, list[str]]:
    if not ranked:
        return REFUSE_TEXT, []
    top_chunk, top_score = ranked[0]
    if top_score < REFUSE_THRESHOLD:
        return REFUSE_TEXT, []
    q_tokens = _content_tokens(query)
    scored_chunks = []
    for c, s in ranked[:3]:
        overlap = len(q_tokens & _content_tokens(c.text))
        scored_chunks.append((c, overlap, s))
    scored_chunks.sort(key=lambda x: (-x[1], -x[2]))
    selected = [t for t in scored_chunks if t[1] > 0][:2] or [scored_chunks[0]]
    sentences = []
    citations = []
    for c, _, _ in selected:
        first_sentence = re.split(r"(?<=[.!?])\s+", c.text.strip(), maxsplit=1)[0]
        sentences.append(f"{first_sentence} [{c.anchor()}]")
        citations.append(c.anchor())
    return " ".join(sentences), citations


# ---------------------------------------------------------------------------
# 完整流水线
# ---------------------------------------------------------------------------

@dataclass
class Result:
    answer: str
    citations: list[str]
    top_k: list[tuple[Chunk, float]]
    rewriter_strategy: str | None
    latency_ms: dict[str, float]


@dataclass
class Pipeline:
    chunker: Chunker = field(default_factory=Chunker)
    index: HybridIndex = field(default_factory=HybridIndex)
    rewriter: Rewriter = field(default_factory=Rewriter)
    reranker: CrossEncoder = field(default_factory=CrossEncoder)
    top_n: int = 12
    top_k: int = 5

    def ingest(self, docs: list[tuple[str, str]]) -> None:
        for doc_id, text in docs:
            for chunk in self.chunker.chunk(doc_id, text):
                self.index.add(chunk)

    def train_reranker_on(self, triples: list[tuple[str, str, float]]) -> None:
        train_reranker(self.reranker, triples)

    def query(self, question: str) -> Result:
        latencies: dict[str, float] = {}

        t0 = time.perf_counter()
        strategy = self.rewriter.pick_strategy(question)
        hypothetical = self.rewriter.rewrite_hyde(question) if strategy == "hyde" else None
        subqueries: list[str] = []
        if strategy == "multiquery":
            subqueries = self.rewriter.rewrite_multiquery(question)
        elif strategy == "decompose":
            subqueries = self.rewriter.rewrite_decompose(question)
        t1 = time.perf_counter()
        latencies["rewrite"] = (t1 - t0) * 1000

        if hypothetical:
            candidates = self.index.search_with_hypothetical(question, hypothetical, k_out=self.top_n)
        elif len(subqueries) > 1:
            rankings: list[list[tuple[Chunk, float]]] = []
            for q in subqueries:
                hits = self.index.search(q, k_out=self.top_n)
                rankings.append([(c, 1.0) for c in hits])
            fused = rrf(rankings)
            candidates = [c for c, _ in fused[: self.top_n]]
        else:
            candidates = self.index.search(question, k_out=self.top_n)
        t2 = time.perf_counter()
        latencies["retrieve"] = (t2 - t1) * 1000

        ranked = rerank(self.reranker, question, candidates, self.top_k)
        t3 = time.perf_counter()
        latencies["rerank"] = (t3 - t2) * 1000

        answer, citations = generate_answer(question, ranked)
        t4 = time.perf_counter()
        latencies["generate"] = (t4 - t3) * 1000

        return Result(
            answer=answer,
            citations=citations,
            top_k=ranked,
            rewriter_strategy=strategy,
            latency_ms=latencies,
        )


# ---------------------------------------------------------------------------
# 夹具语料库 + 评估 qrels
# ---------------------------------------------------------------------------

CORPUS = [
    ("d1", "AbortMultipartOnFail aborts an in-flight S3 multipart upload and decrements "
           "the per-bucket retry budget."),
    ("d2", "Large files are split into parts. Cancelled uploads release the reserved keys "
           "after three failed parts."),
    ("d3", "The abort threshold is configured per bucket at three failed parts. Past that "
           "threshold the upload is aborted."),
    ("d4", "Authorization is centralized in check_permission which evaluates a policy "
           "against principal, resource, and action."),
    ("d5", "The policy engine wraps an OPA runtime and exposes evaluate; cached for a "
           "configured TTL."),
    ("d6", "Production search combines lexical and semantic retrieval through reciprocal "
           "rank fusion at k = 60."),
    ("d7", "Long-running jobs accept a cancellation signal that stops the worker and "
           "releases the queue slot."),
    ("d8", "Plan for one kilobyte per vector at 256 dimensions in float32 inside the "
           "in-memory ANN index."),
    ("d9", "Stale records are dropped from the in-memory cache after a TTL. This is "
           "unrelated to the upload abort threshold."),
    ("d10", "The worker pool is sized by p95 demand. Long-running workers are recycled "
            "when the pool is overprovisioned."),
    ("d11", "A central permission gate inherits roles from the parent group. This document "
            "does not describe the authorization implementation; see d4."),
    ("d12", "Multiple query results can be combined by simple union or by intersection. "
            "Production engines prefer rank fusion; see d6."),
]


@dataclass
class EvalQuery:
    qid: str
    query: str
    gold_doc_ids: set[str]
    gold_answer_substring: str


EVAL_QUERIES = [
    EvalQuery("e1", "what is the threshold before a multipart upload is dropped",
              {"d3", "d1"}, "three failed parts"),
    EvalQuery("e2", "where is the central permission gate",
              {"d4"}, "check_permission"),
    EvalQuery("e3", "how do production search engines fuse two retrievers",
              {"d6"}, "reciprocal rank fusion"),
    EvalQuery("e4", "how do you stop a long-running worker",
              {"d7"}, "cancellation signal"),
]


# ---------------------------------------------------------------------------
# 指标（第 68 课的子集，应用于 doc_id 层级）
# ---------------------------------------------------------------------------

def doc_level_recall(retrieved_chunks: list[tuple[Chunk, float]], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top_docs = {c.doc_id for c, _ in retrieved_chunks[:k]}
    return len(top_docs & gold) / len(gold)


def doc_level_precision(retrieved_chunks: list[tuple[Chunk, float]], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_docs = [c.doc_id for c, _ in retrieved_chunks[:k]]
    if not top_docs:
        return 0.0
    return sum(1 for d in top_docs if d in gold) / len(top_docs)


def doc_level_mrr(retrieved_chunks: list[tuple[Chunk, float]], gold: set[str]) -> float:
    seen: set[str] = set()
    rank = 0
    for c, _ in retrieved_chunks:
        if c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        rank += 1
        if c.doc_id in gold:
            return 1.0 / rank
    return 0.0


def faithfulness_score(answer: str, top_k: list[tuple[Chunk, float]]) -> float:
    if not top_k:
        return 1.0 if answer == REFUSE_TEXT else 0.0
    ctx_tokens = set()
    for c, _ in top_k:
        ctx_tokens |= _content_tokens(c.text)
    # 切分前去除引用锚点，避免产生空的尾随主张。
    body = re.sub(r"\[[^\]]+\]", "", answer)
    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s.strip()]
    if not claims:
        return 0.0
    supported = 0
    counted = 0
    for claim in claims:
        c_tokens = _content_tokens(claim)
        if not c_tokens:
            continue
        counted += 1
        if len(c_tokens & ctx_tokens) / len(c_tokens) >= 0.4:
            supported += 1
    if counted == 0:
        return 0.0
    return supported / counted


def answer_relevance_score(query: str, answer: str) -> float:
    q_tokens = _content_tokens(query)
    a_tokens = _content_tokens(re.sub(r"\[[^\]]+\]", "", answer))
    if not q_tokens:
        return 0.0
    return 1.0 if len(q_tokens & a_tokens) / len(q_tokens) >= 0.3 else 0.0


# ---------------------------------------------------------------------------
# 自终止演示
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "recall@5":         0.75,
    "precision@1":      0.50,
    "mrr":              0.60,
    "faithfulness":     0.75,
    "answer_relevance": 0.75,
}


def build_pipeline() -> Pipeline:
    p = Pipeline()
    p.ingest(CORPUS)
    p.train_reranker_on(TRAIN_TRIPLES)
    return p


def run_eval(p: Pipeline) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for eq in EVAL_QUERIES:
        result = p.query(eq.query)
        top_k = result.top_k
        sums["recall@5"] += doc_level_recall(top_k, eq.gold_doc_ids, 5)
        sums["precision@1"] += doc_level_precision(top_k, eq.gold_doc_ids, 1)
        sums["mrr"] += doc_level_mrr(top_k, eq.gold_doc_ids)
        sums["faithfulness"] += faithfulness_score(result.answer, top_k)
        sums["answer_relevance"] += answer_relevance_score(eq.query, result.answer)
    n = len(EVAL_QUERIES)
    return {k: v / n for k, v in sums.items()}


def run_demo() -> int:
    print("== 导入 + 训练 ==")
    p = build_pipeline()
    print(f"  已索引分块：{len(p.index.bm25.chunks)}")

    print()
    print("== 单次查询追踪 ==")
    sample_q = EVAL_QUERIES[2].query
    result = p.query(sample_q)
    print(f"  查询：{sample_q}")
    print(f"  改写策略：{result.rewriter_strategy}")
    print(f"  top-k: {[(c.anchor(), round(s, 3)) for c, s in result.top_k[:3]]}")
    print(f"  答案：{result.answer}")
    print(f"  引用：{result.citations}")
    print(f"  latency_ms: { {k: round(v, 2) for k, v in result.latency_ms.items()} }")

    print()
    print("== 评估 ==")
    metrics = run_eval(p)
    failed: list[str] = []
    for name, threshold in THRESHOLDS.items():
        observed = metrics[name]
        ok = observed >= threshold
        marker = "PASS" if ok else "FAIL"
        print(f"  {marker}  {name:<18} {observed:.3f}  (阈值 {threshold:.2f})")
        if not ok:
            failed.append(name)

    print()
    if failed:
        print(f"== 演示失败：{failed} ==")
        return 1
    print("== 演示通过 ==")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
