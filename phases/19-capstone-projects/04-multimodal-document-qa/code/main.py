"""多模态文档问答——ColPali 风格的后期交互脚手架。

关键架构原语是后期交互检索：每个查询 token 都会与每个文档 patch 计算分数，
再对每个查询 token 的 MaxSim 求和并返回 top-k 页面。此脚手架基于合成 patch
embedding 端到端实现 MaxSim，无需加载真实 ColQwen 模型即可观察算法，并包含
DocPruner 风格的 patch 剪枝。

运行：python main.py
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# patch embedding——每页使用伪造的 16 维 patch 向量
# ---------------------------------------------------------------------------

EMB_DIM = 16


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def hash_embed(tok: str) -> list[float]:
    rnd = random.Random(hash(tok) & 0xFFFFFFFF)
    v = [rnd.gauss(0, 1) for _ in range(EMB_DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@dataclass
class Page:
    doc_id: str
    page_num: int
    content_tokens: list[str]          # 页面内容的替代数据
    patches: list[list[float]] = field(default_factory=list)

    def embed_patches(self) -> None:
        """多向量表示：每个内容 token 都会变成一个 patch 向量。"""
        self.patches = [hash_embed(t) for t in self.content_tokens]


# ---------------------------------------------------------------------------
# DocPruner——按范数差异保留排名靠前的一部分 patch
# ---------------------------------------------------------------------------

def doc_prune(patches: list[list[float]], keep_fraction: float = 0.5) -> list[list[float]]:
    """保留单 patch 范数最高的 patch（虽然只是信息密度的粗略代理，
    但符合 DocPruner 的直觉：丢弃低信号 patch）。"""
    scored = [(sum(abs(x) for x in p), p) for p in patches]
    scored.sort(key=lambda x: -x[0])
    keep_n = max(1, int(len(scored) * keep_fraction))
    return [p for _, p in scored[:keep_n]]


# ---------------------------------------------------------------------------
# MaxSim 后期交互——ColPali / ColQwen 的算法核心
# ---------------------------------------------------------------------------

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def max_sim_score(query_tokens: list[list[float]],
                  doc_patches: list[list[float]]) -> float:
    """对每个查询 token embedding，取其与任意文档 patch 的最大点积；
    再对所有查询 token 求和。这就是 MaxSim / 后期交互。"""
    total = 0.0
    for q in query_tokens:
        best = -1e9
        for p in doc_patches:
            s = dot(q, p)
            if s > best:
                best = s
        total += best
    return total


# ---------------------------------------------------------------------------
# 索引 + 检索——按 MaxSim 排序取 top-k
# ---------------------------------------------------------------------------

@dataclass
class Index:
    pages: list[Page] = field(default_factory=list)

    def add(self, p: Page) -> None:
        self.pages.append(p)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Page, float]]:
        q_tokens = [hash_embed(t) for t in tokenize(query)]
        scored = [(pg, max_sim_score(q_tokens, pg.patches)) for pg in self.pages]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# 合成语料库——十页内容，涵盖表格、图表、手写内容和文本
# ---------------------------------------------------------------------------

CORPUS = [
    ("10k-2024", 88, "segment EMEA operating margin 18.2 to 16.8 decline 140bp table four"),
    ("10k-2024", 92, "MDA operating performance EMEA macro headwinds FX impact narrative"),
    ("10k-2024", 14, "executive summary revenue growth 7 percent consolidated totals"),
    ("paper-vidore-v3", 3, "late interaction multi vector retrieval ColPali ColQwen benchmark"),
    ("paper-vidore-v3", 7, "nDCG results table vision first vs OCR then text columns"),
    ("paper-m3docrag", 2, "M3DocVQA multi page reasoning evaluation protocol"),
    ("handwritten-lab", 5, "experiment notes circuit board pH readings handwritten"),
    ("handwritten-lab", 6, "graph with annotated error bars figure 3 caption"),
    ("chart-report", 11, "line chart revenue by segment EMEA americas APAC Q1 Q4"),
    ("chart-report", 12, "bar chart operating margin by segment with 2023 2024 comparison"),
]


def build_index(prune: bool = True) -> Index:
    idx = Index()
    for doc, page, text in CORPUS:
        p = Page(doc_id=doc, page_num=page, content_tokens=tokenize(text))
        p.embed_patches()
        if prune:
            p.patches = doc_prune(p.patches, keep_fraction=0.5)
        idx.add(p)
    return idx


def main() -> None:
    print("=== 使用 DocPruner（保留 50% patch）构建索引 ===")
    idx = build_index(prune=True)
    print(f"已索引页面数：{len(idx.pages)}")

    queries = [
        "what was the 2024 operating margin change for EMEA",
        "late interaction retrieval vs OCR",
        "handwritten experimental figures with error bars",
        "bar chart comparing segment margins",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        hits = idx.retrieve(q, k=3)
        for pg, score in hits:
            print(f"  score={score:+.3f}  {pg.doc_id} p.{pg.page_num}")

    # 剪枝消融实验
    print("\n=== 消融实验：关闭剪枝与启用剪枝 ===")
    full = build_index(prune=False)
    pruned = build_index(prune=True)
    q = "chart comparing segment margins"
    full_top = [(p.doc_id, p.page_num) for p, _ in full.retrieve(q, 3)]
    prn_top = [(p.doc_id, p.page_num) for p, _ in pruned.retrieve(q, 3)]
    print(f"  完整索引 top-3：{full_top}")
    print(f"  剪枝索引 top-3：{prn_top}")
    print(f"  重叠数量      ：{len(set(full_top) & set(prn_top))}/3")


if __name__ == "__main__":
    main()
