"""ColPali 玩具示例：patch 编码器 + MaxSim 检索 — 仅用标准库。

五个模拟"页面"的 patch 嵌入，三个带 token 嵌入的文本查询，
MaxSim 评分并使用 top-k 检索。打印排序后的页面及解读说明。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

random.seed(7)


@dataclass
class Page:
    doc_id: str
    patches: list[list[float]]


@dataclass
class Query:
    text: str
    tokens: list[list[float]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-8
    nb = math.sqrt(sum(y * y for y in b)) + 1e-8
    return dot / (na * nb)


def maxsim(query_tokens: list[list[float]],
           patches: list[list[float]]) -> float:
    """ColBERT MaxSim：对查询 token 取各 patch 中的最大值后求和。"""
    s = 0.0
    for q in query_tokens:
        best = max(cosine(q, p) for p in patches)
        s += best
    return s


def random_emb(dim: int, bias: int = 0) -> list[float]:
    return [random.gauss(bias / 10.0, 1.0) for _ in range(dim)]


def build_pages(n_pages: int = 5, n_patches: int = 16, dim: int = 32) -> list[Page]:
    pages = []
    topics = ["finance", "science", "legal", "medical", "engineering"]
    for i, topic in enumerate(topics[:n_pages]):
        bias = i + 1
        patches = [random_emb(dim, bias) for _ in range(n_patches)]
        pages.append(Page(doc_id=f"page_{i}_{topic}", patches=patches))
    return pages


def build_queries(dim: int = 32) -> list[Query]:
    random.seed(100)
    queries = []
    for text, bias in [("第三季度收入增长", 1),
                       ("引理 3 的证明", 2),
                       ("患者诊断", 4)]:
        tokens = [random_emb(dim, bias) for _ in range(4)]
        queries.append(Query(text=text, tokens=tokens))
    return queries


def retrieve(query: Query, pages: list[Page], k: int = 3) -> list[tuple[str, float]]:
    scored = [(p.doc_id, maxsim(query.tokens, p.patches)) for p in pages]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def storage_estimate() -> None:
    print("\n存储开销——COLPALI 与文本 RAG 对比")
    print("-" * 60)
    print(f"  {'系统':<24}{'字节/页':<14}  说明")
    print(f"  {'文本 RAG 768d 双编码器':<24}{'3.0 KB':<14}  每个分块一个向量")
    print(f"  {'ColPali 原始（729 x 128）':<24}{'365 KB':<14}  每个 patch 一个向量")
    print(f"  {'ColPali PQ 8x':<24}{'46 KB':<14}  OPQ 压缩")
    print(f"  {'VisRAG 双编码器':<24}{'3.0 KB':<14}  每个页面一个向量")


def compare_maxsim_vs_mean() -> None:
    print("\nMAXSIM 与平均相似度")
    print("-" * 60)
    random.seed(42)
    q_tokens = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1]]
    strong_patch = [0.9, 0.9, 0.0]
    other_patches = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0]]
    patches = [strong_patch] + other_patches
    max_score = maxsim(q_tokens, patches)
    mean_score = sum(cosine(q, p) for q in q_tokens for p in patches) / (
        len(q_tokens) * len(patches))
    print(f"  MaxSim ：{max_score:.3f}   （捕获每个查询 token 的最佳匹配）")
    print(f"  Mean   ：{mean_score:.3f}   （被无关 patch 稀释）")
    print("MaxSim 的选择性正是后期交互在 bi-encoder 召回上胜出的原因")


def main() -> None:
    print("=" * 60)
    print("COLPALI 视觉原生 RAG（第 12 阶段，第 23 课）")
    print("=" * 60)

    pages = build_pages(n_pages=5, n_patches=16, dim=32)
    queries = build_queries(dim=32)

    print("\n索引与检索")
    print("-" * 60)
    for q in queries:
        hits = retrieve(q, pages, k=3)
        print(f"  查询：'{q.text}'")
        for page_id, score in hits:
            print(f"    {page_id:<22}  得分={score:+.3f}")
        print()

    compare_maxsim_vs_mean()
    storage_estimate()

    print("\n端到端流水线")
    print("-" * 60)
    print("  摄取 ：PDF -> 页面 PNG -> PaliGemma -> patch 向量（已缓存）")
    print("  查询 ：用户文本 -> token -> MaxSim -> top-k 个页面")
    print("  生成 ：top-k 张页面图像 + 查询 -> Qwen2.5-VL -> 答案")
    print("  无需 OCR，无需分块，无版面信息丢失")


if __name__ == "__main__":
    main()
