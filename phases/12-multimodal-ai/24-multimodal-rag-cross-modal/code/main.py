"""多模态 RAG 示例 — 三个检索器 + 分数融合 + 基于检索结果的生成器。

标准库实现。合成餐厅语料库，包含文本评论、图像特征标签和
音频氛围评分。运行三个检索器，融合分数，生成带引用的示例
答案，并演示低置信度时的智能体改写。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Restaurant:
    id: str
    name: str
    review_text: str
    image_tags: list[str]
    ambient_db: float


CORPUS = [
    Restaurant("r1", "Sunday Plant Bistro",
               "best vegan brunch, quiet mornings, lots of windows", ["natural_light", "minimal"], 38),
    Restaurant("r2", "Orange Grove Cafe",
               "all-day vegan brunch, noisy music, industrial style", ["industrial"], 68),
    Restaurant("r3", "Vine & Leaf",
               "vegan lunch, dim lighting", ["warm_lighting"], 55),
    Restaurant("r4", "Morning Glow",
               "vegan brunch, airy space, lots of sun", ["natural_light", "airy"], 42),
    Restaurant("r5", "Steak Central",
               "steakhouse, loud atmosphere", ["dark"], 72),
]

TEXT_QUERY_ALIASES = {
    "安静": ("quiet",),
    "纯素": ("vegan",),
    "早午餐": ("brunch",),
    "自然光": ("windows", "sun"),
    "阳光": ("sun",),
    "通透": ("airy",),
}


def text_retrieve(query: str) -> dict[str, float]:
    """对查询与评论文本进行粗略的关键词匹配。"""
    # 保持英文检索语料稳定；在精确关键词匹配之前，先将中文 UI 查询
    # 映射到同一套面向机器的词汇。
    keywords = [w.lower() for w in query.split() if len(w) > 2 and w.isascii()]
    for phrase, aliases in TEXT_QUERY_ALIASES.items():
        if phrase in query:
            keywords.extend(aliases)
    scores = {}
    for r in CORPUS:
        text = r.review_text.lower()
        s = sum(text.count(k) for k in keywords)
        scores[r.id] = s / len(keywords) if keywords else 0
    return scores


def image_retrieve(query: str) -> dict[str, float]:
    q = query.lower()
    tag_hints = []
    if "light" in q or "sun" in q or "自然光" in q or "阳光" in q:
        tag_hints.append("natural_light")
    if "airy" in q or "spacious" in q or "通透" in q or "宽敞" in q:
        tag_hints.append("airy")
    if "minimal" in q or "极简" in q:
        tag_hints.append("minimal")
    scores = {}
    for r in CORPUS:
        s = sum(1.0 for t in tag_hints if t in r.image_tags)
        scores[r.id] = s / max(1, len(tag_hints))
    return scores


def audio_retrieve(query: str) -> dict[str, float]:
    q = query.lower()
    scores = {}
    if "quiet" in q or "calm" in q or "安静" in q or "平静" in q:
        for r in CORPUS:
            scores[r.id] = max(0.0, 1.0 - r.ambient_db / 80.0)
    else:
        for r in CORPUS:
            scores[r.id] = 0.5
    return scores


def fuse(scores_list: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    fused = {}
    for r in CORPUS:
        s = 0.0
        for w, scores in zip(weights, scores_list):
            s += w * scores.get(r.id, 0)
        fused[r.id] = s
    return fused


def top_k(scored: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(scored.items(), key=lambda x: -x[1])[:k]


def grounded_generate(query: str, ranked: list[tuple[str, float]]) -> str:
    lines = [f"查询答案：'{query}'"]
    for i, (rid, score) in enumerate(ranked, 1):
        r = next(x for x in CORPUS if x.id == rid)
        lines.append(
            f"  {i}. {r.name}（得分 {score:.2f}）"
            f" [评论 {rid}] [图像标签 {r.image_tags}] [环境声 {r.ambient_db}dB]")
    return "\n".join(lines)


def agentic_loop(query: str, confidence_floor: float = 0.8) -> str:
    t = text_retrieve(query)
    i = image_retrieve(query)
    a = audio_retrieve(query)
    fused = fuse([t, i, a], [0.3, 0.4, 0.3])
    top = top_k(fused, k=3)
    confidence = top[0][1] if top else 0

    trace = [f"第 1 轮：最高项={top[0]}  置信度={confidence:.2f}"]
    if confidence < confidence_floor:
        trace.append("  置信度低，正在改写查询")
        query2 = query + " 明亮窗户 低噪声"
        i2 = image_retrieve(query2)
        a2 = audio_retrieve(query2)
        fused = fuse([t, i2, a2], [0.3, 0.5, 0.2])
        top = top_k(fused, k=3)
        trace.append(f"第 2 轮：最高项={top[0]}  置信度={top[0][1]:.2f}")
    return "\n".join(trace) + "\n\n" + grounded_generate(query, top)


def surveys_table() -> None:
    print("\n2025 年多模态 RAG 综述")
    print("-" * 60)
    rows = [
        ("Abootorabi 等", "2025 年 2 月", "全面的分类体系"),
        ("Mei 等",        "2025 年 4 月", "子任务基准 + 失败模式"),
        ("Zhao 等",       "2025 年 3 月", "聚焦视觉，深入讨论 ColPali"),
    ]
    for name, date, note in rows:
        print(f"  {name:<22}{date:<10}{note}")


def main() -> None:
    print("=" * 60)
    print("多模态 RAG（第12阶段，第24课）")
    print("=" * 60)

    query = "帮我找一家安静、有自然光的纯素早午餐厅"
    print(f"\n查询: {query}")
    print("-" * 60)
    result = agentic_loop(query, confidence_floor=0.7)
    print(result)

    surveys_table()

    print("\n融合策略")
    print("-" * 60)
    print("  分数融合：加权和，简单，快速")
    print("  MoE 融合  ：门控路由到专家，可学习，需要训练")
    print("  注意力    ：小型网络对检索到的条目加权")
    print("  默认：分数融合 + 略微偏向主导模态")


if __name__ == "__main__":
    main()
