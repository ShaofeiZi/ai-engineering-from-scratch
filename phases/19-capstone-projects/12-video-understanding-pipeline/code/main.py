"""视频理解流水线——多向量场景索引脚手架。

关键架构原语是每个场景包含多个向量的索引，具有三种表示（caption、frame、
transcript），并行查询后通过倒数排名融合合并，再由时间 grounding 步骤在最佳
场景内选取子窗口。此脚手架实现了索引结构、三路查询融合和子窗口 grounding。

运行：python main.py
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field


EMB_DIM = 24


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def fake_embed(text: str) -> list[float]:
    v = [0.0] * EMB_DIM
    for tok in tokenize(text):
        h = hash(tok)
        v[h % EMB_DIM] += 1.0
        v[(h >> 8) % EMB_DIM] += 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 场景记录——多向量：caption / frame / transcript
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    video_id: str
    scene_id: int
    start_ms: int
    end_ms: int
    caption: str
    transcript: str
    frame_tags: str              # 帧 embedding 特征的替代数据
    caption_emb: list[float] = field(default_factory=list)
    frame_emb: list[float] = field(default_factory=list)
    transcript_emb: list[float] = field(default_factory=list)

    def embed(self) -> None:
        self.caption_emb = fake_embed(self.caption)
        self.frame_emb = fake_embed(self.frame_tags)
        self.transcript_emb = fake_embed(self.transcript)


SAMPLE = [
    Scene("vid_001", 0,       0,  32_000, "sunrise over skyline, drone footage",
          "we start here in tokyo",
          "skyline buildings dawn orange sky haze"),
    Scene("vid_001", 1,  32_000,  68_000, "busy intersection with pedestrians",
          "shibuya crossing after sunrise",
          "street people walking cars traffic signal"),
    Scene("vid_001", 2,  68_000, 132_000, "cars stopped at a red light",
          "let me count the vehicles approaching",
          "cars red light queue crossing lanes"),
    Scene("vid_001", 3, 132_000, 170_000, "kitchen scene chef pouring then stirring",
          "first we pour then we stir it slowly",
          "chef pan stove pour stir ingredient"),
    Scene("vid_001", 4, 170_000, 210_000, "chef plating the finished dish",
          "plated presentation of the dish",
          "plate garnish spoon finishing dish"),
    Scene("vid_002", 0,       0,  40_000, "ocean waves at sunset",
          "beautiful evening at the shore",
          "ocean waves sunset sky shore"),
]


# ---------------------------------------------------------------------------
# 三向量查询 + RRF 合并
# ---------------------------------------------------------------------------

def multi_vector_search(query: str, scenes: list[Scene], k: int = 5) -> list[tuple[Scene, float]]:
    qv = fake_embed(query)
    scored_caption = sorted(scenes, key=lambda s: -cosine(qv, s.caption_emb))
    scored_frame = sorted(scenes, key=lambda s: -cosine(qv, s.frame_emb))
    scored_transcript = sorted(scenes, key=lambda s: -cosine(qv, s.transcript_emb))

    fused: dict[tuple[str, int], float] = defaultdict(float)
    index: dict[tuple[str, int], Scene] = {}
    for ranks, stream in ((scored_caption, "cap"),
                          (scored_frame, "frm"),
                          (scored_transcript, "trn")):
        for rank, sc in enumerate(ranks):
            key = (sc.video_id, sc.scene_id)
            fused[key] += 1.0 / (60 + rank + 1)
            index[key] = sc

    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return [(index[k_], s) for k_, s in ranked[:k]]


# ---------------------------------------------------------------------------
# 时间 grounding stub——在最佳场景内细化起止时间
# ---------------------------------------------------------------------------

def ground_window(query: str, scene: Scene) -> tuple[int, int]:
    """替代实现：根据查询关键词的位置选择场景子窗口。"""
    q = set(tokenize(query))
    t_tokens = tokenize(scene.transcript)
    if not q or not t_tokens:
        return scene.start_ms, scene.end_ms
    positions = [i for i, w in enumerate(t_tokens) if w in q]
    if not positions:
        return scene.start_ms, scene.end_ms
    span = scene.end_ms - scene.start_ms
    start_frac = min(positions) / max(1, len(t_tokens))
    end_frac = (max(positions) + 1) / max(1, len(t_tokens))
    start = int(scene.start_ms + span * max(0.0, start_frac - 0.05))
    end = int(scene.start_ms + span * min(1.0, end_frac + 0.05))
    return start, end


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def main() -> None:
    scenes = SAMPLE
    for s in scenes:
        s.embed()

    queries = [
        ("how many cars pass through the intersection", False),
        ("what happened first pour or stir", False),
        ("plating of the dish", True),
        ("ocean at sunset", True),
    ]

    for q, descriptive in queries:
        print(f"\n问题：{q}（描述性={descriptive}）")
        hits = multi_vector_search(q, scenes, k=3)
        for sc, score in hits:
            print(f"  场景 {sc.video_id}/{sc.scene_id} @ [{fmt_ms(sc.start_ms)}-{fmt_ms(sc.end_ms)}] "
                  f"分数={score:.4f}  caption='{sc.caption[:40]}'")
        top = hits[0][0]
        start, end = ground_window(q, top)
        print(f"  grounding 窗口：[{fmt_ms(start)}-{fmt_ms(end)}] "
              f"（从 {fmt_ms(top.start_ms)}-{fmt_ms(top.end_ms)} 缩小）")


if __name__ == "__main__":
    main()
