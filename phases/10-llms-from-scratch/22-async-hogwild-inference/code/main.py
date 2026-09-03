"""(原始内容存档于2017-09-29). 推论玩具模拟器——stdlib Python.

两名工人同时与共享的“ph4”缓存对峙。 每个工人读
缓存,并使用
简单的协调heuristic:如果其他工人已经生产足够了
类别中的tokens,切换。

产出:
- 固定职档预算产生的总工作-tokens
- 墙上加速度与单一工人基线
- 哪些工人写了 "ph6 " 和什么类别
- 协调加权检查,显示协调不力的影响

不是忠实的LLM模拟。 关键是展示新工作
由 share-cache 驱动的分区读取 。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Literal


Category = Literal["A", "B", "noise", "coord"]


@dataclass
class SharedCache:
    tokens: List[tuple[int, Category]] = field(default_factory=list)

    def counts(self) -> dict:
        c = {"A": 0, "B": 0, "noise": 0, "coord": 0}
        for _, cat in self.tokens:
            c[cat] += 1
        return c


@dataclass
class Worker:
    id: int
    intended: Category
    coordination_weight: float
    rng: random.Random


def decide_next_category(worker: Worker, cache: SharedCache,
                         target_per_category: int) -> Category:
    """读取共享缓存 。 概率协调  重量, 切换
改为最少工作类别(注解冗余)。 否则留下
在工人的预定类别。 坐标 重量=0
无法协调的工人(完全冗余)。 重量=1个型号
理想的推理模式协调。
"""
    if worker.rng.random() < 0.05:
        return "noise"

    counts = cache.counts()
    base = worker.intended

    if worker.rng.random() < worker.coordination_weight:
        candidates = sorted(("A", "B"), key=lambda c: counts[c])
        return candidates[0]

    if worker.rng.random() < 0.1:
        return "coord"

    return base


def run_hogwild(n_workers: int, step_budget: int, target_per_category: int,
                coordination_weight: float, seed: int = 42) -> dict:
    """所有工人都不遵守A类。 协调使他们有分歧。
没有协调,冗余的tokens(同一类别来自多个)
工人)计算一次. 通过协调,工人选择不同的
因此,每个token的分类是独一无二的,有助于全面进展。"""
    cache = SharedCache()
    workers = []
    for i in range(n_workers):
        workers.append(Worker(
            id=i, intended="A",
            coordination_weight=coordination_weight,
            rng=random.Random(seed + i),
        ))

    trace: List[tuple[int, Category, str]] = []
    step = 0
    progress = 0
    while step < step_budget:
        this_step_categories: List[tuple[int, Category]] = []
        for w in workers:
            cat = decide_next_category(w, cache, target_per_category)
            cache.tokens.append((w.id, cat))
            this_step_categories.append((w.id, cat))

        seen_work_categories = set()
        for w_id, cat in this_step_categories:
            tag = "redundant"
            if cat in ("A", "B") and cat not in seen_work_categories:
                seen_work_categories.add(cat)
                progress += 1
                tag = "unique"
            trace.append((w_id, cat, tag))
        step += 1

    counts = cache.counts()
    work_tokens = counts["A"] + counts["B"]
    return {
        "workers": n_workers,
        "step_budget": step_budget,
        "tokens_emitted": len(cache.tokens),
        "work_tokens": work_tokens,
        "unique_progress": progress,
        "category_counts": counts,
        "coord_tokens": counts["coord"],
        "noise_tokens": counts["noise"],
        "tokens_per_step": len(cache.tokens) / step_budget,
        "work_per_step": work_tokens / step_budget,
        "progress_per_step": progress / step_budget,
        "sample_trace": trace[:12],
    }


def expected_speedup(T_serial: int, p: float, c: int, N: int,
                     steps_per_worker: int) -> float:
    parallel = T_serial * ((1 - p) + p / N) + c * N
    return T_serial / parallel


def main() -> None:
    print("=" * 70)
    print("Hogwild 推理模拟器（第 10 阶段，第 22 课）")
    print("=" * 70)
    print()

    print("-" * 70)
    print("步骤 1：基线——单 worker，200 步")
    print("-" * 70)
    r_1 = run_hogwild(n_workers=1, step_budget=200, target_per_category=100,
                      coordination_weight=0.8)
    print(f"  输出 token 数：{r_1['tokens_emitted']}")
    print(f"  work-tokens       : {r_1['work_tokens']}  ({r_1['work_per_step']:.2f} / step)")
    print(f"  唯一进展量： {r_1['unique_progress']}（每步 {r_1['progress_per_step']:.2f}）")
    print(f"  类别计数：   {r_1['category_counts']}")
    print()

    print("-" * 70)
    print("步骤 2：Hogwild——2 个 worker，共享 cache，轻量协调")
    print("-" * 70)
    r_2 = run_hogwild(n_workers=2, step_budget=200, target_per_category=100,
                      coordination_weight=0.8)
    print(f"  输出 token 数：{r_2['tokens_emitted']}（每步 {r_2['tokens_per_step']:.2f}）")
    print(f"  work-tokens       : {r_2['work_tokens']}  ({r_2['work_per_step']:.2f} / step)")
    print(f"  唯一进展量： {r_2['unique_progress']}（每步 {r_2['progress_per_step']:.2f}）")
    print(f"  类别计数：   {r_2['category_counts']}")
    print(f"  相对 N=1 的加速比：{r_2['unique_progress'] / r_1['unique_progress']:.2f}x")
    print()

    print("-" * 70)
    print("步骤3:协调加权扫描(N=2,同一步骤预算)")
    print("-" * 70)
    print(f"  {'coord 重量':>14}  {'进展':>10}  {'速度对 N=1':>15}")
    for cw in (0.0, 0.2, 0.5, 0.8, 1.0):
        r = run_hogwild(n_workers=2, step_budget=200, target_per_category=100,
                        coordination_weight=cw)
        speedup = r["unique_progress"] / r_1["unique_progress"]
        print(f"  {cw:>14.2f}  {r['unique_progress']:>10}  {speedup:>15.2f}x")
    print("(编码重量0.0=两名工人都留在A类=完全冗余)")
    print()

    print("-" * 70)
    print("第四步:阿姆达尔式理论加速")
    print("-" * 70)
    T_serial = 10_000
    print("  推理任务 = 10000 个解码 token")
    print("  c = 每个 worker 的协调开销")
    print(f"  {'p':>5}  " + "".join(
        f"{f'N={N}':>10}" for N in (2, 4, 8)))
    for p in (0.3, 0.5, 0.7, 0.9):
        row = f"  {p:>5.2f}  "
        for N in (2, 4, 8):
            s = expected_speedup(T_serial=T_serial, p=p, c=200, N=N,
                                 steps_per_worker=T_serial // N)
            row += f"{s:>9.2f}x"
        print(row)
    print("(值: Hogwild! 超過序列单工)")
    print()

    print("-" * 70)
    print("步骤5:最坏情况(任务短,协调不力)")
    print("-" * 70)
    print(f"  {'p':>5}  " + "".join(
        f"{f'N={N}':>10}" for N in (2, 4, 8)))
    for p in (0.1, 0.3, 0.5):
        row = f"  {p:>5.2f}  "
        for N in (2, 4, 8):
            s = expected_speedup(T_serial=1000, p=p, c=150, N=N,
                                 steps_per_worker=1000 // N)
            row += f"{s:>9.2f}x"
        print(row)
    print("(短1000-token任务,150-token协调间接费用)")
    print("数值低于 1.0 表示并行推理比串行更慢")
    print()

    print("要点：Hogwild 的加速比取决于可并行比例 p 和协调开销 c。")
    print("p > 0.5 且每步协调开销较低时最有优势；短对话中，")
    print("如果 c 与串行总耗时 T 相当，就不适合采用这种方法。")


if __name__ == "__main__":
    main()
