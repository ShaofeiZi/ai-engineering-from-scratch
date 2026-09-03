"""开放权重 VLM 配方选择器 — 来自 2024-2025 年论文的精简消融表。

将 MM1、Idefics2、Cambrian-1、Molmo、Prismatic VLMs
的关键发现编码为简单数据表。让你可以问：
  - 给定预算和任务组合，哪个配方胜出
  - 如果我替换轴 X，预期增量是多少
  - 先消融哪个轴

不用 numpy，不用 pandas —— 只用字典和打印表格。重点在于证据的结构，
而非数值精度。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Recipe:
    name: str
    encoder: str
    connector: str
    llm_b: int
    data: str
    resolution: str
    mmmu: float
    cv_bench: float
    docvqa: float


RECIPES = [
    Recipe("LLaVA-1.5", "CLIP L/14 @336", "MLP-2", 13, "LLaVA-Inst-150k", "336", 35.3, 56.0, 55.0),
    Recipe("LLaVA-NeXT", "CLIP L/14 @336", "MLP-2", 13, "LLaVA-Inst + shareGPT4V", "AnyRes 672", 36.2, 58.5, 77.4),
    Recipe("Idefics2-8B", "SigLIP SO400m/14", "Perceiver-64", 7, "OBELICS + Cauldron", "980 切片", 43.0, 60.0, 74.0),
    Recipe("MM1-3B", "CLIP L/14", "C-Abstractor", 3, "交错数据 + 图注", "672", 38.6, 59.0, 62.0),
    Recipe("MM1-30B", "CLIP L/14", "C-Abstractor", 30, "交错数据 + 图注", "672", 44.7, 64.0, 74.0),
    Recipe("Molmo-7B-D", "SigLIP SO400m/14", "MLP-2", 7, "PixMo（71.2 万条人工图注）", "AnyRes 672", 45.3, 65.0, 92.4),
    Recipe("Molmo-72B", "SigLIP SO400m/14", "MLP-2", 72, "PixMo（71.2 万条人工图注）", "AnyRes 672", 54.1, 73.0, 93.5),
    Recipe("Cambrian-1-8B", "CLIP + DINOv2 + SigLIP + ConvNeXt", "SVA", 8, "Cambrian-10M", "672", 42.7, 67.8, 77.8),
    Recipe("Prismatic-7B 默认", "SigLIP SO400m/14", "MLP-2", 7, "LLaVA-Inst + shareGPT4V", "336", 40.0, 58.0, 70.0),
]


def axis_impact() -> None:
    print("\n轴影响分解（Prismatic VLM 受控对比）")
    print("-" * 60)
    axes = [
        ("视觉 token 数", 60, "64 -> 576 -> 1024 个 token；超过 1024 后收益递减"),
        ("图像编码器",    20, "CLIP、SigLIP、DINOv2 对比；拼接有帮助"),
        ("连接器架构",     5, "token 数相同时，MLP ~= Q-Former ~= Perceiver"),
        ("数据混合",      10, "详细人工图注优于蒸馏的 GPT-4V 数据"),
        ("LLM 大小",      15, "从 7B 扩至 70B，MMMU 在 55 左右进入平台期"),
        ("分辨率调度",     5, "从 224 渐增至 448 优于固定 448；原生分辨率在 OCR 上胜出"),
    ]
    total_weight = sum(a[1] for a in axes)
    print(f"{'轴':<22}{'方差占比':>8}  注")
    for name, pct, note in axes:
        bar = "#" * (pct // 2)
        print(f"{name:<22}{pct:>6}% {bar}")
        print(f"{'':<22}       {note}")
    print(f"注：权重在四舍五入后从 ~{total_weight}% 重新基准化为 ~100%。")


def compare_encoders() -> None:
    print("\n编码器替换差异（固定 7B LLM，LLaVA-Inst + shareGPT4V 数据）")
    print("-" * 60)
    rows = [
        ("CLIP ViT-L/14 @ 336",        38.5, 56.0, 70.0),
        ("SigLIP SO400m/14 @ 384",     41.0, 60.0, 75.0),
        ("DINOv2 ViT-g/14 @ 224",      37.0, 65.0, 52.0),
        ("SigLIP + DINOv2 拼接",       42.0, 67.0, 74.0),
        ("InternViT-6B @ 448",         43.0, 66.0, 78.0),
    ]
    print(f"{'编码器':<32}{'MMMU':>8}{'CV-B':>8}{'DocVQA':>10}")
    for name, mmmu, cv, doc in rows:
        print(f"{name:<32}{mmmu:>8.1f}{cv:>8.1f}{doc:>10.1f}")
    print("差异：SigLIP 比 CLIP 高 2.5 MMMU；DINOv2 在 CV-Bench 上胜出；"
          "在视觉中心基准上，拼接优于任一单独编码器。")


def compare_data() -> None:
    print("\n数据混合差异（固定 SigLIP + 7B LLM + AnyRes）")
    print("-" * 60)
    rows = [
        ("LLaVA-Inst-150k",         40.0, "网页图注 + GPT-4 对话"),
        ("+ ShareGPT4V",            42.0, "+ GPT-4V 详细图注"),
        ("+ Cauldron",              43.0, "+ OCR + 图表 + 多模态指令"),
        ("PixMo（仅人工图注）",      45.3, "71.2 万条密集人工图注"),
        ("PixMo + Cauldron + 更多",  47.0, "截至 2025 年 7 月的最佳数据混合"),
    ]
    print(f"{'数据混合':<28}{'MMMU':>8}  注释")
    for name, mmmu, note in rows:
        print(f"{name:<28}{mmmu:>8.1f}  {note}")
    print("发现：密集人工标注比蒸馏标注高 +2-3 MMMU")
    print("         在相同训练 token 数下（Molmo 论文）。")


def print_recipes() -> None:
    print("\n典型开放权重 VLM（消融实验报告的 MMMU、CV-Bench、DocVQA）")
    print("-" * 60)
    print(f"{'配方':<22}{'LLM':>6}{'MMMU':>8}{'CV-B':>8}{'DocVQA':>10}")
    for r in RECIPES:
        print(f"{r.name:<22}{r.llm_b:>5}B{r.mmmu:>8.1f}{r.cv_bench:>8.1f}{r.docvqa:>10.1f}")


def pick_recipe(budget_b: int, task: str) -> None:
    print(f"\n选择器：预算 {budget_b}B 参数，任务画像：{task}")
    print("-" * 60)
    weights = {"mmmu": 1.0, "cv": 1.0, "doc": 1.0}
    if task == "ocr":
        weights = {"mmmu": 0.4, "cv": 0.3, "doc": 1.2}
    elif task == "agent":
        weights = {"mmmu": 1.0, "cv": 1.2, "doc": 0.8}
    elif task == "reasoning":
        weights = {"mmmu": 1.5, "cv": 0.5, "doc": 0.8}

    def score(r: Recipe) -> float:
        return r.mmmu * weights["mmmu"] + r.cv_bench * weights["cv"] + r.docvqa * weights["doc"]

    candidates = [r for r in RECIPES if r.llm_b <= budget_b]
    candidates.sort(key=score, reverse=True)
    for r in candidates[:3]:
        print(f"  {r.name:<22} LLM {r.llm_b}B  得分={score(r):.1f}")
        print(f"    编码器={r.encoder}")
        print(f"    数据  ={r.data}")
        print(f"    分辨率={r.resolution}")


def main() -> None:
    print("=" * 60)
    print("开放权重 VLM 配方选择器（阶段 12，课程 07）")
    print("=" * 60)

    print_recipes()
    axis_impact()
    compare_encoders()
    compare_data()

    pick_recipe(10, "ocr")
    pick_recipe(80, "reasoning")
    pick_recipe(10, "agent")


if __name__ == "__main__":
    main()
