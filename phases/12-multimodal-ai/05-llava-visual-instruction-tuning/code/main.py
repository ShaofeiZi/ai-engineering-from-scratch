"""LLaVA 两层 MLP 投影器 + 提示构建器 — 纯标准库 Python。

演示 LLaVA 的前向传播过程：
  - 玩具 ViT 输出 16 个维度为 16 的 patch token
  - 两层 MLP 将每个 patch 投影到维度 24（即 'LLM' 维度）
  - 构建 LLaVA 格式提示，将 <image> 占位符替换为 16 个投影后的 token
  - 分别在 2k / 32k / 128k LLM 窗口下报告上下文预算

不依赖 numpy，不依赖 torch。线性层和 GELU 均为手动实现。
"""

from __future__ import annotations

import math
import random

rng = random.Random(11)

PATCH_COUNT = 16
PATCH_DIM = 16
HIDDEN_DIM = 32
LLM_DIM = 24


def vec(n: int) -> list[float]:
    return [rng.gauss(0, 0.3) for _ in range(n)]


def mat(rows: int, cols: int) -> list[list[float]]:
    return [vec(cols) for _ in range(rows)]


def linear(W: list[list[float]], b: list[float], x: list[float]) -> list[float]:
    return [sum(r * v for r, v in zip(row, x)) + bi
            for row, bi in zip(W, b)]


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def gelu_vec(v: list[float]) -> list[float]:
    return [gelu(x) for x in v]


class MLPProjector:
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        self.W1 = mat(hidden, in_dim)
        self.b1 = [0.0] * hidden
        self.W2 = mat(out_dim, hidden)
        self.b2 = [0.0] * out_dim

    def forward(self, x: list[float]) -> list[float]:
        h = gelu_vec(linear(self.W1, self.b1, x))
        return linear(self.W2, self.b2, h)

    def num_params(self) -> int:
        return (len(self.W1) * len(self.W1[0]) + len(self.b1)
                + len(self.W2) * len(self.W2[0]) + len(self.b2))


def fake_vit_output() -> list[list[float]]:
    return [vec(PATCH_DIM) for _ in range(PATCH_COUNT)]


def build_llava_prompt(system: str, user: str, image_tokens: int) -> dict:
    placeholder = "<image>"
    template = (
        f"SYSTEM: {system}\n"
        f"USER: {placeholder} {user}\n"
        f"ASSISTANT: "
    )
    return {
        "raw_prompt": template,
        "placeholder": placeholder,
        "image_tokens": image_tokens,
        "text_token_estimate": len(template.split()) + 10,
    }


def visualize_context(num_image_tokens: int, text_tokens: int) -> None:
    print("\n不同 LLM 窗口下的上下文预算")
    print("-" * 60)
    totals = (2048, 8192, 32768, 131072)
    for t in totals:
        used = num_image_tokens + text_tokens
        remain = t - used
        pct_image = 100 * num_image_tokens / t
        print(f"  窗口 {t:>6d}：图像 {pct_image:5.1f}% | "
              f"文本 {100*text_tokens/t:4.1f}% | 剩余 {max(remain, 0):>6d} 个 token")


def demo_projector() -> None:
    print("演示 1：两层 MLP 投影器前向传播")
    print("-" * 60)
    patches = fake_vit_output()
    proj = MLPProjector(PATCH_DIM, HIDDEN_DIM, LLM_DIM)

    print(f"  ViT 输出：{PATCH_COUNT} 个 patch，维度为 {PATCH_DIM}")
    print(f"  MLP：     {PATCH_DIM} -> {HIDDEN_DIM} -> {LLM_DIM}")
    print(f"  参数量：  {proj.num_params():,}")

    visual_tokens = [proj.forward(p) for p in patches]
    print(f"  输出：    {len(visual_tokens)} 个维度为 "
          f"{len(visual_tokens[0])} 的视觉 token")
    print(f"  token 0 示例：{[round(x, 3) for x in visual_tokens[0][:6]]}")


def demo_prompt() -> None:
    print("\n演示 2：LLaVA 提示模板")
    print("-" * 60)
    system = "一位好奇的人类与一名人工智能助手之间的对话。"
    user = "请详细描述你在这张图像中看到的内容。"
    prompt = build_llava_prompt(system, user, image_tokens=576)

    print("  原始提示（LLM 在 image-token 替换后接收到的内容）：")
    print("  " + "-" * 56)
    for line in prompt["raw_prompt"].split("\n"):
        print(f"    {line}")
    print(f"  <image> 占位符 -> 替换为 {prompt['image_tokens']} 个视觉 token")
    print(f"  文本 token 估算：~{prompt['text_token_estimate']} 个 token")
    visualize_context(prompt["image_tokens"], prompt["text_token_estimate"])


def demo_anyres() -> None:
    print("\n演示 3：LLaVA-NeXT AnyRes token 开销")
    print("-" * 60)
    tile_tokens = 576
    configs = [
        ("336x336（基础）", 1, 0),
        ("672x336 (1x2)", 2, 1),
        ("672x672 (2x2)", 4, 1),
        ("1344x672 (2x4)", 8, 1),
        ("1344x1344 (4x4)", 16, 1),
    ]
    for name, tiles, thumb in configs:
        total = tiles * tile_tokens + thumb * tile_tokens
        print(f"  {name:20s}：{tiles:2d} 个瓦片 + {thumb} 张缩略图 "
              f"= {total:5d} 个 token")


def main() -> None:
    print("=" * 60)
    print("LLAVA 视觉指令微调（第 12 阶段，第 05 课）")
    print("=" * 60)
    demo_projector()
    demo_prompt()
    demo_anyres()
    print("\n" + "=" * 60)
    print("要点")
    print("-" * 60)
    print("  · 两层 MLP 投影器有 2200 万参数，相比 7B LLM 很小")
    print("  · 将 <image> 占位符替换为 N 个投影后的视觉 token")
    print("  · 基础 LLaVA 每张图像使用 576 个 token，占 2k 上下文的 30%")
    print("  · AnyRes 对高分辨率 OCR / 图表输入最多使用 2880 个 token")
    print("  · 第 1 阶段仅训练投影器，耗时数小时")
    print("  · 第 2 阶段在 15.8 万条 GPT-4 指令上训练投影器与 LLM")


if __name__ == "__main__":
    main()
