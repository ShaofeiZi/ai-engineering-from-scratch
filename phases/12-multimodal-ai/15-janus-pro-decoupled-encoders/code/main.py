"""Janus-Pro 解耦编码器路由——标准库实现。

两个模拟编码器（语义 SigLIP-like、重建 VQ-like），一个共享的
Transformer 主体，一个根据任务标签进行选择的路由器。追踪三个示例
提示词在管道中的处理过程。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

random.seed(3)


@dataclass
class SiglipStub:
    dim: int = 32

    def encode(self, image_seed: int) -> list[float]:
        random.seed(image_seed)
        return [random.gauss(0, 0.5) for _ in range(self.dim)]


@dataclass
class VQStub:
    vocab: int = 256
    n_tokens: int = 16

    def encode(self, image_seed: int) -> list[int]:
        random.seed(image_seed * 7 + 1)
        return [random.randint(0, self.vocab - 1) for _ in range(self.n_tokens)]

    def decode(self, tokens: list[int]) -> str:
        return f"由 token {tokens[:4]}... 经 VQ 解码得到的图像"


@dataclass
class SharedBody:
    name: str = "DeepSeek-7B-init"

    def process(self, input_stream: list, kind: str) -> list:
        if kind == "text_out":
            return [f"word_{i}" for i in range(4)]
        if kind == "image_out":
            return [random.randint(0, 255) for _ in range(16)]
        return []


def route(prompt: str) -> str:
    """将任务分类为 `understand` 或 `generate`。"""
    u_keywords = ["describe", "what", "why", "caption", "explain", "how many",
                  "描述", "什么", "为何", "说明", "多少", "姿势"]
    g_keywords = ["draw", "generate", "sketch", "render", "create", "paint",
                  "画", "生成", "绘制", "渲染", "创作"]
    p = prompt.lower()
    u_score = sum(1 for k in u_keywords if k in p)
    g_score = sum(1 for k in g_keywords if k in p)
    if g_score > u_score:
        return "generate"
    if u_score > g_score:
        return "understand"
    return "ambiguous"


def run_pipeline(prompt: str, image_seed: int = 42) -> dict:
    siglip = SiglipStub()
    vq = VQStub()
    body = SharedBody()

    task = route(prompt)
    trace = {"prompt": prompt, "task": task}

    if task == "understand":
        feats = siglip.encode(image_seed)
        trace["route"] = "SigLIP -> 共享主体 -> 文本"
        trace["input_len"] = len(feats)
        out = body.process(feats, kind="text_out")
        trace["output"] = out
    elif task == "generate":
        tokens = vq.encode(image_seed) if image_seed else []
        trace["route"] = "（可选 VQ）-> 共享主体 -> 图像 VQ -> 解码器"
        out_tokens = body.process(tokens, kind="image_out")
        trace["output"] = vq.decode(out_tokens)
    else:
        trace["route"] = "有歧义：同时运行两条路径并合并"
        feats = siglip.encode(image_seed)
        tokens = vq.encode(image_seed)
        trace["input_len"] = f"SigLIP:{len(feats)} + VQ:{len(tokens)}"
        trace["output"] = (body.process(feats, "text_out"),
                           vq.decode(body.process(tokens, "image_out")))

    return trace


def demo_routing() -> None:
    prompts = [
        "描述这张图像中的内容",
        "生成一幅海上日落的图片",
        "画一只猫，然后描述它的品种",
        "图像中人物的姿势是什么？",
        "渲染一幅夜间赛博朋克城市景观",
    ]
    for p in prompts:
        trace = run_pipeline(p, image_seed=hash(p) % 1000)
        task_name = {"understand": "理解", "generate": "生成", "ambiguous": "有歧义"}[trace["task"]]
        print(f"\n  提示词  : {p}")
        print(f"  任务    : {task_name}")
        print(f"  路由    : {trace['route']}")
        print(f"  输出    : {trace['output']}")


def data_scale_table() -> None:
    print("\n数据规模：Janus 与 Janus-Pro 对比")
    print("-" * 60)
    rows = [
        ("阶段 1（对齐）",         "72M 对",     "90M 对",     "+25%"),
        ("阶段 2（统一）",         "26M 对",     "72M 对",     "+176%"),
        ("阶段 3（指令）",         "1.2M 条",    "1.4M 条",    "+17%"),
        ("模型参数",               "1.3B",       "7B",         "5.4x"),
        ("MMMU",                  "30.5",       "60.3",       "+29.8"),
        ("GenEval",               "0.61",       "0.80",       "+0.19"),
    ]
    print(f"  {'轴':<20}{'Janus':<14}{'Janus-Pro':<14}{'差值'}")
    for r in rows:
        print(f"  {r[0]:<20}{r[1]:<14}{r[2]:<14}{r[3]}")


def main() -> None:
    print("=" * 60)
    print("JANUS-PRO 解耦编码器（第12阶段，第15课）")
    print("=" * 60)

    print("\n路由轨迹：5 个提示词通过双编码器管道")
    print("-" * 60)
    demo_routing()

    data_scale_table()

    print("\n架构一句话概括")
    print("-" * 60)
    print("  输入塔 A (SigLIP)  -> ")
    print("  输入塔 B (VQ)       -> 共享 transformer 主体 ->")
    print("  输出头 1（文本 NTP）或 输出头 2（VQ token）")
    print("  3个阶段：对齐 -> 统一 -> 指令微调")


if __name__ == "__main__":
    main()
