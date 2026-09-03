"""视觉 Transformer 补丁分词器与几何计算器 — 纯标准库 Python。

给定 ViT 配置（补丁大小、分辨率、隐藏维度、深度、头数），计算：
  - 补丁分词后的网格形状与序列长度
  - 分组件参数量（补丁嵌入、位置、各块、LN）
  - 每次前向传播的 FLOPs（主要由注意力 + MLP 主导）
  - 2026 年主流编码器对比表

同时将一张 8x8 灰度玩具图像走一遍补丁化、展平、投影流程，
让这一原语变得具体可见。不使用 numpy，不使用 torch — 只用整数和列表。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViTConfig:
    name: str
    image_size: int
    patch_size: int
    hidden: int
    depth: int
    heads: int
    registers: int = 0
    cls_token: bool = True


ZOO = [
    ViTConfig("ViT-B/16 @ 224", 224, 16, 768, 12, 12),
    ViTConfig("ViT-L/14 @ 336 (CLIP)", 336, 14, 1024, 24, 16),
    ViTConfig("DINOv2 ViT-g/14 @ 224", 224, 14, 1536, 40, 24, registers=4),
    ViTConfig("SigLIP SO400m/14 @ 378", 378, 14, 1152, 27, 16, registers=4,
              cls_token=False),
    ViTConfig("Qwen2.5-VL ViT @ 896x896", 896, 14, 1280, 32, 16),
]


def grid_shape(image_size: int, patch_size: int) -> tuple[int, int]:
    if image_size <= 0 or patch_size <= 0:
        raise ValueError(f"image_size and patch_size must be positive, got {image_size=} {patch_size=}")
    if image_size % patch_size != 0:
        raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
    g = image_size // patch_size
    return (g, g)


def seq_length(cfg: ViTConfig) -> int:
    h, w = grid_shape(cfg.image_size, cfg.patch_size)
    extra = (1 if cfg.cls_token else 0) + cfg.registers
    return h * w + extra


def patch_embed_params(cfg: ViTConfig) -> int:
    p = cfg.patch_size
    return 3 * p * p * cfg.hidden + cfg.hidden


def pos_embed_params(cfg: ViTConfig) -> int:
    return seq_length(cfg) * cfg.hidden


def cls_register_params(cfg: ViTConfig) -> int:
    n = (1 if cfg.cls_token else 0) + cfg.registers
    return n * cfg.hidden


def block_params(cfg: ViTConfig) -> int:
    d = cfg.hidden
    qkvo = 4 * d * d + 4 * d
    mlp = 2 * d * 4 * d + d + 4 * d
    ln = 2 * 2 * d
    return qkvo + mlp + ln


def total_params(cfg: ViTConfig) -> dict:
    pe = patch_embed_params(cfg)
    po = pos_embed_params(cfg)
    cr = cls_register_params(cfg)
    bl = block_params(cfg) * cfg.depth
    fl = 2 * cfg.hidden
    total = pe + po + cr + bl + fl
    return {"patch_embed": pe, "position": po, "cls+reg": cr,
            "blocks": bl, "final_ln": fl, "total": total}


def flops_per_forward(cfg: ViTConfig) -> int:
    n = seq_length(cfg)
    d = cfg.hidden
    attn = 4 * n * d * d + 2 * n * n * d
    mlp = 2 * n * d * 4 * d * 2
    return cfg.depth * (attn + mlp)


def fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def patch_toy_image() -> None:
    """以 P=4 对一张 8x8 灰度图像执行补丁分词。
    网格为 2x2，共 4 个 token；每个补丁展平后含 4x4=16 个像素。"""
    print("\n玩具图像补丁分词（8x8 灰度，patch_size=4）")
    print("-" * 60)
    img = [[(r * 8 + c) % 256 for c in range(8)] for r in range(8)]
    print("像素网格（第 0..7 行）：")
    for row in img:
        print("  " + " ".join(f"{v:3d}" for v in row))

    P = 4
    patches = []
    for pr in range(0, 8, P):
        for pc in range(0, 8, P):
            patch = []
            for dr in range(P):
                for dc in range(P):
                    patch.append(img[pr + dr][pc + dc])
            patches.append(patch)

    print(f"\n补丁（共 {len(patches)} 个，每个长度 {P*P}）：")
    for i, p in enumerate(patches):
        print(f"  补丁 {i}: {p}")

    fake_W = [[((i + j) % 5) - 2 for j in range(P * P)] for i in range(4)]
    embeddings = []
    for patch in patches:
        emb = []
        for row in fake_W:
            s = sum(r * v for r, v in zip(row, patch, strict=True))
            emb.append(s)
        embeddings.append(emb)

    print("\n线性投影（P*P=16 -> 隐藏维度=4）：")
    for i, emb in enumerate(embeddings):
        print(f"  token {i}: {emb}")
    print("→ 4 个维度为 4 的 token，已可输入 Transformer。")


def print_config(cfg: ViTConfig) -> None:
    params = total_params(cfg)
    seq = seq_length(cfg)
    gh, gw = grid_shape(cfg.image_size, cfg.patch_size)
    fl = flops_per_forward(cfg)
    print(f"\n{cfg.name}")
    print("-" * 60)
    print(f"  图像            : {cfg.image_size}x{cfg.image_size}")
    print(f"  补丁大小       : {cfg.patch_size}")
    print(f"  网格             : {gh}x{gw}")
    print(f"  序列长度         : {seq}（{'包含 CLS' if cfg.cls_token else '不含 CLS'}，"
          f"{cfg.registers} 个 register token）")
    print(f"  隐藏维度 / 深度   : {cfg.hidden} / {cfg.depth}")
    print(f"  补丁嵌入      : {fmt(params['patch_embed'])}")
    print(f"  位置嵌入   : {fmt(params['position'])}")
    print(f"  各块合计     : {fmt(params['blocks'])}")
    print(f"  ** 总参数量 **: {fmt(params['total'])}")
    print(f"  FLOPs / 前向传播 : {fmt(fl)}")


def main() -> None:
    print("=" * 60)
    print("VIT 补丁 TOKEN 几何计算器（第 12 阶段，第 01 课）")
    print("=" * 60)

    patch_toy_image()

    for cfg in ZOO:
        print_config(cfg)

    print("\n" + "=" * 60)
    print("关键比率")
    print("-" * 60)
    vit_b = ZOO[0]
    qwen = ZOO[-1]
    print(f"  ViT-B/16 @ 224    序列长度: {seq_length(vit_b)}")
    print(f"  Qwen2.5-VL @ 896  序列长度: {seq_length(qwen)}")
    print(f"  比率: {seq_length(qwen) / seq_length(vit_b):.1f}x 更多的 token")
    print("  这就是高分辨率 VLM 需要 token 合并或池化的原因。")


if __name__ == "__main__":
    main()
