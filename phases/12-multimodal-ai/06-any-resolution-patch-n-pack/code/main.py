"""用于可变分辨率视觉 Transformer 批次的 Patch-n'-pack —— 仅标准库。

给定一个 patch 为 P 的 (H, W) 图像尺寸批次，计算：
  - 每张图像的 patch 网格 (H/P, W/P) 和序列长度 n_i = (H/P)(W/P)
  - 打包后的总长度 N = sum(n_i)
  - 分块对角注意力掩码（稠密，N x N）
  - AnyRes 平铺代价 (瓦片 + 缩略图) 用于对比
  - 方形缩放代价（固定序列长度），用于对比

打印一份针对实际工作负载的预算表：收据、图表、截图、照片。
无需 numpy，无需 torch，让每个单元格的字节数计算保持透明。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Image:
    name: str
    h: int
    w: int

    def grid(self, p: int) -> tuple[int, int]:
        return (self.h // p, self.w // p)

    def seq(self, p: int) -> int:
        gh, gw = self.grid(p)
        return gh * gw


@dataclass
class PackResult:
    total_tokens: int
    per_image: list[int]
    mask_nonzero: int
    mask_size: int
    cu_seqlens: list[int] = field(default_factory=list)


def pack_batch(images: list[Image], patch: int) -> PackResult:
    lens = [img.seq(patch) for img in images]
    total = sum(lens)
    nz = sum(n * n for n in lens)
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    return PackResult(total, lens, nz, total * total, offsets)


def build_dense_mask(pack: PackResult) -> list[list[int]]:
    n = pack.total_tokens
    mask = [[0] * n for _ in range(n)]
    for b in range(len(pack.cu_seqlens) - 1):
        lo = pack.cu_seqlens[b]
        hi = pack.cu_seqlens[b + 1]
        for i in range(lo, hi):
            for j in range(lo, hi):
                mask[i][j] = 1
    return mask


def anyres_cost(img: Image, tile: int = 336, thumb: int = 336) -> dict:
    tile_grid = tile // 14
    thumb_grid = thumb // 14
    if img.h <= tile and img.w <= tile:
        grid_r, grid_c = 1, 1
    else:
        best = None
        for gr in range(1, 4):
            for gc in range(1, 4):
                if gr * gc > 6:
                    continue
                tile_h, tile_w = gr * tile, gc * tile
                ratio = img.h / img.w
                tile_ratio = tile_h / tile_w
                score = abs(ratio - tile_ratio) + 0.1 * (gr + gc)
                if best is None or score < best[0]:
                    best = (score, gr, gc)
        _, grid_r, grid_c = best
    tile_tokens = grid_r * grid_c * tile_grid * tile_grid
    thumb_tokens = thumb_grid * thumb_grid
    return {
        "grid": (grid_r, grid_c),
        "tile_tokens": tile_tokens,
        "thumb_tokens": thumb_tokens,
        "total": tile_tokens + thumb_tokens,
    }


def square_cost(img: Image, side: int = 336, patch: int = 14) -> int:
    g = side // patch
    return g * g


def fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def demo_toy_pack() -> None:
    print("\n示例批次：两张图像，patch 为 2")
    print("-" * 60)
    imgs = [Image("A", 6, 4), Image("B", 4, 8)]
    for img in imgs:
        gh, gw = img.grid(2)
        print(f"  {img.name}: {img.h}x{img.w} -> 网格 {gh}x{gw} = {img.seq(2)} 个 token")
    pack = pack_batch(imgs, 2)
    print(f"打包后的总长度：{pack.total_tokens}")
    print(f"cu_seqlens (FlashAttn varlen): {pack.cu_seqlens}")
    print(f"稠密掩码大小：{pack.mask_size} 个单元格，"
          f"非零项：{pack.mask_nonzero} "
          f"({pack.mask_nonzero * 100 / pack.mask_size:.1f}%)")
    mask = build_dense_mask(pack)
    print("\n分块对角掩码（1=关注，.=遮蔽）：")
    for row in mask:
        print("  " + "".join("1" if v else "." for v in row))


def budget_table(workload: list[Image]) -> None:
    print("\n" + "=" * 72)
    print(f"{'图像':<26}{'原生':>10}{'方形':>10}{'AnyRes':>14}{'网格':>10}")
    print("-" * 72)
    native_sum = 0
    square_sum = 0
    anyres_sum = 0
    for img in workload:
        nat = img.seq(14)
        sq = square_cost(img, 336, 14)
        ar = anyres_cost(img)
        native_sum += nat
        square_sum += sq
        anyres_sum += ar["total"]
        gr, gc = ar["grid"]
        print(f"{img.name:<26}{nat:>10}{sq:>10}{ar['total']:>14}   {gr}x{gc}")
    print("-" * 72)
    print(f"{'合计':<26}{native_sum:>10}{square_sum:>10}{anyres_sum:>14}")
    print(f"\n原生与方形之比：{native_sum / square_sum:>6.2f} 倍 token，"
          f"保留 OCR 与版面细节")
    print(f"原生与 AnyRes 之比：{native_sum / anyres_sum:>6.2f} 倍 token，"
          f"超过约 2 个瓦片后不会产生瓦片与缩略图膨胀")
    print(f"AnyRes 与方形之比：{anyres_sum / square_sum:>6.2f} 倍 token，"
          f"适合编码器固定在 336 分辨率时的折中方案")


def main() -> None:
    print("=" * 60)
    print("适用于任意分辨率 VLM 的 PATCH-N-PACK（第 12 阶段，第 06 课）")
    print("=" * 60)

    demo_toy_pack()

    workload = [
        Image("收据 600x1500（1:2.5）", 600, 1500),
        Image("图表 1280x720（16:9）", 1280, 720),
        Image("手机屏幕 1170x2532", 1170, 2532),
        Image("照片 2048x1536（4:3）", 2048, 1536),
        Image("收据 504x1260（1:2.5）", 504, 1260),
    ]
    for img in workload:
        img.h -= img.h % 14
        img.w -= img.w % 14

    budget_table(workload)

    print("\n" + "=" * 60)
    print("各策略的适用场景")
    print("-" * 60)
    print("  native-pack (NaViT / NaFlex / M-RoPE):")
    print("    多宽高比批次，最高保真度，最少 token")
    print("  AnyRes (LLaVA-NeXT):")
    print("    编码器在 336x336 上冻结，但你需要细节")
    print("  方形缩放：")
    print("    快速基线，仅照片工作负载，无 OCR")


if __name__ == "__main__":
    main()
