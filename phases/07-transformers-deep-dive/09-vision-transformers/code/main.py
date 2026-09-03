"""视觉 Transformer（ViT）——图像分块 + 嵌入前端。

仅使用标准库。将一幅 24x24x3 的玩具图像切成 6x6 图像块，
把每块投影为 d_model 向量，在开头添加 [CLS]，并加入二维位置编码。
验证张量形状，并计算真实 ViT 配置的参数量。
"""

import math
import random


def make_image(H, W, C=3, seed=0):
    rng = random.Random(seed)
    return [[[rng.randint(0, 255) / 255.0 for _ in range(C)] for _ in range(W)] for _ in range(H)]


def patchify(image, patch_size):
    H = len(image)
    W = len(image[0])
    C = len(image[0][0])
    assert H % patch_size == 0 and W % patch_size == 0
    patches = []
    grid = []
    for row_idx, i in enumerate(range(0, H, patch_size)):
        grid_row = []
        for col_idx, j in enumerate(range(0, W, patch_size)):
            patch = []
            for di in range(patch_size):
                for dj in range(patch_size):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
            grid_row.append((row_idx, col_idx))
        grid.append(grid_row)
    return patches, (H // patch_size, W // patch_size)


def linear_project(patches, d_model, rng=None):
    if rng is None:
        rng = random.Random(0)
    in_dim = len(patches[0])
    scale = math.sqrt(2.0 / (in_dim + d_model))
    W = [[rng.gauss(0, scale) for _ in range(d_model)] for _ in range(in_dim)]
    out = []
    for patch in patches:
        row = [0.0] * d_model
        for i, x in enumerate(patch):
            if x == 0.0:
                continue
            for j in range(d_model):
                row[j] += x * W[i][j]
        out.append(row)
    return out, W


def cls_and_pos(tokens, grid_h, grid_w, rng=None):
    """前置可学习的 [CLS]，并添加二维正弦位置编码。"""
    if rng is None:
        rng = random.Random(1)
    d_model = len(tokens[0])
    cls = [rng.gauss(0, 0.02) for _ in range(d_model)]
    pe = pos_2d(grid_h, grid_w, d_model)
    out = [list(cls)]
    idx = 0
    for i in range(grid_h):
        for j in range(grid_w):
            t = [tokens[idx][k] + pe[i][j][k] for k in range(d_model)]
            out.append(t)
            idx += 1
    return out


def pos_2d(H, W, d_model):
    """二维正弦编码：将 d_model 对半拆分，独立编码行和列。"""
    assert d_model % 4 == 0, "使用二维正弦编码时，d_model 必须能被 4 整除"
    half = d_model // 2
    pe = [[[0.0] * d_model for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            for k in range(half // 2):
                theta_row = i / (10000 ** (2 * k / half))
                pe[i][j][2 * k] = math.sin(theta_row)
                pe[i][j][2 * k + 1] = math.cos(theta_row)
            for k in range(half // 2):
                theta_col = j / (10000 ** (2 * k / half))
                pe[i][j][half + 2 * k] = math.sin(theta_col)
                pe[i][j][half + 2 * k + 1] = math.cos(theta_col)
    return pe


def param_count_vit(d_model, n_layers, n_heads, ffn_expansion, num_patches, num_classes):
    """估算 ViT 参数量（图像块嵌入 + transformer + 预测头）。"""
    # 图像块嵌入：(patch_flat_size, d_model)——此处忽略 patch_size，由调用方缩放。
    # 每层自注意力：4 * d_model^2（Q、K、V、O）
    # 每层 FFN：2 * d_model * (ffn_expansion * d_model)
    # 归一化：每层 2 * d_model（LayerNorm gamma+beta）
    per_layer = 4 * d_model ** 2 + 2 * d_model * int(ffn_expansion * d_model) + 4 * d_model
    # 位置嵌入：(num_patches + 1) * d_model
    pos_emb = (num_patches + 1) * d_model
    # CLS token：d_model
    # 分类头：d_model * num_classes
    head = d_model * num_classes
    # 最终层归一化：2 * d_model
    return per_layer * n_layers + pos_emb + d_model + head + 2 * d_model


def main():
    H, W, C = 24, 24, 3
    patch_size = 6
    d_model = 48

    image = make_image(H, W, C, seed=0)
    patches, grid = patchify(image, patch_size)
    tokens, W_proj = linear_project(patches, d_model, rng=random.Random(42))
    tokens_with_pos = cls_and_pos(tokens, grid[0], grid[1], rng=random.Random(7))

    print("=== ViT 前端健全性检查 ===")
    print(f"图像：              ({H}, {W}, {C})")
    print(f"图像块大小：        {patch_size}x{patch_size}")
    print(f"网格：              {grid[0]} x {grid[1]} = {grid[0] * grid[1]} 个图像块")
    print(f"扁平图像块大小：    {patch_size * patch_size * C}")
    print(f"d_model:             {d_model}")
    print(f"序列长度：          {len(tokens_with_pos)}  （图像块 + CLS）")
    print(f"CLS 的单元 [0,0]：  {tokens_with_pos[0][0]:.4f}")
    print(f"p1 的单元 [0,0]：   {tokens_with_pos[1][0]:.4f}")
    print()

    print("=== 参数量（估算）===")
    for name, d, L, H_heads, exp, patch in [
        ("ViT-Tiny/16",   192, 12, 3, 4, 16),
        ("ViT-Small/16",  384, 12, 6, 4, 16),
        ("ViT-Base/16",   768, 12, 12, 4, 16),
        ("ViT-Large/16", 1024, 24, 16, 4, 16),
        ("ViT-Huge/14",  1280, 32, 16, 4, 14),
    ]:
        grid_n = (224 // patch) ** 2
        params = param_count_vit(d, L, H_heads, exp, grid_n, num_classes=1000)
        # 加上图像块嵌入：(P*P*3) * d_model
        params += patch * patch * 3 * d
        print(f"  {name:<14}  d={d:<5}  L={L:<3}  头数={H_heads:<3}  图像块数={grid_n:<4}  约 {params / 1e6:.1f}M 参数")

    print()
    print("要点：ViT 原样复用 BERT 编码器；所有视觉能力都位于")
    print("图像分块 + 位置方案 + [CLS] 池化中。")


if __name__ == "__main__":
    main()
