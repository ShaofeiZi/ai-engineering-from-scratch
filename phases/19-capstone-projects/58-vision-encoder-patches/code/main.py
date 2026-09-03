"""视觉编码器前端：patch 嵌入加二维正弦位置编码。

将一张 224x224x3 的图像切分为 196 个 patch token 序列，并加上一个 CLS
token。patch 投影使用一个 kernel 和 stride 都等于 patch size 的 Conv2d，
在数值上与“先展平再线性变换”完全等价。位置信号是一张固定的二维正弦
表：一半的嵌入维度用于编码行位置，另一半编码列位置，并以多种频率采样。

运行方式：python3 main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class FrontEndConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden: int = 768

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"patch_size {self.patch_size} 必须能整除 image_size {self.image_size}"
            )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size


def sinusoidal_2d(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """构造一张确定性的二维正弦位置表，形状为 (grid_h * grid_w, dim)。

    dim 的一半用于编码行位置，另一半用于编码列位置。在每一半内部，频率
    覆盖标准 Transformer 的 sin/cos 频段。相同的输入永远产生相同的输出，
    不含任何可学习状态。
    """
    if dim % 4 != 0:
        raise ValueError(f"sinusoidal_2d 的 dim 必须能被 4 整除，当前为 {dim}")
    half = dim // 2
    quarter = half // 2

    freq = torch.arange(quarter, dtype=torch.float32)
    inv = torch.exp(-math.log(10000.0) * freq / max(1, quarter))

    rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1) * inv.unsqueeze(0)
    cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(1) * inv.unsqueeze(0)

    row_emb = torch.cat([torch.sin(rows), torch.cos(rows)], dim=1)
    col_emb = torch.cat([torch.sin(cols), torch.cos(cols)], dim=1)

    table = torch.zeros(grid_h, grid_w, dim)
    table[:, :, :half] = row_emb.unsqueeze(1).expand(-1, grid_w, -1)
    table[:, :, half:] = col_emb.unsqueeze(0).expand(grid_h, -1, -1)
    return table.reshape(grid_h * grid_w, dim)


class PatchEmbed(nn.Module):
    """以带 stride 的 Conv2d 实现的 patch 投影。

    对 (B, C, H, W) 输入，输出形状为 (B, N, hidden)，其中
    N = (H / patch_size) * (W / patch_size)。
    """

    def __init__(self, cfg: FrontEndConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            cfg.in_channels,
            cfg.hidden,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"期望 4 维输入 (B,C,H,W)，实际形状为 {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"通道数不匹配：实际为 {x.shape[1]}，期望为 {self.cfg.in_channels}"
            )
        if x.shape[2] != self.cfg.image_size or x.shape[3] != self.cfg.image_size:
            raise ValueError(
                f"空间尺寸不匹配：实际为 {tuple(x.shape[2:])}，期望为 "
                f"({self.cfg.image_size}, {self.cfg.image_size})"
            )
        out = self.proj(x)
        b = out.shape[0]
        out = out.flatten(2).transpose(1, 2)
        return out


class VisionFrontEnd(nn.Module):
    """patch 嵌入 + 前置 CLS + 二维正弦位置编码。

    输出形状：(B, num_patches + 1, hidden)。
    """

    def __init__(self, cfg: FrontEndConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        pos = sinusoidal_2d(cfg.grid_size, cfg.grid_size, cfg.hidden)
        cls_pos = torch.zeros(1, cfg.hidden)
        full = torch.cat([cls_pos, pos], dim=0).unsqueeze(0)
        self.register_buffer("pos_embed", full, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        return tokens


def synthesize_image(seed: int, image_size: int = 224, channels: int = 3) -> torch.Tensor:
    """用 numpy.random 构造一份确定性的 1x3x224x224 测试图像。

    取值为 [0, 1] 区间的 float32。在噪声之上叠加一个平滑梯度，可以让
    patch 投影同时处理到高频和低频内容，从而有更丰富的特征可供汇总。
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((channels, image_size, image_size)).astype("float32") * 0.1
    y_coords = np.linspace(0.0, 1.0, image_size, dtype="float32")
    x_coords = np.linspace(0.0, 1.0, image_size, dtype="float32")
    gx, gy = np.meshgrid(x_coords, y_coords, indexing="xy")
    gradient = np.stack([gx, gy, (gx + gy) * 0.5], axis=0).astype("float32")
    img = np.clip(gradient + noise + 0.5, 0.0, 1.0)
    return torch.from_numpy(img).unsqueeze(0)


def unfold_then_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, patch_size: int) -> torch.Tensor:
    """通过 unfold + matmul 实现的 patch 投影参考实现。

    供测试使用，用于断言 Conv2d 投影与“先展平再线性变换”的数学结果一致。
    """
    if x.dim() != 4:
        raise ValueError(f"期望 4 维输入，实际形状为 {tuple(x.shape)}")
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    b, c, gh, gw, ph, pw = patches.shape
    flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, gh * gw, c * ph * pw)
    w_flat = weight.reshape(weight.shape[0], -1)
    return flat @ w_flat.T + bias


def describe_token_norms(tokens: torch.Tensor, max_show: int = 8) -> str:
    """输出前几个 token 的 L2 范数，用于基本的合理性检查。"""
    norms = tokens.detach().norm(dim=-1)[0].tolist()
    head = norms[:max_show]
    return ", ".join(f"{v:.3f}" for v in head)


def main() -> None:
    print("=" * 60)
    print("视觉编码器 PATCH 前端")
    print("=" * 60)

    cfg = FrontEndConfig()
    print(f"  图像尺寸   : {cfg.image_size}")
    print(f"  patch 尺寸 : {cfg.patch_size}")
    print(f"  网格尺寸   : {cfg.grid_size}x{cfg.grid_size}")
    print(f"  patch 数量 : {cfg.num_patches}")
    print(f"  隐藏维度   : {cfg.hidden}")
    print(f"  序列长度   : {cfg.num_patches + 1} (含 CLS)")

    torch.manual_seed(0)
    img = synthesize_image(seed=0)
    print(f"\n测试图像形状 : {tuple(img.shape)}")
    print(f"测试图像类型 : {img.dtype}")
    print(f"测试像素范围 : [{img.min().item():.3f}, {img.max().item():.3f}]")

    model = VisionFrontEnd(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n前端参数量    : {n_params:,}")

    with torch.no_grad():
        tokens = model(img)

    print(f"输出 token 形状 : {tuple(tokens.shape)}")
    print(f"CLS token 范数  : {tokens[0, 0].norm().item():.3f}")
    print(f"前 8 个 token 范数 : {describe_token_norms(tokens)}")

    print("\n位置编码行签名:")
    pos_row = model.pos_embed[0, 1, :8].tolist()
    print("  pos[1, :8] =", ", ".join(f"{v:+.3f}" for v in pos_row))

    print("\n批次一致性检查:")
    img_b4 = synthesize_image(seed=1).repeat(4, 1, 1, 1)
    with torch.no_grad():
        out_b4 = model(img_b4)
    print(f"  batch=4 输出形状: {tuple(out_b4.shape)}")
    drift = (out_b4 - out_b4[0:1]).abs().max().item()
    print(f"  相同批次各行间最大漂移: {drift:.6f}")

    print("\nunfold 参考实现 vs Conv2d 投影:")
    weight = model.patch.proj.weight.detach()
    bias = model.patch.proj.bias.detach()
    ref = unfold_then_linear(img, weight, bias, cfg.patch_size)
    conv = model.patch(img)
    diff = (ref - conv).abs().max().item()
    print(f"  最大绝对误差 : {diff:.6e}")
    if diff < 1e-4:
        print("  通过: unfold 参考实现与 Conv2d 在浮点精度内一致")
    else:
        print("  失败: 投影结果与参考实现存在漂移")

    print("\n完成。")


if __name__ == "__main__":
    main()
