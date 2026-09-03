"""基于第 58 课 patch 前端构建的 Vision Transformer 编码器。

十二个 Pre-LN 模块、十二个注意力头、4 倍扩展的 GELU 前馈网络。编码器
接收一张 224x224x3 的测试图像，返回上下文化的 token 序列，并暴露
CLS 池化向量供下游任务头使用。

运行方式：python3 main.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
LESSON_58 = THIS_DIR.parent.parent / "58-vision-encoder-patches" / "code"


def _load_front_end_module():
    import importlib.util

    name = "vision_front_end_lesson58"
    if name in sys.modules:
        return sys.modules[name]
    src = LESSON_58 / "main.py"
    spec = importlib.util.spec_from_file_location(name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载位于 {src} 的第 58 课 main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_front_module = _load_front_end_module()
FrontEndConfig = _front_module.FrontEndConfig
VisionFrontEnd = _front_module.VisionFrontEnd
synthesize_image = _front_module.synthesize_image


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.hidden % self.heads != 0:
            raise ValueError(f"hidden {self.hidden} 不能被 heads {self.heads} 整除")
        return self.hidden // self.heads

    def front_end_config(self) -> FrontEndConfig:
        return FrontEndConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden=self.hidden,
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.hidden, cfg.hidden * 3, bias=True)
        self.out = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"期望形状为 (B, N, D)，实际得到 {tuple(x.shape)}")
        b, n, d = x.shape
        h = self.cfg.heads
        hd = self.cfg.head_dim

        qkv = self.qkv(x).reshape(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        if store_attn:
            self.last_attn = attn.detach()

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.out(out)
        out = self.drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        inner = int(cfg.hidden * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.hidden, inner)
        self.fc2 = nn.Linear(inner, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), store_attn=store_attn)
        x = x + self.ffn(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.hidden, eps=1e-6)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, store_attn=store_attn)
        return self.norm(x)


class VisionEncoder(nn.Module):
    """完整编码器：patch 前端 + ViT 堆叠。

    返回 (tokens, cls)，其中 tokens 形状为 (B, num_patches + 1, hidden)，
    cls 形状为 (B, hidden)。
    """

    def __init__(self, cfg: ViTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ViTConfig()
        self.front = VisionFrontEnd(self.cfg.front_end_config())
        self.vit = ViT(self.cfg)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.front(x)
        tokens = self.vit(tokens, store_attn=store_attn)
        cls = tokens[:, 0]
        return tokens, cls


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    print("=" * 60)
    print("VISION TRANSFORMER 编码器")
    print("=" * 60)

    cfg = ViTConfig()
    print(f"  图像尺寸        : {cfg.image_size}")
    print(f"  patch 尺寸      : {cfg.patch_size}")
    print(f"  隐藏维度        : {cfg.hidden}")
    print(f"  深度 x head 数  : {cfg.depth} x {cfg.heads}（head 维度 {cfg.head_dim}）")
    print(f"  MLP 比率        : {cfg.mlp_ratio}")

    torch.manual_seed(0)
    encoder = VisionEncoder(cfg).eval()
    print(f"\n前端参数量       : {count_params(encoder.front):,}")
    print(f"ViT 参数量       : {count_params(encoder.vit):,}")
    print(f"总参数量         : {count_params(encoder):,}")

    img = synthesize_image(seed=0)
    print(f"\n夹具图像         : {tuple(img.shape)}")

    with torch.no_grad():
        tokens, cls = encoder(img)
    print(f"输出 token       : {tuple(tokens.shape)}")
    print(f"CLS 形状         : {tuple(cls.shape)}")
    print(f"CLS L2 范数      : {cls.norm().item():.3f}")

    print("\n逐层 CLS 范数追踪：")
    with torch.no_grad():
        x = encoder.front(img)
        print(f"  第 0 层（前端之后）：CLS 范数 {x[0, 0].norm().item():.3f}")
        for i, block in enumerate(encoder.vit.blocks, start=1):
            x = block(x)
            if i % 2 == 0 or i == cfg.depth:
                print(f"  第 {i:2d} 层             ：CLS 范数 {x[0, 0].norm().item():.3f}")
        x = encoder.vit.norm(x)
        print(f"  最终 LN             ：CLS 范数 {x[0, 0].norm().item():.3f}")

    print("\n注意力健全性检查：")
    encoder.vit.blocks[0].attn(encoder.front(img), store_attn=True)
    attn = encoder.vit.blocks[0].attn.last_attn
    if attn is not None:
        row_sums = attn[0, 0, 0].sum().item()
        print(f"  block 0 head 0 的 CLS 行和（应为 1.0）：{row_sums:.6f}")
        spread = attn[0, 0, 0].std().item()
        print(f"  block 0 head 0 的 CLS 行标准差：{spread:.4f}")

    print("\n梯度健全性检查：")
    img2 = synthesize_image(seed=2)
    enc2 = VisionEncoder(cfg)
    _, c = enc2(img2)
    loss = (c * c).sum()
    loss.backward()
    grad_norm = enc2.front.patch.proj.weight.grad.norm().item()
    cls_grad = enc2.front.cls_token.grad.norm().item()
    print(f"  patch.proj.weight 梯度范数：{grad_norm:.3e}")
    print(f"  front.cls_token 梯度范数  ：{cls_grad:.3e}")
    print("  ok: 梯度从 CLS 反向流经整个编码器")

    print("\n完成。")


if __name__ == "__main__":
    main()
