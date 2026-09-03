"""两层 MLP 投影：从视觉 token 空间到文本 embedding 空间。

视觉编码器（第 58 和 59 课）保持冻结。一个冻结的 mock 文本
embedding 表为合成 caption 提供目标向量。只有投影器参与训练。
优化目标为成对的余弦相似度对齐。

运行方式：python3 main.py
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
LESSON_59 = THIS_DIR.parent.parent / "59-vit-transformer" / "code"


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


_encoder_mod = _load_module("vit_encoder_lesson59", LESSON_59 / "main.py")
ViTConfig = _encoder_mod.ViTConfig
VisionEncoder = _encoder_mod.VisionEncoder
synthesize_image = _encoder_mod.synthesize_image


@dataclass(frozen=True)
class AlignConfig:
    vision_hidden: int = 768
    projection_hidden: int = 1024
    text_hidden: int = 512
    vocab_size: int = 4096
    max_caption_len: int = 16
    pairs: int = 32
    steps: int = 200
    lr: float = 3e-4
    seed: int = 0


class MLPProjector(nn.Module):
    """两层 MLP，LLaVA 风格 VLM 中经典的适配器结构。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MockTextEmbedding(nn.Module):
    """作为对齐目标的冻结文本表。

caption 是 token id 序列；caption 的 embedding 是各 id embedding 的均值。
给定 seed 时结果确定。
"""

    def __init__(self, vocab_size: int, dim: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(vocab_size, dim, generator=gen) * (1.0 / dim ** 0.5)
        self.table = nn.Embedding(vocab_size, dim, _weight=weight)
        for p in self.table.parameters():
            p.requires_grad_(False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(f"期望 (B, L) 的 ids，得到 {tuple(ids.shape)}")
        embed = self.table(ids)
        mask = (ids != 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (embed * mask).sum(dim=1) / denom
        return pooled


def make_pair(seed: int, vocab_size: int, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一个合成的 (image, caption_ids) 对。

图像为第 58 课中确定的 224x224x3 固定样本，并带有每对独立的 seed。
caption 为长度为 `max_len` 的 token id 序列，同样由 seed 确定。
token id 0 保留为 padding。
"""
    img = synthesize_image(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    length = int(rng.integers(4, max_len + 1))
    ids = np.zeros((max_len,), dtype=np.int64)
    ids[:length] = rng.integers(1, vocab_size, size=length)
    return img, torch.from_numpy(ids).unsqueeze(0)


def cosine_alignment_loss(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    if image_emb.shape != text_emb.shape:
        raise ValueError(
            f"形状不匹配：image {tuple(image_emb.shape)} vs text {tuple(text_emb.shape)}"
        )
    img_n = F.normalize(image_emb, dim=-1)
    txt_n = F.normalize(text_emb, dim=-1)
    cos = (img_n * txt_n).sum(dim=-1)
    return (1.0 - cos).mean()


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


@dataclass
class TrainStats:
    initial_loss: float
    final_loss: float
    final_cos: float
    losses: list[float]


def train(cfg: AlignConfig) -> tuple[MLPProjector, TrainStats]:
    if cfg.pairs <= 0:
        raise ValueError(f"pairs 必须 > 0，得到 {cfg.pairs}")
    if cfg.steps <= 0:
        raise ValueError(f"steps 必须 > 0，得到 {cfg.steps}")
    if cfg.max_caption_len < 4:
        raise ValueError(
            f"make_pair() 要求 max_caption_len >= 4，得到 {cfg.max_caption_len}"
        )

    torch.manual_seed(cfg.seed)

    encoder_cfg = ViTConfig(image_size=224, patch_size=16, hidden=cfg.vision_hidden,
                            depth=4, heads=8, mlp_ratio=2.0)
    encoder = VisionEncoder(encoder_cfg).eval()
    freeze(encoder)

    text = MockTextEmbedding(cfg.vocab_size, cfg.text_hidden, seed=cfg.seed + 1)
    freeze(text)

    projector = MLPProjector(cfg.vision_hidden, cfg.projection_hidden, cfg.text_hidden)

    pairs = [make_pair(seed=cfg.seed + 1000 + i,
                       vocab_size=cfg.vocab_size,
                       max_len=cfg.max_caption_len) for i in range(cfg.pairs)]

    opt = torch.optim.Adam(projector.parameters(), lr=cfg.lr)
    losses: list[float] = []

    initial_loss = 0.0
    final_loss = 0.0
    final_cos = 0.0
    for step in range(cfg.steps):
        img, ids = pairs[step % cfg.pairs]
        with torch.no_grad():
            _, cls = encoder(img)
            text_emb = text(ids)

        image_emb = projector(cls)
        loss = cosine_alignment_loss(image_emb, text_emb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step == 0:
            initial_loss = loss.item()
        if step % 25 == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                cos = F.cosine_similarity(image_emb, text_emb).mean().item()
            print(f"  步骤 {step:4d}  loss {loss.item():.4f}  cos {cos:+.4f}")
        if step == cfg.steps - 1:
            final_loss = loss.item()
            with torch.no_grad():
                final_cos = F.cosine_similarity(image_emb, text_emb).mean().item()

    return projector, TrainStats(
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_cos=final_cos,
        losses=losses,
    )


def main() -> None:
    print("=" * 60)
    print("模态对齐的投影层")
    print("=" * 60)

    cfg = AlignConfig()
    print(f"  视觉隐藏维度      : {cfg.vision_hidden}")
    print(f"  投影隐藏维度      : {cfg.projection_hidden}")
    print(f"  文本隐藏维度      : {cfg.text_hidden}")
    print(f"  词表大小          : {cfg.vocab_size}")
    print(f"  配对数量          : {cfg.pairs}")
    print(f"  步数              : {cfg.steps}")
    print(f"  学习率            : {cfg.lr}")

    print("\n训练中（视觉编码器冻结、文本表冻结、投影器训练）：")
    projector, stats = train(cfg)

    n_proj = sum(p.numel() for p in projector.parameters())
    print(f"\n投影器参数量  : {n_proj:,}")
    print(f"初始损失       : {stats.initial_loss:.4f}")
    print(f"最终损失       : {stats.final_loss:.4f}")
    print(f"最终余弦相似度 : {stats.final_cos:+.4f}")
    drop = stats.initial_loss - stats.final_loss
    print(f"损失下降       : {drop:.4f}")
    if drop > 0.0:
        print("  通过：投影器学到了对齐方向")
    else:
        print("  失败：损失未下降")

    print("\n完成。")


if __name__ == "__main__":
    main()
