"""视觉-语言预训练：对比 InfoNCE 加语言建模。

模型组合了小型 ViT 编码器（第 59 课）、两层投影（第 60 课）
和交叉注意力解码器（第 61 课）。训练在合成的 200 对 mock 语料上
运行 50 步。对比损失和 LM 损失通过编码器和投影共享梯度。

运行方式：python3 main.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
LESSON_59 = THIS_DIR.parent.parent / "59-vit-transformer" / "code"
LESSON_60 = THIS_DIR.parent.parent / "60-projection-layer-modality-align" / "code"
LESSON_61 = THIS_DIR.parent.parent / "61-cross-attention-fusion" / "code"


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _load_module("vit_encoder_lesson59", LESSON_59 / "main.py")
_align_mod = _load_module("align_lesson60", LESSON_60 / "main.py")
_dec_mod = _load_module("decoder_lesson61", LESSON_61 / "main.py")

ViTConfig = _encoder_mod.ViTConfig
VisionEncoder = _encoder_mod.VisionEncoder
synthesize_image = _encoder_mod.synthesize_image
MLPProjector = _align_mod.MLPProjector
DecoderConfig = _dec_mod.DecoderConfig
VisionLanguageDecoder = _dec_mod.VisionLanguageDecoder


PAD_ID = 0


@dataclass(frozen=True)
class PretrainConfig:
    vision_hidden: int = 128
    projection_hidden: int = 256
    embed_dim: int = 128
    text_vocab: int = 512
    max_text_len: int = 16
    n_pairs: int = 200
    batch_size: int = 16
    steps: int = 50
    lr: float = 5e-4
    lm_weight: float = 1.0
    init_log_tau: float = math.log(1.0 / 0.07)
    seed: int = 0


def info_nce_loss(image_emb: torch.Tensor, text_emb: torch.Tensor,
                  log_tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CLIP 等模型中使用的双向 InfoNCE。

返回 (loss, similarity_matrix)。image_emb 和 text_emb 必须具有
相同形状 (N, D)。相似度矩阵在语义上对称但数值上不对称
（行为图像，列为文本）。
"""
    if image_emb.shape != text_emb.shape:
        raise ValueError(
            f"形状不匹配：image {tuple(image_emb.shape)} vs text {tuple(text_emb.shape)}"
        )
    n = image_emb.shape[0]
    img_n = F.normalize(image_emb, dim=-1)
    txt_n = F.normalize(text_emb, dim=-1)

    scale = log_tau.exp().clamp(min=1e-3, max=100.0)
    sim = (img_n @ txt_n.T) * scale

    targets = torch.arange(n, device=sim.device)
    loss_i2t = F.cross_entropy(sim, targets)
    loss_t2i = F.cross_entropy(sim.T, targets)
    return (loss_i2t + loss_t2i) * 0.5, sim


def lm_loss(logits: torch.Tensor, target_ids: torch.Tensor,
            padding_id: int = PAD_ID) -> torch.Tensor:
    """带 padding 掩码的下一 token 交叉熵。

`logits` 形状为 (B, L, V)。`target_ids` 形状为 (B, L)。
位移在本函数外部完成，因此调用方控制哪些位置是预测、哪些是输入。
"""
    if logits.dim() != 3 or target_ids.dim() != 2:
        raise ValueError(f"logits 必须为 3D 且 targets 为 2D，得到 {logits.shape} {target_ids.shape}")
    b, l, v = logits.shape
    flat_logits = logits.reshape(b * l, v)
    flat_target = target_ids.reshape(b * l)
    return F.cross_entropy(flat_logits, flat_target, ignore_index=padding_id)


class TextSideEncoder(nn.Module):
    """轻量文本编码器：embedding 查表 + 对非 padding token 做均值池化。"""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(f"期望 (B, L)，得到 {tuple(ids.shape)}")
        x = self.embed(ids)
        mask = (ids != PAD_ID).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom


class MultimodalModel(nn.Module):
    """编码器 + 投影 + 文本侧 + 交叉注意力解码器，全部可训练。"""

    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

        vit_cfg = ViTConfig(
            image_size=32,
            patch_size=16,
            hidden=cfg.vision_hidden,
            depth=2,
            heads=4,
            mlp_ratio=2.0,
        )
        self.encoder = VisionEncoder(vit_cfg)
        self.projector = MLPProjector(cfg.vision_hidden, cfg.projection_hidden, cfg.embed_dim)
        self.text_encoder = TextSideEncoder(cfg.text_vocab, cfg.embed_dim)

        dec_cfg = DecoderConfig(
            hidden=cfg.embed_dim,
            heads=4,
            depth=2,
            mlp_ratio=2.0,
            text_vocab=cfg.text_vocab,
            max_text_len=cfg.max_text_len,
            vision_dim=cfg.vision_hidden,
            vision_tokens=(32 // 16) ** 2 + 1,
        )
        self.decoder = VisionLanguageDecoder(dec_cfg)

        self.log_tau = nn.Parameter(torch.tensor(cfg.init_log_tau))

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, cls = self.encoder(images)
        return tokens, self.projector(cls)

    def caption_logits(self, memory: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(text_ids, memory)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        memory, image_emb = self.encode_image(images)
        text_emb = self.text_encoder(text_ids)

        contrast, sim = info_nce_loss(image_emb, text_emb, self.log_tau)

        b, l = text_ids.shape
        inputs = text_ids[:, :-1]
        targets = text_ids[:, 1:]
        if inputs.shape[1] == 0:
            lm = torch.tensor(0.0, device=images.device)
        else:
            logits = self.caption_logits(memory, inputs)
            lm = lm_loss(logits, targets, padding_id=PAD_ID)

        diag = sim.diag().mean().item()
        offdiag = (sim.sum() - sim.diag().sum()).item() / max(1, b * b - b)
        stats = {"diag": diag, "off_diag": offdiag, "tau": self.log_tau.exp().item()}
        return contrast, lm, stats


def make_mock_corpus(seed: int, n_pairs: int, vocab_size: int, max_len: int
                     ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """构建确定性的 mock 语料，包含 n_pairs 个合成 image-caption 对。

caption token 与图像 seed 相关，使模型在对比 batch 中能获得少量
可学习的信号。token id 0 保留为 padding。
"""
    if vocab_size <= 50:
        raise ValueError(f"vocab_size 必须 > 50，得到 {vocab_size}")
    pairs = []
    rng = np.random.default_rng(seed)
    for i in range(n_pairs):
        img_seed = seed * 100 + i
        rng_i = np.random.default_rng(img_seed)
        noise = rng_i.standard_normal((3, 32, 32)).astype("float32") * 0.2
        gx, gy = np.meshgrid(np.linspace(0.0, 1.0, 32), np.linspace(0.0, 1.0, 32))
        bias = (i % 7) / 7.0
        img = np.clip(noise + bias, -1.0, 1.0).astype("float32")
        img = torch.from_numpy(img).unsqueeze(0)

        length = min(6 + (i % 8), max_len)
        ids = np.zeros((max_len,), dtype=np.int64)
        base = (i * 17) % (vocab_size - 50)
        for j in range(length):
            ids[j] = 1 + (base + j * 3 + (i % 5)) % (vocab_size - 1)
        pairs.append((img, torch.from_numpy(ids).unsqueeze(0)))
    return pairs


def sample_batch(pairs: list[tuple[torch.Tensor, torch.Tensor]], indices: list[int]
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.cat([pairs[i][0] for i in indices], dim=0)
    ids = torch.cat([pairs[i][1] for i in indices], dim=0)
    return imgs, ids


def train(cfg: PretrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    model = MultimodalModel(cfg).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    corpus = make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    if cfg.batch_size > len(corpus):
        raise ValueError(
            f"batch_size（{cfg.batch_size}）不能超过语料大小（{len(corpus)}） "
            "（replace=False）"
        )

    rng = np.random.default_rng(cfg.seed + 2)
    history = {"contrast": [], "lm": [], "total": []}

    for step in range(cfg.steps):
        idx = rng.choice(len(corpus), size=cfg.batch_size, replace=False).tolist()
        imgs, ids = sample_batch(corpus, idx)
        contrast, lm, stats = model(imgs, ids)
        total = contrast + cfg.lm_weight * lm
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        history["contrast"].append(contrast.item())
        history["lm"].append(lm.item())
        history["total"].append(total.item())

        if step % 5 == 0 or step == cfg.steps - 1:
            print(f"  步骤 {step:3d}  对比损失 {contrast.item():.4f}  "
                  f"lm {lm.item():.4f}  tau {stats['tau']:.3f}  "
                  f"diag {stats['diag']:+.3f}  off {stats['off_diag']:+.3f}")
    return history


def main() -> None:
    print("=" * 60)
    print("视觉-语言预训练")
    print("=" * 60)

    cfg = PretrainConfig()
    print(f"  文本词表大小   : {cfg.text_vocab}")
    print(f"  最大文本长度   : {cfg.max_text_len}")
    print(f"  嵌入维度       : {cfg.embed_dim}")
    print(f"  配对数量       : {cfg.n_pairs}")
    print(f"  批次大小       : {cfg.batch_size}")
    print(f"  步数           : {cfg.steps}")
    print(f"  LM 权重        : {cfg.lm_weight}")
    print(f"  初始 tau       : {math.exp(cfg.init_log_tau):.3f}")

    print("\n训练：")
    hist = train(cfg)

    init_contrast = hist["contrast"][0]
    final_contrast = hist["contrast"][-1]
    init_lm = hist["lm"][0]
    final_lm = hist["lm"][-1]
    print(f"\n对比损失：{init_contrast:.4f} -> {final_contrast:.4f}"
          f"  （下降 {init_contrast - final_contrast:+.4f}）")
    print(f"LM 损失：{init_lm:.4f} -> {final_lm:.4f}"
          f"  （下降 {init_lm - final_lm:+.4f}）")

    if final_contrast < init_contrast and final_lm < init_lm:
        print("通过：两个损失均下降")
    elif final_contrast < init_contrast or final_lm < init_lm:
        print("部分通过：至少一个损失下降")
    else:
        print("失败：两个损失均未下降")

    print("\n完成。")


if __name__ == "__main__":
    main()
