"""视觉-语言解码器的交叉注意力融合。

解码器块执行：
  1. 对文本 token 的因果自注意力
  2. 以文本为 query、图像 memory 为 key/value 的交叉注意力
  3. 前馈 MLP

掩码规则：
  - 自注意力使用 (Nt, Nt) 的下三角因果掩码
  - 交叉注意力不使用掩码；整张图像对每个文本位置都可见

运行方式：python3 main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DecoderConfig:
    hidden: int = 256
    heads: int = 8
    depth: int = 4
    mlp_ratio: float = 4.0
    text_vocab: int = 1024
    max_text_len: int = 32
    vision_dim: int = 256
    vision_tokens: int = 197
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.hidden % self.heads != 0:
            raise ValueError(f"hidden {self.hidden} 不能被 heads {self.heads} 整除")
        return self.hidden // self.heads


def causal_mask(length: int) -> torch.Tensor:
    """形状为 (length, length) 的下三角布尔掩码。

当 token i 可以注意到 token j（j <= i）时，单元格 [i, j] 为 True。
"""
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.hidden, cfg.hidden * 3, bias=True)
        self.out = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        h, hd = self.cfg.heads, self.cfg.head_dim
        qkv = self.qkv(x).reshape(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.shape != (n, n):
                raise ValueError(
                    f"因果掩码形状 {tuple(mask.shape)} 与 (n, n) = ({n}, {n}) 不匹配"
                )
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.drop(self.out(out))


class CrossAttention(nn.Module):
    """多头交叉注意力。

query 来自文本 token；key 和 value 来自图像 memory。
支持 kv_cache 参数，使图像 memory 的投影只需计算一次并在多个解码步骤中复用。
"""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.kv_proj = nn.Linear(cfg.vision_dim, cfg.hidden * 2, bias=True)
        self.out = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def project_memory(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if memory.dim() != 3:
            raise ValueError(f"期望 (B, Nv, vision_dim)，得到 {tuple(memory.shape)}")
        b, nv, _ = memory.shape
        h, hd = self.cfg.heads, self.cfg.head_dim
        kv = self.kv_proj(memory).reshape(b, nv, 2, h, hd).permute(2, 0, 3, 1, 4)
        return kv[0], kv[1]

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"期望 (B, Nt, hidden)，得到 {tuple(x.shape)}")
        if memory.shape[0] != x.shape[0]:
            raise ValueError(
                f"batch 不匹配：text {x.shape[0]} vs memory {memory.shape[0]}"
            )
        b, nt, d = x.shape
        h, hd = self.cfg.heads, self.cfg.head_dim

        q = self.q_proj(x).reshape(b, nt, h, hd).transpose(1, 2)
        if kv_cache is None:
            k, v = self.project_memory(memory)
        else:
            k, v = kv_cache
            expected = (b, h, memory.shape[1], hd)
            if k.shape != expected or v.shape != expected:
                raise ValueError(
                    f"kv_cache 必须为 (B,H,Nv,hd)={expected}，得到 "
                    f"k={tuple(k.shape)} v={tuple(v.shape)}"
                )

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, nt, d)
        return self.drop(self.out(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        inner = int(cfg.hidden * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.hidden, inner)
        self.fc2 = nn.Linear(inner, cfg.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.self_attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.cross_attn = CrossAttention(cfg)
        self.ln3 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                text_mask: torch.Tensor,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), mask=text_mask)
        x = x + self.cross_attn(self.ln2(x), memory, kv_cache=kv_cache)
        x = x + self.ffn(self.ln3(x))
        return x


class VisionLanguageDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.text_vocab, cfg.hidden)
        self.pos_emb = nn.Embedding(cfg.max_text_len, cfg.hidden)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.head = nn.Linear(cfg.hidden, cfg.text_vocab, bias=False)

    def build_kv_cache(self, memory: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cache = []
        for block in self.blocks:
            k, v = block.cross_attn.project_memory(memory)
            cache.append((k, v))
        return cache

    def forward(self, text_ids: torch.Tensor, memory: torch.Tensor,
                use_cache: bool = False) -> torch.Tensor:
        if text_ids.dim() != 2:
            raise ValueError(f"期望 (B, Nt) 的 ids，得到 {tuple(text_ids.shape)}")
        b, nt = text_ids.shape
        if nt > self.cfg.max_text_len:
            raise ValueError(f"文本长度 {nt} 超过最大值 {self.cfg.max_text_len}")

        positions = torch.arange(nt, device=text_ids.device)
        x = self.tok_emb(text_ids) + self.pos_emb(positions).unsqueeze(0)

        mask = causal_mask(nt).to(text_ids.device)

        cache = self.build_kv_cache(memory) if use_cache else [None] * len(self.blocks)

        for block, kv in zip(self.blocks, cache):
            x = block(x, memory=memory, text_mask=mask, kv_cache=kv)

        x = self.norm(x)
        return self.head(x)


def synth_memory(batch: int, n_tokens: int, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_tokens, dim, generator=gen)


def synth_text(batch: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 1)
    return torch.randint(low=0, high=vocab, size=(batch, length), generator=gen)


def main() -> None:
    print("=" * 60)
    print("交叉注意力融合解码器")
    print("=" * 60)

    cfg = DecoderConfig()
    print(f"  隐藏维度        : {cfg.hidden}")
    print(f"  head 数         : {cfg.heads}（head 维度 {cfg.head_dim}）")
    print(f"  深度            : {cfg.depth}")
    print(f"  文本词表大小    : {cfg.text_vocab}")
    print(f"  最大文本长度    : {cfg.max_text_len}")
    print(f"  视觉 token 数   : {cfg.vision_tokens}")
    print(f"  视觉维度        : {cfg.vision_dim}")

    torch.manual_seed(0)
    decoder = VisionLanguageDecoder(cfg).eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n解码器参数量 : {n_params:,}")

    text_ids = synth_text(batch=2, length=10, vocab=cfg.text_vocab, seed=0)
    memory = synth_memory(batch=2, n_tokens=cfg.vision_tokens, dim=cfg.vision_dim, seed=1)
    print(f"\ntext_ids 形状 : {tuple(text_ids.shape)}")
    print(f"memory 形状   : {tuple(memory.shape)}")

    mask = causal_mask(10)
    print(f"\n因果掩码形状 : {tuple(mask.shape)}")
    print("因果掩码左上角 5x5：")
    for row in mask[:5, :5].int().tolist():
        print("  " + " ".join(str(v) for v in row))

    with torch.no_grad():
        logits = decoder(text_ids, memory, use_cache=False)
        logits_cached = decoder(text_ids, memory, use_cache=True)
    print(f"\nlogits 形状    : {tuple(logits.shape)}")
    print(f"缓存 logits   : {tuple(logits_cached.shape)}")
    drift = (logits - logits_cached).abs().max().item()
    print(f"缓存 vs 非缓存最大漂移 : {drift:.6e}")
    if drift < 1e-4:
        print("  通过：KV cache 路径与非缓存一致")
    else:
        print("  失败：cache 漂移超出容差")

    print("\n各文本位置的交叉注意力输出范数（head 0，样本 0）：")
    block = decoder.blocks[0]
    ln_x = block.ln2(decoder.tok_emb(text_ids) + decoder.pos_emb(torch.arange(10)))
    with torch.no_grad():
        cross_out = block.cross_attn(ln_x, memory)
    norms = cross_out[0].norm(dim=-1).tolist()
    for i, val in enumerate(norms):
        print(f"  位置 {i:2d}  范数 {val:.3f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
