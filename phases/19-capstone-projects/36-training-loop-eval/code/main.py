"""第 35 课 GPT 模型的训练循环与评测框架。

实现内容包括：输入/目标错开一位的批次构造、`calc_loss_batch` 中的交叉熵损失、
`evaluate_model` 中的留出评测、`generate_and_print_sample` 中的定性生成探针、
区分衰减/不衰减参数的 AdamW、线性预热加余弦学习率调度、梯度范数裁剪，
以及记录每步损失的 `outputs/losses.jsonl` JSONL 日志。

演示在合成字节级 token 上用少量步骤训练微型模型，写入 JSONL 日志，
并在探针点打印评测损失和生成样本。CPU 上端到端运行远少于一分钟。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUTS / "losses.jsonl"


@dataclass
class TrainConfig:
    """演示运行使用的训练与评测超参数。"""

    batch_size: int = 4
    context_length: int = 32
    num_steps: int = 80
    eval_every: int = 20
    eval_batches: int = 4
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    sample_max_new_tokens: int = 16
    seed: int = 0


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 32
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_expansion: int = 4
    dropout: float = 0.1
    use_bias: bool = True
    weight_tying: bool = True


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.context_length = cfg.context_length
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.use_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.use_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        mask = torch.triu(
            torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:seq, :seq], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_expansion * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=cfg.use_bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.lm_head.weight = self.tok_embed.weight
        self.register_buffer(
            "position_ids",
            torch.arange(cfg.context_length, dtype=torch.long),
            persistent=False,
        )
        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2 * cfg.num_layers)
        for block in self.blocks:
            block.attn.out_proj.weight.data.mul_(scale)
            block.mlp.fc2.weight.data.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        tok = self.tok_embed(tokens)
        pos = self.pos_embed(self.position_ids[:seq])
        x = self.embed_dropout(tok + pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_ln(x))


def make_batches(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """生成 (input, target) 批次，其中 target 是错开一位的 input。

    在有效起始位置上均匀随机采样。固定种子时，批次序列可跨运行复现。
    """
    if token_ids.dim() != 1:
        raise ValueError("token_ids must be a 1D tensor")
    if token_ids.numel() < context_length + 1:
        raise ValueError("token_ids too short for the requested context_length")

    generator = torch.Generator().manual_seed(seed)
    max_start = token_ids.numel() - context_length - 1

    while True:
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
        inputs = torch.stack([token_ids[s : s + context_length] for s in starts.tolist()])
        targets = torch.stack(
            [token_ids[s + 1 : s + 1 + context_length] for s in starts.tolist()]
        )
        yield inputs, targets


def calc_loss_batch(
    model: GPTModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """执行前向传播，展平批次和时间维度，返回标量交叉熵。"""
    logits = model(inputs)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


@torch.no_grad()
def evaluate_model(
    model: GPTModel,
    val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    max_batches: int,
) -> float:
    """在 `max_batches` 个验证批次上计算平均交叉熵；不计算梯度，不使用 dropout。"""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for inputs, targets in val_loader:
        if count >= max_batches:
            break
        loss = calc_loss_batch(model, inputs, targets)
        total += float(loss.item())
        count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


@torch.no_grad()
def generate_and_print_sample(
    model: GPTModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 40,
    seed: int = 0,
) -> list[int]:
    """打印从固定 prompt 生成的短续写，并返回 token。"""
    sample_gen = torch.Generator(device=prompt.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    tokens = prompt.clone()
    for _ in range(max_new_tokens):
        window = tokens[:, -model.cfg.context_length :]
        logits = model(window)
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_k_eff = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, top_k_eff, dim=-1)
            threshold = values[..., -1:]
            next_logits = torch.where(
                next_logits < threshold, torch.full_like(next_logits, float("-inf")), next_logits
            )
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=sample_gen)
        tokens = torch.cat([tokens, next_token], dim=1)
    if was_training:
        model.train()
    seq = tokens.tolist()[0]
    print(f"  样本 token             : {seq}")
    return seq


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """拆分参数：矩阵形张量使用衰减，缩放/偏置/嵌入偏置不使用。"""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or name.endswith(".bias") or name.endswith(".shift") or name.endswith(".scale"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """先线性预热，再在剩余步骤中按余弦衰减到 min_lr。"""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def train(
    model: GPTModel,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    cfg: TrainConfig,
    prompt: torch.Tensor,
    log_path: Path = LOG_PATH,
) -> list[dict]:
    """运行训练循环，持久化 losses.jsonl，并返回内存日志。"""
    torch.manual_seed(cfg.seed)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.max_lr,
        betas=(0.9, 0.95),
    )

    train_loader = make_batches(train_tokens, cfg.batch_size, cfg.context_length, seed=cfg.seed)

    if log_path.exists():
        log_path.unlink()

    records: list[dict] = []
    model.train()
    for step in range(cfg.num_steps):
        lr = cosine_with_warmup(step, cfg.warmup_steps, cfg.num_steps, cfg.max_lr, cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = next(train_loader)
        optimizer.zero_grad(set_to_none=True)
        loss = calc_loss_batch(model, inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        record = {"step": step, "train_loss": float(loss.item()), "lr": lr}

        if (step + 1) % cfg.eval_every == 0 or step == cfg.num_steps - 1:
            val_loader = make_batches(
                val_tokens, cfg.batch_size, cfg.context_length, seed=cfg.seed + 1
            )
            val_loss = evaluate_model(model, val_loader, cfg.eval_batches)
            record["val_loss"] = val_loss
            print(
                f"step {step:4d} | lr {lr:.5f} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}"
            )
            generate_and_print_sample(
                model, prompt, cfg.sample_max_new_tokens, temperature=0.8, top_k=20, seed=step
            )

        records.append(record)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    return records


def _synthetic_byte_tokens(length: int, vocab_size: int, seed: int) -> torch.Tensor:
    """确定性的合成 token。

    字节取自小型重复模式，使模型在少量步骤内就有结构可学；演示期间评测损失
    应明显下降。
    """
    rng = torch.Generator().manual_seed(seed)
    base = torch.randint(0, vocab_size, (32,), generator=rng)
    repeats = (length + base.numel() - 1) // base.numel()
    tokens = base.repeat(repeats)[:length]
    noise = torch.randint(0, vocab_size, (length,), generator=rng)
    mask = (torch.rand(length, generator=rng) < 0.1)
    tokens = torch.where(mask, noise, tokens)
    return tokens.to(dtype=torch.long)


def demo() -> None:
    torch.manual_seed(0)
    cfg = TrainConfig()
    mcfg = ModelConfig(
        vocab_size=256,
        context_length=cfg.context_length,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

    train_tokens = _synthetic_byte_tokens(length=4096, vocab_size=mcfg.vocab_size, seed=1)
    val_tokens = _synthetic_byte_tokens(length=1024, vocab_size=mcfg.vocab_size, seed=2)

    print(f"训练 token 数  : {train_tokens.numel():,}")
    print(f"验证 token 数  : {val_tokens.numel():,}")
    print(f"模型参数量      : {sum(p.numel() for p in GPTModel(mcfg).parameters()):,}（未绑定计数）")

    model = GPTModel(mcfg)
    prompt = torch.tensor([[7, 11, 13, 17]], dtype=torch.long)

    print("\n训练运行：")
    records = train(model, train_tokens, val_tokens, cfg, prompt)

    print("\n最终日志记录（最后 3 条）：")
    for record in records[-3:]:
        print(" ", record)
    print(f"\n已将损失写入 {LOG_PATH}")

    first_loss = records[0]["train_loss"]
    last_loss = records[-1]["train_loss"]
    print(f"首步 train_loss  ：{first_loss:.4f}")
    print(f"末步 train_loss  ：{last_loss:.4f}")
    assert last_loss < first_loss, "演示运行期间训练损失应下降"
    print("训练循环检查通过。")


if __name__ == "__main__":
    demo()
