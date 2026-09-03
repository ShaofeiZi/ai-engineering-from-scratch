"""
从零实现直接偏好优化（DPO）。

参见：phases/19-capstone-projects/40-dpo-from-scratch/docs/en.md

构建内容：
  - 带 INST / RESP 特殊 token 的 InstructionTokenizer（字节级）
  - TinyGPT（仅解码器因果 Transformer）
  - (prompt, chosen, rejected) 三元组偏好夹具
  - 掩蔽 prompt，并对补全部分的下一 token 对数概率求和的 sequence_log_prob
  - 实现下式的 dpo_loss：
       L = -log sigmoid( beta * ( (logp_w_pol - logp_w_ref)
                                - (logp_l_pol - logp_l_ref) ) )
  - 使用冻结参考模型和可训练策略模型的 train_dpo 循环
  - 打印每轮损失与 chosen-rejected 间隔的 run_demo

训练使 chosen-rejected 对数概率间隔增大时，以状态码 0 退出。
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 分词器
# ---------------------------------------------------------------------------


class InstructionTokenizer:
    INST_ID = 256
    RESP_ID = 257
    PAD_ID = 258
    VOCAB = 260

    def encode_prompt(self, prompt: str) -> List[int]:
        ids = [self.INST_ID]
        ids.extend(prompt.encode("utf-8", errors="ignore"))
        ids.append(self.RESP_ID)
        return ids

    def encode_completion(self, completion: str) -> List[int]:
        return list(completion.encode("utf-8", errors="ignore"))


# ---------------------------------------------------------------------------
# 微型 GPT
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, max_len: int):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must divide heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal_mask[:T, :T].view(1, 1, T, T)
        att = att.masked_fill(~causal, float("-inf"))
        weights = F.softmax(att, dim=-1)
        ctx = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads, max_len)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, hidden: int, heads: int, depth: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, max_len) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.max_len = max_len

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(positions)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# 偏好夹具
# ---------------------------------------------------------------------------


def make_preferences() -> List[Dict[str, str]]:
    """覆盖简单任务类型的十二个偏好三元组。"""
    return [
        {
            "prompt": "What is the capital of France?",
            "chosen": "Paris.",
            "rejected": "France is in Europe and has many beautiful cities including Paris.",
        },
        {
            "prompt": "What is the capital of Japan?",
            "chosen": "Tokyo.",
            "rejected": "Japan is an island nation. Its government sits in Tokyo.",
        },
        {
            "prompt": "What is the capital of Spain?",
            "chosen": "Madrid.",
            "rejected": "Spain has many cities. Madrid is the largest of them.",
        },
        {
            "prompt": "Compute 2 + 3.",
            "chosen": "5.",
            "rejected": "Let me think. 2 plus 3 is something close to 5 I believe.",
        },
        {
            "prompt": "Compute 7 * 6.",
            "chosen": "42.",
            "rejected": "7 multiplied by 6 gives a number around the forties.",
        },
        {
            "prompt": "Compute 12 / 4.",
            "chosen": "3.",
            "rejected": "Twelve divided by four is roughly three or so.",
        },
        {
            "prompt": "List three colors.",
            "chosen": "red, green, blue.",
            "rejected": "Colors are everywhere. Some of them are red, green, and there is blue too.",
        },
        {
            "prompt": "List three vowels.",
            "chosen": "a, e, i.",
            "rejected": "Vowels are letters that produce open mouth sounds, like a and e and also i.",
        },
        {
            "prompt": "Define variable.",
            "chosen": "a name bound to a value.",
            "rejected": "A variable is a thing that you can use in programming to store stuff.",
        },
        {
            "prompt": "Define function.",
            "chosen": "a reusable block of code that returns an output.",
            "rejected": "A function is basically something that does things when you call it on inputs.",
        },
        {
            "prompt": "Python: print 42.",
            "chosen": "print(42)",
            "rejected": "You can print numbers in python. For 42 you would call print on it.",
        },
        {
            "prompt": "Python: sort items.",
            "chosen": "items.sort()",
            "rejected": "Sorting a list in python is easy, just call sort on the items list.",
        },
    ]


# ---------------------------------------------------------------------------
# 对数概率机制
# ---------------------------------------------------------------------------


def sequence_log_prob(
    model: TinyGPT,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
) -> torch.Tensor:
    """对以 prompt 为条件的补全 token 对数概率求和。

    返回与模型位于同一设备上的零维张量。

    实现：
      - 拼接 prompt + completion。
      - 通过模型执行前向传播。
      - 对 logits 计算 log-softmax。
      - 对每个补全位置 i（按完整序列计数），取出
        log p(completion[i] | tokens[<i]) 并求和。
    """
    if len(completion_ids) == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    full = list(prompt_ids) + list(completion_ids)
    if len(full) > model.max_len:
    # 从左侧截断，以保留最近的上下文。
        full = full[-model.max_len :]
        prompt_len = max(0, len(full) - len(completion_ids))
    else:
        prompt_len = len(prompt_ids)
    ids = torch.tensor([full], dtype=torch.long, device=next(model.parameters()).device)
    logits = model(ids)
    log_probs = F.log_softmax(logits, dim=-1)
    # 位置 i 预测 token i+1。补全位于索引 [prompt_len, len(full))。
    # 对该范围内的 k，需要 log p(索引 k 处的 token | 截至 k-1 的 token)。
    # 该概率为 log_probs[0, k-1, token_k]。
    completion_targets = torch.tensor(full[prompt_len:], dtype=torch.long, device=ids.device)
    pred_positions = torch.arange(prompt_len - 1, len(full) - 1, device=ids.device)
    # 防御 prompt_len == 0 的退化情况。
    if prompt_len == 0:
        pred_positions = torch.arange(0, len(full) - 1, device=ids.device)
        completion_targets = torch.tensor(full[1:], dtype=torch.long, device=ids.device)
    gathered = log_probs[0, pred_positions, completion_targets]
    return gathered.sum()


def dpo_loss(
    logp_w_pol: torch.Tensor,
    logp_l_pol: torch.Tensor,
    logp_w_ref: torch.Tensor,
    logp_l_ref: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算单个样本的 DPO 损失与隐式奖励间隔。

    L = -log sigmoid( beta * ( (logp_w_pol - logp_w_ref) - (logp_l_pol - logp_l_ref) ) )

    返回 ``(loss_scalar, reward_margin)``，其中 reward_margin 是 sigmoid
    的输入除以 beta（即隐式奖励差）。
    """
    diff_w = logp_w_pol - logp_w_ref
    diff_l = logp_l_pol - logp_l_ref
    margin = diff_w - diff_l
    z = beta * margin
    # logsigmoid 在数值上更稳定；loss 是单样本标量。
    loss = -F.logsigmoid(z)
    return loss, margin


def ipo_loss(
    logp_w_pol: torch.Tensor,
    logp_l_pol: torch.Tensor,
    logp_w_ref: torch.Tensor,
    logp_l_ref: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IPO 变体：不会饱和的平方损失。

    L_IPO = ( ( (logp_w_pol - logp_w_ref) - (logp_l_pol - logp_l_ref) ) - 1 / (2 * beta) ) ** 2

    ``1 / (2 * beta)`` 偏移量是标准 IPO 目标间隔。本课程提供该变体用于
    扩展对比；演示和 DPO 测试不使用它。
    """
    diff_w = logp_w_pol - logp_w_ref
    diff_l = logp_l_pol - logp_l_ref
    margin = diff_w - diff_l
    target = 1.0 / (2.0 * beta) if beta > 0 else 0.0
    loss = (margin - target) ** 2
    return loss, margin


def length_normalised_log_prob(
    model: TinyGPT,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
) -> torch.Tensor:
    """用序列对数概率除以补全长度。

    可用于诊断长度偏差：如果长度归一化后的间隔为正，而原始间隔为负
    （或反之），说明模型的偏好对长度敏感。
    """
    if len(completion_ids) == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    raw = sequence_log_prob(model, prompt_ids, completion_ids)
    return raw / float(len(completion_ids))


@dataclass(frozen=True)
class MarginRow:
    prompt: str
    chosen: str
    rejected: str
    margin: float
    chosen_logprob: float
    rejected_logprob: float


def margin_table(
    policy: TinyGPT,
    tok: InstructionTokenizer,
    triples: Sequence[Dict[str, str]],
) -> List[MarginRow]:
    """生成策略模型下逐三元组的间隔报告，便于调试。"""
    rows: List[MarginRow] = []
    with torch.no_grad():
        for tri in triples:
            prompt = tok.encode_prompt(tri["prompt"])
            chosen = tok.encode_completion(tri["chosen"])
            rejected = tok.encode_completion(tri["rejected"])
            lp_w = sequence_log_prob(policy, prompt, chosen).item()
            lp_l = sequence_log_prob(policy, prompt, rejected).item()
            rows.append(
                MarginRow(
                    prompt=tri["prompt"],
                    chosen=tri["chosen"],
                    rejected=tri["rejected"],
                    margin=lp_w - lp_l,
                    chosen_logprob=lp_w,
                    rejected_logprob=lp_l,
                )
            )
    return rows


def print_margin_table(rows: Sequence[MarginRow], log: Callable[[str], None] = print) -> None:
    log("  margin   chosen_lp   rejected_lp   提示词")
    log("  -------  ----------  ------------  -------------------------")
    for row in rows:
        log(
            f"  {row.margin:+.4f}   {row.chosen_logprob:+.4f}    {row.rejected_logprob:+.4f}     {row.prompt[:35]}"
        )


# ---------------------------------------------------------------------------
# 参考模型/策略模型管理
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig:
    vocab: int = InstructionTokenizer.VOCAB
    hidden: int = 64
    heads: int = 4
    depth: int = 2
    max_len: int = 96
    beta: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    seed: int = 0
    warmup_epochs: int = 8  # 短暂预训练参考模型，使对数概率不再是无意义的初始值


def build_models(cfg: DPOConfig) -> Tuple[TinyGPT, TinyGPT]:
    """构建参考模型和策略模型。策略模型用参考模型的 state dict 初始化，
    因此二者起点相同；随后策略模型在 DPO 训练中逐渐分化，而参考模型保持冻结。"""
    torch.manual_seed(cfg.seed)
    reference = TinyGPT(cfg.vocab, cfg.hidden, cfg.heads, cfg.depth, cfg.max_len)
    torch.manual_seed(cfg.seed)  # 重新设种子，确保训练前策略模型权重一致
    policy = TinyGPT(cfg.vocab, cfg.hidden, cfg.heads, cfg.depth, cfg.max_len)
    policy.load_state_dict(reference.state_dict())
    # 冻结参考模型。
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    return reference, policy


def warmup_pretrain(
    model: TinyGPT,
    tok: InstructionTokenizer,
    triples: Sequence[Dict[str, str]],
    epochs: int = 8,
    lr: float = 3e-3,
    seed: int = 0,
) -> List[float]:
    """在 chosen 补全上进行短暂的下一 token 预训练，使参考模型能为
    夹具中的任务结构给出有意义的概率。"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    model.train()
    sequences: List[List[int]] = []
    for tri in triples:
        prompt = tok.encode_prompt(tri["prompt"])
        chosen = tok.encode_completion(tri["chosen"])
        sequences.append(prompt + chosen)
    for _ in range(epochs):
        ep_loss = 0.0
        for seq in sequences:
            if len(seq) > model.max_len:
                seq = seq[: model.max_len]
            ids = torch.tensor([seq], dtype=torch.long)
            logits = model(ids)
            pred = logits[:, :-1, :].contiguous()
            target = ids[:, 1:].contiguous()
            loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        losses.append(ep_loss / max(len(sequences), 1))
    return losses


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------


@dataclass
class DPOReport:
    losses: List[float] = field(default_factory=list)
    margins: List[float] = field(default_factory=list)
    initial_margin: float = 0.0
    final_margin: float = 0.0


def evaluate_margins(
    policy: TinyGPT,
    reference: TinyGPT,
    tok: InstructionTokenizer,
    triples: Sequence[Dict[str, str]],
) -> float:
    """策略模型下 ``chosen - rejected`` 对数概率差的均值。

    未经 DPO 训练时该值可能任意；DPO 训练会推动它变为正值。
    """
    margins: List[float] = []
    with torch.no_grad():
        for tri in triples:
            prompt = tok.encode_prompt(tri["prompt"])
            chosen = tok.encode_completion(tri["chosen"])
            rejected = tok.encode_completion(tri["rejected"])
            lp_w = sequence_log_prob(policy, prompt, chosen).item()
            lp_l = sequence_log_prob(policy, prompt, rejected).item()
            margins.append(lp_w - lp_l)
    return float(np.mean(margins)) if margins else 0.0


def train_dpo(
    policy: TinyGPT,
    reference: TinyGPT,
    tok: InstructionTokenizer,
    triples: Sequence[Dict[str, str]],
    cfg: DPOConfig,
    log: Callable[[str], None] = print,
) -> DPOReport:
    report = DPOReport()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    # 预先保存参考模型的对数概率快照；这些值始终不变。
    ref_logps: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for tri in triples:
            prompt = tok.encode_prompt(tri["prompt"])
            chosen = tok.encode_completion(tri["chosen"])
            rejected = tok.encode_completion(tri["rejected"])
            lp_w_ref = sequence_log_prob(reference, prompt, chosen).detach()
            lp_l_ref = sequence_log_prob(reference, prompt, rejected).detach()
            ref_logps.append((lp_w_ref, lp_l_ref))
    report.initial_margin = evaluate_margins(policy, reference, tok, triples)
    for ep in range(1, cfg.epochs + 1):
        policy.train()
        total_loss = 0.0
        total_margin = 0.0
        for tri, (lp_w_ref, lp_l_ref) in zip(triples, ref_logps):
            prompt = tok.encode_prompt(tri["prompt"])
            chosen = tok.encode_completion(tri["chosen"])
            rejected = tok.encode_completion(tri["rejected"])
            lp_w_pol = sequence_log_prob(policy, prompt, chosen)
            lp_l_pol = sequence_log_prob(policy, prompt, rejected)
            loss, margin = dpo_loss(lp_w_pol, lp_l_pol, lp_w_ref, lp_l_ref, beta=cfg.beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            total_margin += float(margin.item())
        report.losses.append(total_loss / max(len(triples), 1))
        report.margins.append(total_margin / max(len(triples), 1))
        log(f"  epoch {ep:>3d}: loss={report.losses[-1]:.4f}  margin={report.margins[-1]:+.4f}")
    report.final_margin = evaluate_margins(policy, reference, tok, triples)
    return report


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def run_demo(cfg: Optional[DPOConfig] = None) -> int:
    cfg = cfg or DPOConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tok = InstructionTokenizer()
    triples = make_preferences()

    print("从零实现 DPO 演示")
    print(f"triples={len(triples)} beta={cfg.beta} lr={cfg.lr} epochs={cfg.epochs}")
    print("")

    reference, policy = build_models(cfg)

    print(f"[预热] 在 chosen 补全上短暂预训练（{cfg.warmup_epochs} 轮）……")
    # build_models() 会冻结参考模型，避免 DPO 循环意外更新它。仅在预热阶段
    # 解冻，正式训练前再重新冻结。
    for p in reference.parameters():
        p.requires_grad = True
    reference.train()
    warm_losses = warmup_pretrain(
        reference,
        tok,
        triples,
        epochs=cfg.warmup_epochs,
        seed=cfg.seed,
    )
    # 将预热后的权重复制到策略模型，并重新冻结参考模型。
    policy.load_state_dict(reference.state_dict())
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    print(f"         预热最终 loss = {warm_losses[-1]:.4f}")

    initial = evaluate_margins(policy, reference, tok, triples)
    print(f"         初始 chosen-rejected margin = {initial:+.4f}")
    print("")

    print("[DPO 训练]")
    report = train_dpo(policy, reference, tok, triples, cfg)

    print("")
    print("[训练后的逐三元组间隔]")
    print_margin_table(margin_table(policy, tok, triples))

    print("")
    print(f"最终 margin = {report.final_margin:+.4f}  (初始值 {report.initial_margin:+.4f})")
    print(f"最终 loss   = {report.losses[-1]:.4f}  (第 1 轮 loss {report.losses[0]:.4f})")

    # 合理性检查：训练应提高间隔。
    if report.final_margin <= report.initial_margin:
        print("错误：训练未增大 chosen-rejected 间隔", file=sys.stderr)
        return 1
    # 同时 loss 应下降。
    if report.losses[-1] >= report.losses[0]:
        print("错误：训练未使损失随轮次下降", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
