"""
通过监督微调（SFT）进行指令微调。

See: phases/19-capstone-projects/39-instruction-tuning-sft/docs/en.md

构建内容：
  - byte-level tokenizer with INST / RESP / PAD specials
  - SFT dataset over 200 instruction-response pairs
  - 用 -100 掩蔽指令和填充 token 的 collate 函数
  - TinyGPT (decoder-only transformer) body and LM head
  - SFT 循环、贪心生成器、精确匹配指标
  - 训练 20 轮并打印各类别精确匹配率的 run_demo。

训练后的模型在留出集上超过 0.0 的随机基线时，以状态码 0 退出。
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
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class InstructionTokenizer:
    """带 INST、RESP、PAD 特殊 token 的字节级 tokenizer。"""

    INST_ID = 256
    RESP_ID = 257
    PAD_ID = 258
    VOCAB = 260
    IGNORE_INDEX = -100

    def encode_pair(self, instruction: str, response: str, max_len: int) -> Tuple[List[int], int]:
        """返回 (token_ids, response_start_index)。必要时截断到 max_len，
        但始终保留 RESP 标记和至少一个响应 token，避免 SFT 整理产生全掩蔽标签。"""
        if max_len < 3:
            raise ValueError("max_len must be >= 3 to fit INST, RESP, and one response token")
        inst_bytes = list(instruction.encode("utf-8", errors="ignore"))
        resp_bytes = list(response.encode("utf-8", errors="ignore"))
        # 为 2 个控制 token 和至少 1 个响应字节预留空间。
        max_inst = max_len - 3
        inst_bytes = inst_bytes[:max_inst]
        ids = [self.INST_ID] + inst_bytes + [self.RESP_ID]
        resp_start = len(ids)
        ids.extend(resp_bytes[: max_len - len(ids)])
        return ids, resp_start

    def encode_prefix(self, instruction: str, max_len: int) -> List[int]:
        """只编码用于生成的指令前缀。始终保留 RESP 标记，使模型看到与训练时
        相同的边界。"""
        if max_len < 2:
            raise ValueError("max_len must be >= 2 to fit INST and RESP")
        inst_bytes = list(instruction.encode("utf-8", errors="ignore"))[: max_len - 2]
        ids = [self.INST_ID] + inst_bytes + [self.RESP_ID]
        return ids

    def decode_response(self, ids: Sequence[int]) -> str:
        """解码生成的响应，并丢弃特殊 token。"""
        chunk = bytes(i for i in ids if i < 256)
        return chunk.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 微型 GPT
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, max_len: int):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must divide by heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, key_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal_mask[:T, :T].view(1, 1, T, T)
        att = att.masked_fill(~causal, float("-inf"))
        if key_pad_mask is not None:
            # key_pad_mask：B x T，真实 token 为 1，pad 为 0。
            km = key_pad_mask.view(B, 1, 1, T).to(torch.bool)
            att = att.masked_fill(~km, float("-inf"))
        weights = F.softmax(att, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
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

    def forward(self, x: torch.Tensor, key_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_pad_mask)
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + h


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, hidden: int, heads: int, depth: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, max_len) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.max_len = max_len

    def forward(self, ids: torch.Tensor, key_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(positions)
        for blk in self.blocks:
            x = blk(x, key_pad_mask)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# 指令夹具
# ---------------------------------------------------------------------------


CAPITALS = [
    ("France", "Paris"),
    ("Spain", "Madrid"),
    ("Italy", "Rome"),
    ("Japan", "Tokyo"),
    ("Egypt", "Cairo"),
    ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Kenya", "Nairobi"),
    ("Sweden", "Stockholm"),
]

ARITHMETIC = [
    (2, 3, "+"),
    (5, 4, "*"),
    (9, 7, "-"),
    (12, 4, "/"),
    (7, 6, "+"),
    (11, 3, "-"),
    (8, 8, "+"),
    (15, 5, "/"),
    (6, 9, "*"),
    (20, 4, "/"),
]

LIST_NAMES = [
    ("colors", ["red", "green", "blue"]),
    ("planets", ["mercury", "venus", "earth"]),
    ("vowels", ["a", "e", "i"]),
    ("seasons", ["spring", "summer", "autumn"]),
    ("metals", ["iron", "gold", "silver"]),
    ("primes", ["2", "3", "5"]),
    ("oceans", ["pacific", "atlantic", "indian"]),
    ("languages", ["python", "rust", "go"]),
    ("instruments", ["guitar", "piano", "drums"]),
    ("fruits", ["apple", "banana", "cherry"]),
]

SUMMARIES = [
    ("the sun rises in the east and sets in the west.", "the sun moves east to west."),
    ("water boils at one hundred degrees celsius at sea level.", "water boils at 100c."),
    ("the moon orbits the earth roughly every 27 days.", "the moon orbits earth in 27 days."),
    ("light travels at about 300000 km per second.", "light moves at 300000 km/s."),
    ("the human body has 206 bones in adulthood.", "adults have 206 bones."),
    ("plants make food using sunlight, water, and co2.", "plants use sun, water, and co2."),
    ("dna stores genetic information in cells.", "dna stores genes."),
    ("the heart pumps blood through the body.", "the heart pumps blood."),
    ("electricity flows through conductors easily.", "current flows in conductors."),
    ("bees pollinate flowers as they collect nectar.", "bees pollinate flowers."),
]

CODE = [
    ("print hello world", "print('hello world')"),
    ("print the number 42", "print(42)"),
    ("sort a list named items ascending", "items.sort()"),
    ("reverse a list named items", "items.reverse()"),
    ("get the length of items", "len(items)"),
    ("open a file in read mode", "open('f.txt', 'r')"),
    ("get the keys of a dict d", "d.keys()"),
    ("return the absolute value of x", "abs(x)"),
    ("round x to two decimals", "round(x, 2)"),
    ("convert x to a string", "str(x)"),
]

DEFINITIONS = [
    ("variable", "a name bound to a value in memory."),
    ("function", "a reusable block of code that takes inputs and returns an output."),
    ("loop", "a control structure that repeats a block of code."),
    ("class", "a template for creating objects with shared structure."),
    ("list", "an ordered, mutable sequence of values."),
    ("dict", "a mapping from keys to values."),
    ("string", "a sequence of characters."),
    ("integer", "a whole number without a fractional part."),
    ("float", "a number with a fractional part stored in binary."),
    ("module", "a file of python code that can be imported."),
]


def _arithmetic_response(a: int, b: int, op: str) -> str:
    if op == "+":
        return f"{a} + {b} = {a + b}"
    if op == "-":
        return f"{a} - {b} = {a - b}"
    if op == "*":
        return f"{a} * {b} = {a * b}"
    if op == "/":
        return f"{a} / {b} = {a // b}"
    raise ValueError(f"unknown op {op}")


def make_dataset(seed: int = 0) -> Tuple[List[Dict[str, str]], List[str]]:
    """返回 (pairs, categories)；每一对都包含 instruction 和 response。"""
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    categories: List[str] = []

    # 首都（40 对：10 个基础样例 x 4 个模板）
    cap_templates = [
        "What is the capital of {country}?",
        "Name the capital city of {country}.",
        "Capital of {country}?",
        "Tell me the capital of {country}.",
    ]
    cap_resp_templates = [
        "the capital of {country} is {city}.",
        "{city} is the capital of {country}.",
        "{city}.",
        "the answer is {city}.",
    ]
    for country, city in CAPITALS:
        for i, t in enumerate(cap_templates):
            pairs.append(
                {
                    "instruction": t.format(country=country),
                    "response": cap_resp_templates[i].format(country=country, city=city),
                }
            )
            categories.append("capitals")

    # 算术（30 对：10 个基础样例 x 3 个模板）
    arith_templates = [
        "Compute {a} {op} {b}.",
        "What is {a} {op} {b}?",
        "{a} {op} {b}?",
    ]
    for a, b, op in ARITHMETIC:
        for t in arith_templates:
            pairs.append(
                {
                    "instruction": t.format(a=a, b=b, op=op),
                    "response": _arithmetic_response(a, b, op),
                }
            )
            categories.append("arithmetic")

    # 列表（30 对：10 个基础样例 x 3 个模板）
    list_templates = [
        "List three {name}.",
        "Give me three {name}.",
        "Name three {name}.",
    ]
    for name, items in LIST_NAMES:
        for t in list_templates:
            pairs.append(
                {
                    "instruction": t.format(name=name),
                    "response": ", ".join(items) + ".",
                }
            )
            categories.append("lists")

    # 摘要（30 对：10 个基础样例 x 3 个模板）
    sum_templates = [
        "Summarise: {text}",
        "One-sentence summary of: {text}",
        "TLDR: {text}",
    ]
    for text, summary in SUMMARIES:
        for t in sum_templates:
            pairs.append({"instruction": t.format(text=text), "response": summary})
            categories.append("summaries")

    # 代码（30 对：10 个基础样例 x 3 个模板）
    code_templates = [
        "Write python code to {task}.",
        "Python: {task}.",
        "Code that will {task}.",
    ]
    for task, code in CODE:
        for t in code_templates:
            pairs.append({"instruction": t.format(task=task), "response": code})
            categories.append("code")

    # 定义（40 对：10 个基础样例 x 4 个模板）
    def_templates = [
        "Define {term}.",
        "What is a {term}?",
        "Explain the term {term}.",
        "{term}: what is it?",
    ]
    for term, defn in DEFINITIONS:
        for t in def_templates:
            pairs.append({"instruction": t.format(term=term), "response": defn})
            categories.append("definitions")

    # 总计 = 40 + 30 + 30 + 30 + 30 + 40 = 200。打乱后返回。
    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[i] for i in order], [categories[i] for i in order]


def split_dataset(
    pairs: Sequence[Dict[str, str]],
    cats: Sequence[str],
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[List[Dict[str, str]], List[str], List[Dict[str, str]], List[str]]:
    """按类别分层拆分。"""
    rng = random.Random(seed)
    by_cat: Dict[str, List[int]] = {}
    for i, c in enumerate(cats):
        by_cat.setdefault(c, []).append(i)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for c, idxs in by_cat.items():
        idxs_copy = list(idxs)
        rng.shuffle(idxs_copy)
        cut = max(1, int(len(idxs_copy) * (1.0 - test_frac)))
        train_idx.extend(idxs_copy[:cut])
        test_idx.extend(idxs_copy[cut:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return (
        [pairs[i] for i in train_idx],
        [cats[i] for i in train_idx],
        [pairs[i] for i in test_idx],
        [cats[i] for i in test_idx],
    )


# ---------------------------------------------------------------------------
# 数据集与整理函数
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Dict[str, str]],
        tok: InstructionTokenizer,
        max_len: int,
    ):
        self.pairs = list(pairs)
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        pair = self.pairs[idx]
        ids, resp_start = self.tok.encode_pair(pair["instruction"], pair["response"], self.max_len)
        return ids, resp_start


def sft_collate(
    batch: Sequence[Tuple[List[int], int]],
    pad_id: int = InstructionTokenizer.PAD_ID,
    ignore_index: int = InstructionTokenizer.IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """填充到批次内最长长度，并用 -100 掩蔽指令和填充部分来构建标签。"""
    max_t = max(len(x[0]) for x in batch)
    input_ids: List[List[int]] = []
    labels: List[List[int]] = []
    attn_mask: List[List[int]] = []
    for ids, resp_start in batch:
        seq_len = len(ids)
        pad = [pad_id] * (max_t - seq_len)
        padded = list(ids) + pad
        # 位置 i 的标签应为 input_ids[i+1]；随后掩蔽与指令和填充对应的位置。
        lbl: List[int] = list(padded)
        for i in range(len(lbl)):
            if i < resp_start:
                # 指令或边界 token：不训练对这些 token 的预测。
                lbl[i] = ignore_index
            elif i >= seq_len:
                # 填充区域。
                lbl[i] = ignore_index
        am = [1] * seq_len + [0] * (max_t - seq_len)
        input_ids.append(padded)
        labels.append(lbl)
        attn_mask.append(am)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
    )


def shifted_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = InstructionTokenizer.IGNORE_INDEX
) -> torch.Tensor:
    """标准因果 LM 损失：预测下一 token，并忽略已掩蔽位置。"""
    # 位置 i 预测 labels 中的位置 i+1。
    pred = logits[:, :-1, :].contiguous()
    target = labels[:, 1:].contiguous()
    return F.cross_entropy(
        pred.view(-1, pred.size(-1)),
        target.view(-1),
        ignore_index=ignore_index,
    )


# ---------------------------------------------------------------------------
# 训练与生成
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    vocab: int = InstructionTokenizer.VOCAB
    hidden: int = 96
    heads: int = 4
    depth: int = 2
    max_len: int = 96
    batch_size: int = 16
    epochs: int = 20
    lr: float = 5e-4
    seed: int = 0


def build_model(cfg: SFTConfig) -> TinyGPT:
    torch.manual_seed(cfg.seed)
    return TinyGPT(cfg.vocab, cfg.hidden, cfg.heads, cfg.depth, cfg.max_len)


@dataclass
class SFTReport:
    losses: List[float] = field(default_factory=list)
    eval_em: List[float] = field(default_factory=list)
    final_em: float = 0.0


def train_sft(
    model: TinyGPT,
    train_loader: DataLoader,
    cfg: SFTConfig,
    eval_pairs: Optional[Sequence[Dict[str, str]]] = None,
    tok: Optional[InstructionTokenizer] = None,
    eval_every: int = 5,
    log: Callable[[str], None] = print,
) -> SFTReport:
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    report = SFTReport()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for input_ids, labels, attn_mask in train_loader:
            logits = model(input_ids, key_pad_mask=attn_mask)
            loss = shifted_loss(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        report.losses.append(avg)
        if eval_pairs is not None and tok is not None and ep % eval_every == 0:
            em = exact_match_set(model, tok, eval_pairs, cfg.max_len)
            report.eval_em.append(em)
            log(f"  epoch {ep:>3d}: loss={avg:.4f}  EM={em:.3f}")
        elif ep % eval_every == 0:
            log(f"  epoch {ep:>3d}: loss={avg:.4f}")
    if eval_pairs is not None and tok is not None:
        report.final_em = exact_match_set(model, tok, eval_pairs, cfg.max_len)
    return report


@torch.no_grad()
def generate(
    model: TinyGPT,
    tok: InstructionTokenizer,
    instruction: str,
    max_len: int,
    temperature: float = 0.0,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> str:
    """执行贪心（temperature=0）或采样生成；遇到两个连续句末符号时停止。"""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)
    ids = tok.encode_prefix(instruction, max_len=max_len)
    sentence_ends = {ord("."), ord("!"), ord("?")}
    out_chars: List[int] = []
    for _ in range(max_new_tokens):
        if len(ids) >= max_len:
            break
        x = torch.tensor([ids], dtype=torch.long)
        attn_mask = torch.ones_like(x)
        logits = model(x, key_pad_mask=attn_mask)
        next_logits = logits[0, -1, :]
        if temperature <= 0.0:
            next_id = int(next_logits.argmax().item())
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        if next_id == tok.PAD_ID:
            break
        if next_id == tok.INST_ID or next_id == tok.RESP_ID:
            # 模型生成了控制 token，停止。
            break
        ids.append(next_id)
        out_chars.append(next_id)
        if (
            len(out_chars) >= 2
            and out_chars[-1] in sentence_ends
            and out_chars[-2] in sentence_ends
        ):
            break
    return tok.decode_response(out_chars)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    s = text.lower().strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def exact_match(pred: str, gold: str) -> int:
    return 1 if normalise(pred) == normalise(gold) else 0


@torch.no_grad()
def exact_match_set(
    model: TinyGPT,
    tok: InstructionTokenizer,
    pairs: Sequence[Dict[str, str]],
    max_len: int,
) -> float:
    if not pairs:
        return 0.0
    hits = 0
    for pair in pairs:
        pred = generate(model, tok, pair["instruction"], max_len=max_len)
        hits += exact_match(pred, pair["response"])
    return hits / len(pairs)


def per_category_em(
    model: TinyGPT,
    tok: InstructionTokenizer,
    pairs: Sequence[Dict[str, str]],
    cats: Sequence[str],
    max_len: int,
) -> Dict[str, float]:
    by_cat: Dict[str, List[int]] = {}
    for p, c in zip(pairs, cats):
        pred = generate(model, tok, p["instruction"], max_len=max_len)
        by_cat.setdefault(c, []).append(exact_match(pred, p["response"]))
    return {c: sum(vs) / max(len(vs), 1) for c, vs in by_cat.items()}


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def run_demo(cfg: Optional[SFTConfig] = None) -> int:
    cfg = cfg or SFTConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tok = InstructionTokenizer()
    pairs, cats = make_dataset(seed=cfg.seed)
    tr_pairs, tr_cats, te_pairs, te_cats = split_dataset(pairs, cats, test_frac=0.2, seed=cfg.seed)

    train_ds = SFTDataset(tr_pairs, tok, cfg.max_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate(b),
    )

    print("指令微调（SFT）演示")
    print(f"训练集={len(tr_pairs)} 测试集={len(te_pairs)} 最大长度={cfg.max_len}")
    print(f"类别：{sorted(set(cats))}")
    print("")

    model = build_model(cfg)
    initial_em = exact_match_set(model, tok, te_pairs, cfg.max_len)
    print(f"基线（未训练）EM = {initial_em:.3f}")
    print("")

    print("[训练]")
    report = train_sft(
        model,
        train_dl,
        cfg,
        eval_pairs=te_pairs,
        tok=tok,
        eval_every=5,
    )

    print("")
    print("[留出集各类别精确匹配率]")
    cat_em = per_category_em(model, tok, te_pairs, te_cats, cfg.max_len)
    for cat in sorted(cat_em):
        print(f"  {cat:>12s}: {cat_em[cat]:.3f}")

    print("")
    print("[生成样本]")
    for pair in te_pairs[:3]:
        pred = generate(model, tok, pair["instruction"], max_len=cfg.max_len)
        match = "匹配" if exact_match(pred, pair["response"]) else "未匹配"
        print(f"  [{match}] 指令：{pair['instruction']}")
        print(f"          标准答案：{pair['response']}")
        print(f"          预测结果：{pred}")

    print("")
    print(f"最终精确匹配率 = {report.final_em:.3f}（基线为 {initial_em:.3f}）")

    if report.final_em <= initial_em:
        print("错误：训练后的 EM 未超过未训练基线", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
