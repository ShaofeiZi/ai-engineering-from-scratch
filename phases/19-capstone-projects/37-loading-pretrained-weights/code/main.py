"""从 safetensors 加载 GPT-2 风格的预训练权重到第 35 课架构。

使用 `safetensors` 库读取 safetensors 文件，把预训练参数名（`wte`、`wpe`、
`h.N.attn.c_attn` 等）映射到本地名称（`tok_embed`、`pos_embed`、
`blocks.N.attn.qkv` 等），检查形状，转置公开 GPT-2 检查点采用的 conv1d 风格
权重布局，并在 `torch.no_grad()` 下赋值。LM 头是 `tok_embed` 的权重绑定别名，
因此不在文件中。

为保持演示离线，`make_stub_safetensors` 会在首次运行时生成严格遵循预训练命名
约定的夹具。把夹具替换为真实 GPT-2 文件后，加载器无需修改即可工作。

运行：python3 code/main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
STUB_PATH = OUTPUTS / "gpt2-stub.safetensors"


@dataclass
class ModelConfig:
    """与第 35 课参考模型对齐的配置；桩模型使用更小的 d_model。"""

    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_expansion: int = 4
    dropout: float = 0.0
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        if seq > self.cfg.context_length:
            raise ValueError(
                f"sequence length {seq} exceeds context_length={self.cfg.context_length}"
            )
        tok = self.tok_embed(tokens)
        pos = self.pos_embed(self.position_ids[:seq])
        x = self.embed_dropout(tok + pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_ln(x))


@dataclass
class LoadReport:
    """加载结果；打印它即可判断加载是否成功。"""

    loaded: list[tuple[str, str, tuple[int, ...]]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)} "
            f"missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} "
            f"shape_mismatch={len(self.shape_mismatch)}"
        )

    def ok(self) -> bool:
        return not self.missing and not self.shape_mismatch


# 公开 GPT-2 检查点中以转置形式存储的名称。
# 公开格式使用 TensorFlow conv1d 布局；nn.Linear 需要 (out, in)。
CONV1D_SUFFIXES = ("c_attn.weight", "c_proj.weight", "c_fc.weight")


def make_pretrained_to_local(num_layers: int) -> dict[str, str]:
    """返回含 `num_layers` 个块的模型所用的完整预训练名称到本地名称映射。"""
    mapping: dict[str, str] = {
        "wte.weight": "tok_embed.weight",
        "wpe.weight": "pos_embed.weight",
        "ln_f.weight": "final_ln.scale",
        "ln_f.bias": "final_ln.shift",
    }
    for layer in range(num_layers):
        prefix_src = f"h.{layer}"
        prefix_dst = f"blocks.{layer}"
        mapping[f"{prefix_src}.ln_1.weight"] = f"{prefix_dst}.ln1.scale"
        mapping[f"{prefix_src}.ln_1.bias"] = f"{prefix_dst}.ln1.shift"
        mapping[f"{prefix_src}.ln_2.weight"] = f"{prefix_dst}.ln2.scale"
        mapping[f"{prefix_src}.ln_2.bias"] = f"{prefix_dst}.ln2.shift"
        mapping[f"{prefix_src}.attn.c_attn.weight"] = f"{prefix_dst}.attn.qkv.weight"
        mapping[f"{prefix_src}.attn.c_attn.bias"] = f"{prefix_dst}.attn.qkv.bias"
        mapping[f"{prefix_src}.attn.c_proj.weight"] = f"{prefix_dst}.attn.out_proj.weight"
        mapping[f"{prefix_src}.attn.c_proj.bias"] = f"{prefix_dst}.attn.out_proj.bias"
        mapping[f"{prefix_src}.mlp.c_fc.weight"] = f"{prefix_dst}.mlp.fc1.weight"
        mapping[f"{prefix_src}.mlp.c_fc.bias"] = f"{prefix_dst}.mlp.fc1.bias"
        mapping[f"{prefix_src}.mlp.c_proj.weight"] = f"{prefix_dst}.mlp.fc2.weight"
        mapping[f"{prefix_src}.mlp.c_proj.bias"] = f"{prefix_dst}.mlp.fc2.bias"
    return mapping


def _needs_transpose(pretrained_name: str) -> bool:
    return any(pretrained_name.endswith(suffix) for suffix in CONV1D_SUFFIXES)


def load_safetensors(model: GPTModel, path: Path, verbose: bool = True) -> LoadReport:
    """把权重加载到模型中；形状不匹配时拒绝赋值，并返回报告。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 safetensors 文件：{path}")

    mapping = make_pretrained_to_local(model.cfg.num_layers)
    local_params = dict(model.named_parameters())
    report = LoadReport()

    seen_local: set[str] = set()
    pending: list[tuple[str, str, torch.Tensor]] = []
    with safe_open(str(path), framework="pt") as reader:
        pretrained_names = list(reader.keys())
        for src_name in pretrained_names:
            local_name = mapping.get(src_name)
            if local_name is None:
                report.unexpected.append(src_name)
                if verbose:
                    print(f"  [跳过] {src_name}（无映射）")
                continue
            if local_name not in local_params:
                report.unexpected.append(src_name)
                if verbose:
                    print(f"  [跳过] {src_name} -> {local_name}（无此参数）")
                continue

            tensor = reader.get_tensor(src_name)
            if _needs_transpose(src_name):
                tensor = tensor.t().contiguous()

            target = local_params[local_name]
            if tuple(tensor.shape) != tuple(target.shape):
                report.shape_mismatch.append(
                    (src_name, tuple(tensor.shape), tuple(target.shape))
                )
                if verbose:
                    print(
                        f"  [错误] {src_name} -> {local_name} "
                        f"src_shape={tuple(tensor.shape)} dst_shape={tuple(target.shape)}"
                    )
                continue

            pending.append((src_name, local_name, tensor))

    if report.shape_mismatch:
        expected = set(local_params.keys())
        for name in sorted(expected - seen_local):
            report.missing.append(name)
        return report

    with torch.no_grad():
        for src_name, local_name, tensor in pending:
            target = local_params[local_name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            seen_local.add(local_name)
            report.loaded.append((src_name, local_name, tuple(tensor.shape)))
            if verbose:
                print(f"  [成功] {src_name} -> {local_name} shape={tuple(tensor.shape)}")

    if model.cfg.weight_tying:
        if model.lm_head.weight.data_ptr() != model.tok_embed.weight.data_ptr():
            model.lm_head.weight = model.tok_embed.weight
        seen_local.add("lm_head.weight")

    expected = set(local_params.keys())
    for name in sorted(expected - seen_local):
        report.missing.append(name)

    return report


def make_stub_safetensors(path: Path, cfg: ModelConfig, seed: int = 42) -> None:
    """生成采用预训练命名约定的夹具文件。

    张量随机生成，但可通过 `seed` 复现。形状与 `cfg` 对应的真实 GPT-2 检查点
    一致，包括 `c_attn`、`c_proj`、`c_fc` 的 conv1d 转置。
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    tensors: dict[str, torch.Tensor] = {}
    tensors["wte.weight"] = randn(cfg.vocab_size, cfg.d_model) * 0.02
    tensors["wpe.weight"] = randn(cfg.context_length, cfg.d_model) * 0.02
    tensors["ln_f.weight"] = torch.ones(cfg.d_model)
    tensors["ln_f.bias"] = torch.zeros(cfg.d_model)

    hidden = cfg.mlp_expansion * cfg.d_model
    for layer in range(cfg.num_layers):
        tensors[f"h.{layer}.ln_1.weight"] = torch.ones(cfg.d_model)
        tensors[f"h.{layer}.ln_1.bias"] = torch.zeros(cfg.d_model)
        tensors[f"h.{layer}.ln_2.weight"] = torch.ones(cfg.d_model)
        tensors[f"h.{layer}.ln_2.bias"] = torch.zeros(cfg.d_model)
        tensors[f"h.{layer}.attn.c_attn.weight"] = (
            randn(3 * cfg.d_model, cfg.d_model).t().contiguous() * 0.02
        )
        tensors[f"h.{layer}.attn.c_attn.bias"] = torch.zeros(3 * cfg.d_model)
        tensors[f"h.{layer}.attn.c_proj.weight"] = (
            randn(cfg.d_model, cfg.d_model).t().contiguous() * 0.02
        )
        tensors[f"h.{layer}.attn.c_proj.bias"] = torch.zeros(cfg.d_model)
        tensors[f"h.{layer}.mlp.c_fc.weight"] = (
            randn(hidden, cfg.d_model).t().contiguous() * 0.02
        )
        tensors[f"h.{layer}.mlp.c_fc.bias"] = torch.zeros(hidden)
        tensors[f"h.{layer}.mlp.c_proj.weight"] = (
            randn(cfg.d_model, hidden).t().contiguous() * 0.02
        )
        tensors[f"h.{layer}.mlp.c_proj.bias"] = torch.zeros(cfg.d_model)

    save_file(tensors, str(path))


@torch.no_grad()
def quick_generate(model: GPTModel, prompt: torch.Tensor, n: int, seed: int = 0) -> list[int]:
    torch.manual_seed(seed)
    model.eval()
    tokens = prompt.clone()
    for _ in range(n):
        window = tokens[:, -model.cfg.context_length :]
        logits = model(window)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens.tolist()[0]


def _state_fingerprint(model: GPTModel) -> float:
    """所有参数 L2 范数之和；加载后会变化的粗粒度指纹。"""
    return float(sum(p.detach().norm().item() for p in model.parameters()))


def demo() -> None:
    torch.manual_seed(0)

    cfg = ModelConfig(
        vocab_size=256,
        context_length=64,
        d_model=192,
        num_heads=6,
        num_layers=4,
        mlp_expansion=4,
        dropout=0.0,
    )
    print(f"模型配置                : vocab={cfg.vocab_size} d_model={cfg.d_model} layers={cfg.num_layers}")

    print(f"\n正在写入桩夹具：{STUB_PATH}")
    make_stub_safetensors(STUB_PATH, cfg, seed=42)
    print(f"  文件大小              : {STUB_PATH.stat().st_size:,} 字节")

    print("\n正在构建新模型（随机初始化）……")
    model = GPTModel(cfg)
    before_fp = _state_fingerprint(model)
    prompt = torch.tensor([[7, 11, 13, 17]], dtype=torch.long)
    before_tokens = quick_generate(model, prompt, n=8, seed=0)
    print(f"  指纹                  : {before_fp:.4f}")
    print(f"  样本（随机初始化）    : {before_tokens}")

    print("\n正在加载桩……")
    report = load_safetensors(model, STUB_PATH, verbose=False)
    print(f"  报告                  : {report.summary()}")
    if not report.ok():
        print("  警告：加载未完全成功")
    else:
        print("  加载成功")

    after_fp = _state_fingerprint(model)
    after_tokens = quick_generate(model, prompt, n=8, seed=0)
    print(f"  加载后指纹            : {after_fp:.4f}")
    print(f"  样本（已加载）        : {after_tokens}")

    assert before_fp != after_fp, "加载后指纹应发生变化"
    assert before_tokens != after_tokens, "加载后样本应发生变化"

    print("\n加载后的权重绑定检查：")
    tied = model.lm_head.weight.data_ptr() == model.tok_embed.weight.data_ptr()
    print(f"  lm_head 已绑定到 tok_embed：{tied}")
    assert tied

    print("\n形状不匹配路径：注入错误张量并重新加载……")
    bad_path = OUTPUTS / "gpt2-bad.safetensors"
    bad_tensors = {}
    with safe_open(str(STUB_PATH), framework="pt") as reader:
        for name in reader.keys():
            bad_tensors[name] = reader.get_tensor(name)
    bad_tensors["wte.weight"] = torch.randn(cfg.vocab_size, cfg.d_model + 1)
    save_file(bad_tensors, str(bad_path))
    bad_model = GPTModel(cfg)
    bad_report = load_safetensors(bad_model, bad_path, verbose=False)
    print(f"  错误报告              : {bad_report.summary()}")
    assert bad_report.shape_mismatch, "预期至少出现一个形状不匹配"
    print(f"  首个不匹配项          : {bad_report.shape_mismatch[0]}")

    bad_path.unlink()
    print("\n预训练权重加载检查通过。")


if __name__ == "__main__":
    demo()
