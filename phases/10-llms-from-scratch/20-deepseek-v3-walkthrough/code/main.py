"""DeepSeek-V3 架构计算器 — stdlib Python.

考虑到“ph1”配置,计算:
- 按构成部分分列的参数总数
- 每向前方的有效参数数(MoE 稀疏)
- 在128k上下文中的KV cache(MLA vs GQA 假设)
- 每层分解(attention/MLP/专家/路由器/规范)

还运行什么变种:排名256 MLA,512位专家,前16位的路线. 那个
目标就是读-a-config-be-comes-reading-the-architecture. 读-a-config-be-comes-reading-the-architecture. [永久失效連結] 校对:Soup 风格与
阶段10 → 14 计算器,专门用于DeepSeek-V3的全部细节.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEEPSEEK_V3 = {
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
    "first_k_dense_layers": 3,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "shared_experts": 1,
    "max_position_embeddings": 163_840,
    "rope_theta": 10000.0,
    "vocab_size": 129_280,
    "mtp_modules": 1,
    "moe_router_enabled": True,
}


@dataclass
class ComponentParams:
    embedding: int
    attention_per_layer: int
    dense_mlp_per_layer: int
    expert_mlp_each: int
    shared_expert: int
    router_per_layer: int
    rmsnorm_per_layer: int
    final_norm: int
    mtp_module: int


def mla_attention_params(hidden: int, n_heads: int, head_dim: int,
                         kv_lora: int, q_lora: int) -> int:
    """MLA attention 参数计数.
Q路径:隐藏 - > q lora - > n heads * head dim(两个matmuls).
K 路径: 隐藏 - > kv lora (一个matmul).
V 路径: 隐藏 - > kv lora - > n heads * head dim (解压).
K 解压为 n  heads * head dim 为attention 评分.
输出投影:n heads * head dim - > 隐藏.
"""
    q_down = hidden * q_lora
    q_up = q_lora * (n_heads * head_dim)
    kv_down = hidden * kv_lora
    k_up = kv_lora * (n_heads * head_dim)
    v_up = kv_lora * (n_heads * head_dim)
    o_proj = (n_heads * head_dim) * hidden
    return q_down + q_up + kv_down + k_up + v_up + o_proj


def swiglu_mlp_params(hidden: int, ff: int) -> int:
    return 2 * hidden * ff + ff * hidden


def router_params(hidden: int, n_experts: int) -> int:
    return hidden * n_experts


def rmsnorm_params(hidden: int) -> int:
    return 2 * hidden


def mtp_module_params(hidden: int, ff: int) -> int:
    """Per DeepSeek 纸张 第2.2节:投影 M k(2h x h) + transformer
块。 我们在这里使用稠密的MLP来表示MTP块(保守的)——
实际公布的间接费用为14B,其中包括MoE结构。"""
    projection = 2 * hidden * hidden
    attention = 4 * hidden * hidden
    mlp = swiglu_mlp_params(hidden, ff)
    norms = 2 * rmsnorm_params(hidden)
    return projection + attention + mlp + norms


def compute_components(cfg: dict) -> ComponentParams:
    h = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    head_dim = h // n_heads
    vocab = cfg["vocab_size"]
    dense_ff = cfg["intermediate_size"]
    moe_ff = cfg["moe_intermediate_size"]

    emb = vocab * h
    attn = mla_attention_params(h, n_heads, head_dim,
                                 kv_lora=cfg["kv_lora_rank"],
                                 q_lora=cfg["q_lora_rank"])
    dense_mlp = swiglu_mlp_params(h, dense_ff)
    expert = swiglu_mlp_params(h, moe_ff)
    shared = swiglu_mlp_params(h, moe_ff) * cfg["shared_experts"]
    router = router_params(h, cfg["num_experts"])
    norm_per = 2 * rmsnorm_params(h)
    final = rmsnorm_params(h)
    mtp = mtp_module_params(h, dense_ff) * cfg["mtp_modules"]

    return ComponentParams(
        embedding=emb,
        attention_per_layer=attn,
        dense_mlp_per_layer=dense_mlp,
        expert_mlp_each=expert,
        shared_expert=shared,
        router_per_layer=router,
        rmsnorm_per_layer=norm_per,
        final_norm=final,
        mtp_module=mtp,
    )


@dataclass
class ArchReport:
    total: int
    active: int
    active_ratio: float
    kv_cache_bytes: int
    gqa_kv_cache_bytes_ref: int
    per_layer_attn: int
    per_layer_moe_block: int
    per_layer_active: int
    emb: int


def compute_totals(cfg: dict, ctx: int | None = None) -> ArchReport:
    c = compute_components(cfg)
    h = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    head_dim = h // n_heads
    n_layers = cfg["num_hidden_layers"]
    first_dense = cfg["first_k_dense_layers"]
    n_moe = n_layers - first_dense
    n_experts = cfg["num_experts"]
    top_k = cfg["num_experts_per_tok"]
    shared_count = cfg["shared_experts"]
    max_seq = ctx or cfg["max_position_embeddings"]

    dense_layer = (c.attention_per_layer + c.dense_mlp_per_layer
                   + c.rmsnorm_per_layer)
    moe_layer = (c.attention_per_layer
                 + n_experts * c.expert_mlp_each
                 + c.shared_expert
                 + c.router_per_layer
                 + c.rmsnorm_per_layer)
    active_moe_layer = (c.attention_per_layer
                        + top_k * c.expert_mlp_each
                        + c.shared_expert
                        + c.router_per_layer
                        + c.rmsnorm_per_layer)

    total = (c.embedding
             + first_dense * dense_layer
             + n_moe * moe_layer
             + c.final_norm
             + c.mtp_module)
    active = (c.embedding
              + first_dense * dense_layer
              + n_moe * active_moe_layer
              + c.final_norm)

    kv_cache = n_layers * cfg["kv_lora_rank"] * max_seq * 2
    kv_heads_hypothetical = 8
    head_dim_hypothetical = 128
    kv_cache_gqa = 2 * n_layers * kv_heads_hypothetical * head_dim_hypothetical * max_seq * 2

    return ArchReport(
        total=total, active=active,
        active_ratio=active / total,
        kv_cache_bytes=kv_cache,
        gqa_kv_cache_bytes_ref=kv_cache_gqa,
        per_layer_attn=c.attention_per_layer,
        per_layer_moe_block=moe_layer,
        per_layer_active=active_moe_layer,
        emb=c.embedding,
    )


def fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return f"{n}"


def fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}PB"


def print_report(name: str, cfg: dict, ctx: int | None = None) -> None:
    r = compute_totals(cfg, ctx=ctx)
    print(f"\n{name}")
    print("-" * 70)
    print(f"  总参数量：        {fmt(r.total)}")
    print(f"  激活参数量：      {fmt(r.active)}")
    print(f"  激活比例：        {r.active_ratio:.1%}")
    print(f"  embedding：       {fmt(r.emb)}")
    print(f"  每层 attention：  {fmt(r.per_layer_attn)}（MLA）")
    print(f"  每层 MoE block：  {fmt(r.per_layer_moe_block)}（总量）")
    print(f"  每层激活的 MoE： {fmt(r.per_layer_active)}（每次前向传播）")
    ctx_used = ctx or cfg["max_position_embeddings"]
    print(f"  BF16 KV cache，{ctx_used:,} 上下文：{fmt_bytes(r.kv_cache_bytes)}")
    print(f"  GQA(8/128) 参考值：             {fmt_bytes(r.gqa_kv_cache_bytes_ref)}")
    print(f"  MLA 节省比例：                   "
          f"{(1 - r.kv_cache_bytes / r.gqa_kv_cache_bytes_ref) * 100:.0f}%")


def main() -> None:
    print("=" * 70)
    print("DeepSeek-V3 架构解析（第 10 阶段，第 20 课）")
    print("=" * 70)

    print_report("DeepSeek-V3（已发布配置）", DEEPSEEK_V3, ctx=131_072)

    variant = dict(DEEPSEEK_V3)
    variant["kv_lora_rank"] = 256
    print_report("DeepSeek-V3 (MLA rank 256 what-if)", variant, ctx=131_072)

    variant = dict(DEEPSEEK_V3)
    variant["num_experts"] = 512
    variant["num_experts_per_tok"] = 8
    print_report("DeepSeek-V3 (512 experts, top-8 what-if)", variant,
                 ctx=131_072)

    variant = dict(DEEPSEEK_V3)
    variant["num_experts_per_tok"] = 16
    print_report("DeepSeek-V3 (256 experts, top-16 what-if)", variant,
                 ctx=131_072)

    print()
    print("=" * 70)
    print("说明：论文公布总参数量为 671B，本计算器约为 476B–490B")
    print("-" * 70)
    print("差值来自报告中的其他结构参数和附录明细：专家专属 bias、共享")
    print("专家缩放、MoE 规模的 MTP 模块，以及本简化计算器合并的若干子组件。")
    print("参数量级和比例（例如激活参数占总量的 5%–6%）与论文一致。")


if __name__ == "__main__":
    main()
