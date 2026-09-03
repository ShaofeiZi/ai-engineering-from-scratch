"""Jamba/ Mamba-3 内存计算器 — stdlib Python.

计算 KV cache, SSM 状态, 和一个区域的总 attention 层内存
混合构型:纯Transformer、Jamba 1:7、1:3、1:15和纯
(原始内容存档于2018-09-25). SSM. 在8k,64k,128k,256k上下文中打印比较.

数字是说明性的,而不是准确的生产记忆预算。 重点是
以显示混合比率为何重要,以及 "ph7 " 的256k-on-80GB索赔在何处
来者.
"""

from __future__ import annotations

from dataclasses import dataclass


BYTES_BF16 = 2
BYTES_FP8 = 1


@dataclass
class HybridConfig:
    name: str
    total_layers: int
    attn_layers: int
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    ssm_state_size: int


def kv_cache_bytes(cfg: HybridConfig, ctx: int, bytes_per_elem: int) -> int:
    return (2 * cfg.attn_layers * cfg.n_kv_heads * cfg.head_dim * ctx
            * bytes_per_elem)


def ssm_state_bytes(cfg: HybridConfig, bytes_per_elem: int) -> int:
    ssm_layers = cfg.total_layers - cfg.attn_layers
    return ssm_layers * cfg.hidden * cfg.ssm_state_size * bytes_per_elem


def fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.2f}{unit}"
        b /= 1024
    return f"{b:.2f}PB"


def main() -> None:
    print("=" * 74)
    print("Jamba 混合 SSM-Transformer 内存计算器（第 10 阶段，第 21 课）")
    print("=" * 74)
    print()

    configs = [
        HybridConfig(
            name="pure Transformer 32L",
            total_layers=32, attn_layers=32,
            hidden=4096, n_q_heads=32, n_kv_heads=32, head_dim=128,
            ssm_state_size=0,
        ),
        HybridConfig(
            name="pure Transformer 32L (GQA 8)",
            total_layers=32, attn_layers=32,
            hidden=4096, n_q_heads=32, n_kv_heads=8, head_dim=128,
            ssm_state_size=0,
        ),
        HybridConfig(
            name="Jamba 1:7 hybrid 32L",
            total_layers=32, attn_layers=4,
            hidden=4096, n_q_heads=32, n_kv_heads=32, head_dim=128,
            ssm_state_size=16,
        ),
        HybridConfig(
            name="Jamba 1:3 hybrid 32L",
            total_layers=32, attn_layers=8,
            hidden=4096, n_q_heads=32, n_kv_heads=32, head_dim=128,
            ssm_state_size=16,
        ),
        HybridConfig(
            name="Jamba 1:15 hybrid 32L",
            total_layers=32, attn_layers=2,
            hidden=4096, n_q_heads=32, n_kv_heads=32, head_dim=128,
            ssm_state_size=16,
        ),
        HybridConfig(
            name="pure Mamba 32L",
            total_layers=32, attn_layers=0,
            hidden=4096, n_q_heads=0, n_kv_heads=0, head_dim=128,
            ssm_state_size=16,
        ),
    ]

    contexts = [8_192, 65_536, 131_072, 262_144]

    print("-" * 74)
    print("BF16 存储（每个元素 2 字节）")
    print("-" * 74)
    header = "  " + "config".ljust(32)
    for ctx in contexts:
        header += f"{ctx // 1000}k".rjust(10)
    print(header)
    for cfg in configs:
        row = "  " + cfg.name.ljust(32)
        for ctx in contexts:
            kv = kv_cache_bytes(cfg, ctx, BYTES_BF16)
            ss = ssm_state_bytes(cfg, BYTES_BF16)
            total = kv + ss
            row += fmt_bytes(total).rjust(10)
        print(row)
    print()

    print("-" * 74)
    print("256K 上下文的主要节省（BF16，相对纯 Transformer full-MHA）")
    print("-" * 74)
    baseline = kv_cache_bytes(configs[0], 262_144, BYTES_BF16)
    for cfg in configs:
        kv = kv_cache_bytes(cfg, 262_144, BYTES_BF16)
        ss = ssm_state_bytes(cfg, BYTES_BF16)
        total = kv + ss
        savings = (1 - total / baseline) * 100
        print(f"  {cfg.name:<32} total {fmt_bytes(total):>10}  "
              f"（相对基线 {savings:+.1f}%）")
    print()

    print("-" * 74)
    print("Attention 层比例与 256K 上下文的内存占比（BF16）")
    print("-" * 74)
    for cfg in configs:
        attn_frac = cfg.attn_layers / cfg.total_layers if cfg.total_layers else 0
        kv = kv_cache_bytes(cfg, 262_144, BYTES_BF16)
        ss = ssm_state_bytes(cfg, BYTES_BF16)
        mem_frac = kv / (kv + ss + 1) if (kv + ss) > 0 else 0
        print(f"  {cfg.name:<32} attn_frac={attn_frac:.3f}  "
              f"kv_frac_of_total_cache={mem_frac:.3f}")
    print()

    print("=" * 74)
    print("结论")
    print("-" * 74)
    print("纯 Transformer 于 256k = 67 GB 仅用于 KV cache —— 不适合")
    print("  加上权重和激活后，无法装入单张 80GB GPU。")
    print("Jamba 1:7 = 8.4GB KV cache + 约 4MB SSM 状态，可以装入。")
    print("这具体说明了 AI21 文档中“单 GPU 支持 256K 上下文”的说法。")
    print("Mamba-3 进一步推进了纯 SSM；下一代混合架构很可能采用它作为 SSM 分支。")


if __name__ == "__main__":
    main()
