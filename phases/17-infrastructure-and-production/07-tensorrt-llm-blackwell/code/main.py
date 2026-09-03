"""玩具版 Blackwell + TRT-LLM 经济性计算器，使用 Python stdlib。

计算模型在以下技术栈中的 HBM 占用和 decode 吞吐量：
  H100 + BF16 + vLLM
  H100 + FP8 + vLLM
  B200 + NVFP4 weights / FP8 KV + TRT-LLM + Dynamo
  GB200 NVL72 + NVFP4 / FP8 + TRT-LLM + Dynamo

decode 吞吐量模型受内存带宽限制：token/秒与 HBM 带宽 / 每 token 字节数成正比。
这些数值仅用于展示 2026 年 Blackwell 经济性的趋势。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stack:
    name: str
    hbm_gb: int               # 单块 GPU 的 HBM
    hbm_bw_tbs: float         # HBM 带宽，单位 TB/s
    weight_bits: float        # 有效权重精度
    kv_bits: float            # KV cache 精度
    mtp_factor: float         # 1.0 = 无 draft，1.8 = 启用 MTP
    disagg_factor: float      # 分离式架构带来的额外吞吐量
    price_per_gpu_hour: float


STACKS = [
    Stack("H100 + BF16 + vLLM",           80, 3.35,  16, 16, 1.0,  1.0,  2.50),
    Stack("H100 + FP8 + vLLM",            80, 3.35,   8,  8, 1.0,  1.0,  2.50),
    Stack("H200 + FP8 + vLLM",           141, 4.80,   8,  8, 1.0,  1.0,  3.50),
    Stack("B200 + NVFP4 + FP8 + TRT-LLM", 192, 8.00,   4,  8, 1.8,  1.6,  4.80),
    Stack("GB200 NVL72 + TRT-LLM + Dyn", 192, 8.00,   4,  8, 1.8,  2.5,  6.20),
]


def hbm_footprint_gb(params_b: float, active_b: float, seq_len: int, stack: Stack) -> tuple[float, float]:
    weight_gb = params_b * stack.weight_bits / 8
    # 典型 head 配置的 KV cache：num_layers * 2 * num_kv_heads * head_dim * seq_len * 每元素字节数
    # 使用一个有代表性的 70B 形状，再按激活参数规模缩放
    layers = 64 * (active_b / 35.0)**0.5
    kv_heads = 8
    head_dim = 128
    kv_gb = layers * 2 * kv_heads * head_dim * seq_len * (stack.kv_bits / 8) / 1e9
    return weight_gb, kv_gb


def decode_throughput(active_b: float, stack: Stack) -> float:
    """每块 GPU 每秒处理的 token 数，受内存带宽限制。
    每个 decode token 会读取 `active_b * weight_bits/8` 字节的权重。
    """
    bytes_per_token = active_b * 1e9 * stack.weight_bits / 8
    raw_tokens_per_s = stack.hbm_bw_tbs * 1e12 / bytes_per_token
    return raw_tokens_per_s * stack.mtp_factor * stack.disagg_factor


def cost_per_million_tokens(active_b: float, stack: Stack) -> float:
    tps = decode_throughput(active_b, stack)
    tokens_per_hour = tps * 3600
    return stack.price_per_gpu_hour / tokens_per_hour * 1e6


def print_stack(params_b: float, active_b: float, seq_len: int = 8192) -> None:
    print(f"模型：总计 {params_b}B 参数，激活 {active_b}B，上下文 {seq_len:,} token")
    print("-" * 90)
    print(f"{'技术栈':40} {'权重 GB':>7} {'KV GB':>7} {'token/秒':>9} {'$/M token':>10}")
    for s in STACKS:
        w, kv = hbm_footprint_gb(params_b, active_b, seq_len, s)
        tps = decode_throughput(active_b, s)
        cost = cost_per_million_tokens(active_b, s)
        fits = "" if (w + kv) <= s.hbm_gb else "  （多 GPU）"
        print(f"{s.name:40} {w:7.1f} {kv:7.2f} {tps:9.0f} {cost:10.4f}{fits}")
    print()


def main() -> None:
    print("=" * 90)
    print("玩具版 BLACKWELL + TRT-LLM 经济性 — 受内存带宽限制的 decode")
    print("=" * 90)
    print()

    print_stack(70, 70)    # 70B 稠密模型
    print_stack(120, 36)   # GPT-OSS-120B MoE（30% 激活）
    print_stack(405, 405)  # Llama 3.1 405B 稠密模型
    print_stack(671, 37)   # DeepSeek-V3 规模的 MoE

    print("=" * 90)
    print("关键发现")
    print("-" * 90)
    print("  7 倍成本差距由四个来源叠加而成：")
    print("    1. HBM 带宽（H100 3.35 TB/s，B200 8.0 TB/s）约 2.4x")
    print("    2. NVFP4 权重（每 token 字节数减半）             约 2.0x")
    print("    3. MTP draft（对已接受 token 约 1.8x）           约 1.8x")
    print("    4. 分离式架构（Dynamo：约 1.6-2.5x）            约 2.0x")
    print("  原始乘积约 14x；计入开销和真实流量 alpha 后更接近 7x。")
    print("  迁移推理密集型工作负载前，请先验证 NVFP4 的质量。")


if __name__ == "__main__":
    main()
