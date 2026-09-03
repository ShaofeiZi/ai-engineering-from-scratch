"""玩具版量化内存与吞吐量计算器，使用 Python stdlib。

针对一组量化格式和模型规模，计算：
  - 权重内存
  - KV cache 内存（单独计算，随并发数和上下文增长）
  - activation 内存（近似值）
  - 相对 decode 吞吐量（受内存带宽限制的趋势）

格式由有效权重位数和 KV 位数表示，仅用于教学。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Format:
    name: str
    weight_bits: float
    kv_bits: float
    engine: str
    notes: str


FORMATS = [
    Format("BF16 基线（vLLM）",            16, 16, "vLLM",     "参考配置"),
    Format("GGUF Q5_K_M（llama.cpp）",      5, 16, "llama.cpp", "CPU/边缘设备"),
    Format("GGUF Q4_K_M（llama.cpp）",      4, 16, "llama.cpp", "CPU/边缘设备，默认配置"),
    Format("GPTQ-Int4 + Marlin（vLLM）",    4, 16, "vLLM",     "支持多 LoRA"),
    Format("AWQ-Int4 + Marlin（vLLM）",     4, 16, "vLLM",     "INT4 下 Pass@1 最佳"),
    Format("FP8（vLLM / TRT-LLM）",         8,  8, "multi",    "推理任务的安全默认值"),
    Format("NVFP4 + FP8 KV（TRT-LLM）",     4,  8, "TRT-LLM",  "Blackwell 激进配置"),
]


def memory_breakdown(params_b: float, fmt: Format,
                     concurrency: int = 128, ctx: int = 2048) -> dict:
    weight_gb = params_b * fmt.weight_bits / 8
    # KV cache 近似值：num_layers * 2 * kv_heads * head_dim * ctx * 每元素字节数
    layers = 64 * (params_b / 70.0)**0.5
    kv_heads = 8
    head_dim = 128
    per_seq_kv_gb = layers * 2 * kv_heads * head_dim * ctx * (fmt.kv_bits / 8) / 1e9
    kv_total = per_seq_kv_gb * concurrency
    activations_gb = 0.05 * params_b       # 粗略常量
    return {
        "weight": weight_gb,
        "kv": kv_total,
        "act": activations_gb,
        "total": weight_gb + kv_total + activations_gb,
    }


def relative_throughput(fmt: Format) -> float:
    """decode 受内存带宽限制。每 token 的权重字节数越少，吞吐量越高。
    归一化为 BF16 = 1.0。"""
    return 16 / fmt.weight_bits


def gpu_check(total_gb: float) -> str:
    if total_gb <= 80:
        return "H100 80GB"
    if total_gb <= 141:
        return "H200 141GB"
    if total_gb <= 192:
        return "B200 192GB"
    return "多 GPU"


def print_scenario(params_b: float, concurrency: int, ctx: int) -> None:
    print(f"模型：{params_b}B 参数  |  并发数 {concurrency}  |  上下文 {ctx}")
    print("-" * 98)
    print(f"{'格式':36} {'权重 GB':>7} {'KV GB':>7} {'激活 GB':>7} "
          f"{'总计':>7} {'适用设备':>14} {'相对吞吐':>10}")
    for f in FORMATS:
        m = memory_breakdown(params_b, f, concurrency, ctx)
        tput = relative_throughput(f)
        print(f"{f.name:36} {m['weight']:7.1f} {m['kv']:7.1f} {m['act']:7.1f} "
              f"{m['total']:7.1f} {gpu_check(m['total']):>14} {tput:10.2f}x")
    print()


def main() -> None:
    print("=" * 98)
    print("玩具版量化计算器 — 各格式的内存与相对吞吐量")
    print("=" * 98)
    print()

    print_scenario(params_b=7, concurrency=128, ctx=2048)
    print_scenario(params_b=70, concurrency=128, ctx=2048)
    print_scenario(params_b=70, concurrency=256, ctx=8192)
    print_scenario(params_b=405, concurrency=128, ctx=2048)

    print("=" * 98)
    print("关键发现")
    print("-" * 98)
    print("  1. KV cache 随并发数 x 上下文线性增长。")
    print("     对 70B 模型使用 256 并发 / 8k 上下文时，仅 KV 就会远超权重节省量。")
    print("  2. AWQ 与 GPTQ 的 4-bit 占用相同；选择取决于 LoRA 支持和 kernel。")
    print("  3. NVFP4 + FP8 KV 组合会同时缩小权重和 KV，但仅限 Blackwell。")
    print("  4. 对推理工作负载而言，尽管内存占用更高，FP8 仍是安全默认值。")
    print("  5. GGUF 在 CPU 上胜出；vLLM 中约 93 token/秒不是缺陷，而是选错了引擎。")


if __name__ == "__main__":
    main()
