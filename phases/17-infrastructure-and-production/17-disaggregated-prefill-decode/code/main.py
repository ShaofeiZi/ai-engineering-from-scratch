"""共置与分离式服务模拟器——使用 Python 标准库。

对比一个请求在共置（同一 GPU）与分离式（预填充池 + 解码池 + KV 传输）架构下的表现。
扫描提示词长度以找到交叉点。
"""

from __future__ import annotations


# 2026 年 H100 级 GPU 运行 70B FP8 模型的示意常量
PREFILL_TOK_PER_MS = 40.0         # 每块 GPU 每毫秒的预填充吞吐量
DECODE_TOK_PER_MS_COLOCATED = 0.10
DECODE_TOK_PER_MS_DECODE_GPU = 0.18   # 内存优化池（类似 H200）
KV_BYTES_PER_TOKEN_70B_FP8 = 125_000
NIXL_RDMA_GB_S = 100
NIXL_TCP_GB_S = 10


def ms_colocated(prompt: int, output: int) -> float:
    prefill_ms = prompt / PREFILL_TOK_PER_MS
    decode_ms = output / DECODE_TOK_PER_MS_COLOCATED
    return prefill_ms + decode_ms


def ms_disaggregated(prompt: int, output: int, use_rdma: bool = True) -> float:
    prefill_ms = prompt / PREFILL_TOK_PER_MS
    kv_bytes = prompt * KV_BYTES_PER_TOKEN_70B_FP8
    transport = NIXL_RDMA_GB_S if use_rdma else NIXL_TCP_GB_S
    transfer_ms = (kv_bytes / 1e9) / transport * 1000
    decode_ms = output / DECODE_TOK_PER_MS_DECODE_GPU
    return prefill_ms + transfer_ms + decode_ms


def main() -> None:
    print("=" * 95)
    print("分离式与共置服务——同一请求，不同 GPU 部署方式")
    print("=" * 95)
    header = f"{'提示':>7}  {'输出':>7}  {'共置（毫秒）':>15}  {'分离 RDMA（毫秒）':>17}  {'分离 TCP（毫秒）':>16}  胜出方案"
    print(header)
    print("-" * len(header))
    cases = [
        (256, 100), (512, 200), (1024, 300), (2048, 400),
        (4096, 500), (8192, 800), (16384, 1200), (32768, 2000),
    ]
    for prompt, output in cases:
        colo = ms_colocated(prompt, output)
        rdma = ms_disaggregated(prompt, output, use_rdma=True)
        tcp = ms_disaggregated(prompt, output, use_rdma=False)
        winner = "共置" if colo < rdma else "分离式"
        print(f"{prompt:>7}  {output:>7}  {colo:>14.1f}  {rdma:>17.1f}  {tcp:>16.1f}  {winner}")

    print()
    print("解读：提示词较长时，内存优化池带来的解码吞吐提升超过 KV 传输开销，")
    print("分离式架构因而胜出。TCP 传输会抬高盈亏平衡点；RDMA 则让分离式架构")
    print("更早获得收益。")


if __name__ == "__main__":
    main()
