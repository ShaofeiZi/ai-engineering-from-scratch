"""管道调度模拟器 — 1F1B vs Zero Bubble vs DualPipe vs DualPipeV.

教学工具. 按给定的时间表计算输油管泡(P, micro batches).
产出:
- 固定时的气泡分数(P,微微小)
- 随着微型小炮管的增大而扩大气泡

不是生产模拟器 转发/backward块成本为单位正常化。
通信成本的模型是重叠窗口,而不是完整的内核模型.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ScheduleStats:
    name: str
    stable_bubble_frac: float
    scales_with_micro_batches: bool
    param_copies: int
    comm_overlap: str


def bubble_1f1b(P: int, M: int) -> float:
    """1F1B:热身阶段有(P-1)前位,没有后向重叠.
冷却镜. 稳定阶段每个小批量的等级为零气泡,
但热热/cooldown气泡是(P-1)前方+(P-1)后方块/
军衔,在2 * M + 2 * (P-1) 中,共块。
"""
    total = 2 * M + 2 * (P - 1)
    bubble = 2 * (P - 1)
    return bubble / total


def bubble_zero_bubble(P: int, M: int) -> float:
    """零泡(Qi 2023)向后分裂为B+W. W部分可以填充
1F1B的气泡。 大约残泡为(P-1) / 2块
热身加同样的冷却,从3 * M + 2 * (P-1) 子圆柱中.
"""
    total = 3 * M + 2 * (P - 1)
    bubble = (P - 1)
    return bubble / total


def bubble_dualpipe(P: int, M: int) -> float:
    """DualPipe从管道两端注入微管。 稳定
相位气泡为零. 温泡/cooldown有独立于M的固定气泡.
"""
    total = 3 * M + (P - 1)
    bubble = (P - 1) // 2
    return bubble / total


def bubble_dualpipev(P: int, M: int) -> float:
    """DualPipeV在单个参数副本上使用V形表. 内容
气泡比“ ph1 ” 稍大一点, 其作用是将气泡减半 。
记忆。 约似1.2x DualPipe泡."""
    return bubble_dualpipe(P, M) * 1.2


def summarize(P: int, M: int) -> List[tuple[str, float, int, str]]:
    return [
        ("1F1B",       bubble_1f1b(P, M),        1, "minimal"),
        ("Zero Bubble", bubble_zero_bubble(P, M), 1, "partial"),
        ("DualPipe",   bubble_dualpipe(P, M),    2, "full"),
        ("DualPipeV",  bubble_dualpipev(P, M),   1, "partial"),
    ]


def gpu_hours_recovered(P: int, M: int, total_gpu_hours: float) -> dict:
    b1 = bubble_1f1b(P, M)
    bd = bubble_dualpipe(P, M)
    recovered = (b1 - bd) * total_gpu_hours
    return {
        "1F1B_bubble_frac": b1,
        "DualPipe_bubble_frac": bd,
        "recovered_gpu_hours": recovered,
    }


def main() -> None:
    print("=" * 70)
    print("DualPipe 并行模拟器（第 10 阶段，第 19 课）")
    print("=" * 70)
    print()

    print("-" * 70)
    print("步骤 1：P=8、micro-batch=16 时的气泡比例")
    print("-" * 70)
    print(f"  {'调度方式':<14} {'气泡':>10} {'参数副本':>14} {'通信重叠':>14}")
    for name, b, pc, co in summarize(P=8, M=16):
        print(f"  {name:<14} {b:>9.1%}  {pc:>14}  {co:>14}")
    print()

    print("-" * 70)
    print("步骤 2：气泡比例随 micro-batch 数量缩放（P=8）")
    print("-" * 70)
    header = "  " + "M".rjust(6)
    for name in ("1F1B", "ZeroBubble", "DualPipe", "DualPipeV"):
        header += name.rjust(12)
    print(header)
    for M in (4, 8, 16, 32, 64, 128):
        row = f"  {M:>6}"
        for _, b, _, _ in summarize(P=8, M=M):
            row += f"{b:>12.1%}"
        print(row)
    print()

    print("-" * 70)
    print("步骤 3：气泡比例随流水线深度缩放（M=64）")
    print("-" * 70)
    header = "  " + "P".rjust(6)
    for name in ("1F1B", "ZeroBubble", "DualPipe", "DualPipeV"):
        header += name.rjust(12)
    print(header)
    for P in (4, 8, 16, 32, 64):
        row = f"  {P:>6}"
        for _, b, _, _ in summarize(P=P, M=64):
            row += f"{b:>12.1%}"
        print(row)
    print()

    print("-" * 70)
    print("步骤 4：回收 GPU 小时（DeepSeek-V3 规模运行）")
    print("-" * 70)
    print("DeepSeek-V3：2048 张 H800 GPU，总计约 280 万 GPU 小时。")
    print("假设流水线深度 P=16，每步有 M=128 个 micro-batch。")
    r = gpu_hours_recovered(P=16, M=128, total_gpu_hours=2_800_000)
    print(f"  1F1B 气泡：    {r['1F1B_bubble_frac']:.1%}")
    print(f"  DualPipe 气泡：{r['DualPipe_bubble_frac']:.1%}")
    print(f"  回收：         {r['recovered_gpu_hours']:,.0f} GPU 小时")
    print("  （约等于一次完整 70B 稠密模型预训练的成本）")
    print()

    print("要点：DualPipe 的气泡不会随 M 增长。2 倍参数副本在 MoE 规模下")
    print("尚可接受，因为专家并行已经分散了占主导的权重。")
    print("DualPipeV 将副本减至 1 倍，同时只增加很小的气泡成本。")


if __name__ == "__main__":
    main()
