"""冷启动缓解路径模拟器，使用 Python stdlib。

对 70B 模型在不同缓解技术栈下的冷启动进行建模：
  RAW              ：无缓解措施（名义基线）
  PRE_SEEDED       ：+ Bottlerocket 预置节点镜像
  STREAMER         ：+ NVIDIA Run:ai Model Streamer
  GPU_SNAPSHOT     ：+ Modal 风格的 GPU snapshot
  WARM_POOL        ：min_workers=1（warm path 完全没有冷启动）

报告各层耗时和总耗时，同时计算 warm pool 的盈亏平衡点。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Phase:
    name: str
    raw_sec: float
    pre_seeded_sec: float    # 如果被消除则为 0
    streamer_sec: float      # streamer 激活时替代原始值
    snapshot_sec: float      # snapshot 激活时替代全部步骤


PHASES_70B = [
    Phase("节点预配",          50.0, 50.0,  50.0,  0.5),
    Phase("拉取镜像",         180.0,  0.0, 180.0,  0.0),
    Phase("权重载入 HBM",      75.0, 75.0,  35.0,  0.0),
    Phase("引擎初始化",        20.0, 20.0,  20.0,  2.0),
    Phase("首次前向传播",       3.0,  3.0,   3.0,  0.5),
]


def total_for_stack(stack: set[str]) -> float:
    seconds = 0.0
    for phase in PHASES_70B:
        if "gpu_snapshot" in stack:
            seconds += phase.snapshot_sec
        elif "streamer" in stack and "pre_seeded" in stack:
            used = phase.pre_seeded_sec
            if phase.name == "权重载入 HBM":
                used = phase.streamer_sec
            seconds += used
        elif "pre_seeded" in stack:
            seconds += phase.pre_seeded_sec
        elif "streamer" in stack:
            seconds += phase.streamer_sec if phase.name == "权重载入 HBM" else phase.raw_sec
        else:
            seconds += phase.raw_sec
    return seconds


def report_stack(label: str, stack: set[str]) -> None:
    total = total_for_stack(stack)
    mins = total / 60
    print(f"{label:20}  {total:6.1f} 秒  （{mins:4.1f} 分钟）  技术栈={sorted(stack) if stack else '{基线}'}")


def warm_pool_break_even(gpu_hourly: float, cold_seconds: float, sla_tolerated_drops_per_day: int) -> None:
    print("\n" + "=" * 80)
    print("WARM POOL 盈亏平衡")
    print("=" * 80)
    print(f"GPU 成本：${gpu_hourly:.2f}/小时  |  冷启动：{cold_seconds:.0f} 秒  |  每日丢弃预算：{sla_tolerated_drops_per_day}\n")
    warm_monthly = gpu_hourly * 24 * 30
    print(f"热池（min_workers=1）每月成本：${warm_monthly:.2f}")
    print()
    print(f"{'请求/小时':>8}  {'预期冷启动/天':>24}  {'超预算丢弃数':>20}  {'热池更优？':>15}")
    for rate in (1, 5, 10, 25, 50, 100, 250):
        cold_starts_per_day = 24 / max(rate, 1) if rate < 1 else 1
        cold_starts_per_day = min(20, max(1, int(24 * 3600 / (rate * 3600))))
        drops = cold_starts_per_day
        warm_better = "是" if drops > sla_tolerated_drops_per_day else "否"
        print(f"{rate:>8}  {cold_starts_per_day:>24}  {max(0, drops - sla_tolerated_drops_per_day):>20}  {warm_better:>15}")


def main() -> None:
    print("=" * 80)
    print("冷启动缓解 — 新 H100 节点上的 70B 模型")
    print("=" * 80)
    print(f"{'技术栈':20}  {'总计':>8}             技术栈组成")
    print("-" * 80)

    report_stack("RAW",                      set())
    report_stack("+ PRE_SEEDED",             {"pre_seeded"})
    report_stack("+ STREAMER",               {"streamer"})
    report_stack("+ PRE_SEEDED + STREAMER",  {"pre_seeded", "streamer"})
    report_stack("+ GPU_SNAPSHOT",           {"gpu_snapshot"})

    print("\n（WARM_POOL 在热路径上完全避免冷启动；代价是全天候租用 GPU）")

    warm_pool_break_even(gpu_hourly=4.50, cold_seconds=328, sla_tolerated_drops_per_day=5)


if __name__ == "__main__":
    main()
