"""可观测性采样与成本模拟器，使用 Python stdlib。

模拟一天 100 万条 trace 在不同保留策略下的情况。报告存储成本和各策略损失的内容。
用于教学：成本采用 2026 年近似值。
"""

from __future__ import annotations

from dataclasses import dataclass
import random


BYTES_PER_TRACE = 4_500            # prompt + 响应 + metadata
COST_PER_GB_MONTH = 0.023          # S3 标准存储
OBSERVABILITY_INGEST_PER_GB = 0.50 # 例如 Datadog 级别
ARIZE_AX_PER_GB = 0.005            # zero-copy 宣称值


@dataclass
class Strategy:
    name: str
    sample_rate: float
    keep_errors: bool
    keep_highcost: bool


STRATEGIES = [
    Strategy("保留 100%",                   1.00, True, True),
    Strategy("随机采样 10%",                0.10, False, False),
    Strategy("成功项 5% + 错误项 100%",     0.05, True, False),
    Strategy("成功项 5% + 错误项 + 高成本项", 0.05, True, True),
    Strategy("仅保留 1% 聚合数据",           0.01, True, True),
]


def simulate_day(strategy: Strategy, traces_per_day: int = 1_000_000) -> dict:
    rng = random.Random(7)
    retained = 0
    lost = 0
    for i in range(traces_per_day):
        is_error = rng.random() < 0.02
        is_highcost = rng.random() < 0.01
        keep = rng.random() < strategy.sample_rate
        if strategy.keep_errors and is_error:
            keep = True
        if strategy.keep_highcost and is_highcost:
            keep = True
        if keep:
            retained += 1
        else:
            lost += 1
    bytes_retained = retained * BYTES_PER_TRACE
    gb = bytes_retained / 1e9
    return {
        "name": strategy.name,
        "retained": retained,
        "lost": lost,
        "gb_per_day": gb,
        "s3_month": gb * 30 * COST_PER_GB_MONTH,
        "monolithic_month": gb * 30 * OBSERVABILITY_INGEST_PER_GB,
        "arize_month": gb * 30 * ARIZE_AX_PER_GB,
    }


def report(row: dict) -> None:
    print(f"{row['name']:30}  保留={row['retained']:7}  "
          f"丢弃={row['lost']:7}  {row['gb_per_day']:6.2f} GB/天  "
          f"mono=${row['monolithic_month']:8.2f}  "
          f"arize=${row['arize_month']:6.2f}  "
          f"s3=${row['s3_month']:5.2f}")


def main() -> None:
    print("=" * 120)
    print("可观测性采样 — 每天 100 万条 trace，2026 年价格近似值")
    print("=" * 120)
    for s in STRATEGIES:
        report(simulate_day(s))

    print()
    print("解读：Datadog 级别的 100% 保留每天要花费数百美元。")
    print("保留 5% 成功项 + 100% 错误项 + 高成本项，既保留信号，又削减 90% 账单。")
    print("已有数据湖时，Arize AX zero-copy 模式在大规模场景下更有优势。")


if __name__ == "__main__":
    main()
