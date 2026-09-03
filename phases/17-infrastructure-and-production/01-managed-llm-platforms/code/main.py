"""托管 LLM 平台比较器，使用 Python stdlib。

在相同的合成工作负载上对三个平台（Bedrock on-demand、Azure PTU、Vertex on-demand）
建模。报告每日成本、TTFT 中位数 / P99 和归因保真度。用于教学：价格和延迟是依据
2026 年公开资料得到的近似值。
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import statistics


@dataclass
class Platform:
    name: str
    per_mtok_input: float        # 按需输入 token 的 $/M
    per_mtok_output: float       # 按需输出 token 的 $/M
    ptu_hourly: float | None     # 一个预留单元的 $/小时（None 表示不提供）
    ptu_tokens_per_hour: int     # 单个 PTU 每小时可处理的 token 数
    ttft_median_ms: float        # 共享容量下的 TTFT 中位数
    ttft_p99_ms: float           # 共享容量下的 TTFT P99
    ttft_median_ptu_ms: float    # 专用 PTU 下的 TTFT 中位数
    attribution: str             # 定性的 FinOps 能力等级


PLATFORMS = [
    Platform("Bedrock 按需服务",    3.00, 15.00, 21.0, 1_200_000, 75, 180, 55, "A（应用推理配置文件）"),
    Platform("Azure OpenAI（PTU）",  2.50, 10.00, 10.0, 2_000_000, 50, 140, 38, "B（作用域 + 标签 + PTU 对象）"),
    Platform("Vertex AI Gemini",     1.25,  5.00, None,          0, 60, 160,  0, "B+（BQ 账单导出）"),
]


def simulate(tokens_in_per_day: int, tokens_out_per_day: int, sla_ttft_ms: float, use_ptu: bool) -> None:
    print(f"\n工作负载：每天 {tokens_in_per_day/1e6:.1f}M 输入、{tokens_out_per_day/1e6:.1f}M 输出")
    print(f"SLA：TTFT P99 < {sla_ttft_ms:.0f} ms   |   PTU 路径：{'启用' if use_ptu else '关闭'}\n")
    header = f"{'平台':25}  {'$/天':>9}  {'TTFT P50':>10}  {'TTFT P99':>10}  {'SLA':>6}  归因能力"
    print(header)
    print("-" * len(header))

    for p in PLATFORMS:
        cost_ondemand = (tokens_in_per_day / 1e6) * p.per_mtok_input + \
                        (tokens_out_per_day / 1e6) * p.per_mtok_output

        if use_ptu and p.ptu_hourly is not None:
            total_tokens = tokens_in_per_day + tokens_out_per_day
            daily_capacity_per_ptu = p.ptu_tokens_per_hour * 24
            ptu_count = max(1, (total_tokens + daily_capacity_per_ptu - 1) // daily_capacity_per_ptu)
            cost_ptu = ptu_count * p.ptu_hourly * 24
            cost = min(cost_ondemand, cost_ptu)
            ttft_p50 = p.ttft_median_ptu_ms if cost == cost_ptu else p.ttft_median_ms
            ttft_p99 = ttft_p50 * 1.5 if cost == cost_ptu else p.ttft_p99_ms
            path = "PTU" if cost == cost_ptu else "按需"
        else:
            cost = cost_ondemand
            ttft_p50 = p.ttft_median_ms
            ttft_p99 = p.ttft_p99_ms
            path = "按需"

        sla_ok = "通过" if ttft_p99 < sla_ttft_ms else "失败"
        print(f"{p.name:25}  ${cost:8.2f}  {ttft_p50:7.0f} 毫秒  {ttft_p99:7.0f} 毫秒  {sla_ok:>6}  {p.attribution}  [{path}]")


def break_even_demo() -> None:
    print("\n" + "=" * 80)
    print("PTU 盈亏平衡扫描 — Azure OpenAI，GPT-4o 级别")
    print("=" * 80)
    p = PLATFORMS[1]  # Azure
    print(f"按需费率：${p.per_mtok_output:.2f}/M 输出  |  PTU：${p.ptu_hourly:.0f}/小时，{p.ptu_tokens_per_hour/1e6:.1f}M token/小时\n")
    print(f"{'利用率 %':>8}  {'按需 $/天':>18}  {'PTU $/天':>12}  胜出方案")
    for util_pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        tokens_per_day = int(p.ptu_tokens_per_hour * 24 * (util_pct / 100.0))
        ondemand = (tokens_per_day / 1e6) * p.per_mtok_output
        ptu = 24 * p.ptu_hourly
        winner = "PTU" if ptu < ondemand else "按需"
        print(f"{util_pct:>7}%  ${ondemand:>16.2f}  ${ptu:>10.2f}  {winner}")


def lock_in_cost() -> None:
    print("\n" + "=" * 80)
    print("双提供商最低配置 — 冗余带来的成本增量")
    print("=" * 80)
    tokens_per_day = 5_000_000
    primary_cost = (tokens_per_day / 1e6) * 10.00
    gateway_overhead_pct = 3.0
    failover_headroom_pct = 10.0
    uplift = primary_cost * (gateway_overhead_pct + failover_headroom_pct) / 100
    print(f"主提供商每日支出：${primary_cost:.2f}")
    print(f"网关开销（{gateway_overhead_pct:.0f}%）：${primary_cost * gateway_overhead_pct / 100:.2f}/天")
    print(f"空闲的次级容量余量（{failover_headroom_pct:.0f}%）：${primary_cost * failover_headroom_pct / 100:.2f}/天")
    print(f"总增量：${uplift:.2f}/天")
    print(f"每月增量：${uplift * 30:.2f}")
    print("没有冗余时，一次持续数小时的区域中断会造成：客户流失、SLA 赔偿和作战室时间")


def main() -> None:
    print("=" * 80)
    print("托管 LLM 平台比较器 — 2026 年近似值")
    print("=" * 80)

    simulate(tokens_in_per_day=3_000_000, tokens_out_per_day=1_000_000, sla_ttft_ms=200, use_ptu=False)
    simulate(tokens_in_per_day=30_000_000, tokens_out_per_day=15_000_000, sla_ttft_ms=100, use_ptu=True)

    break_even_demo()
    lock_in_cost()


if __name__ == "__main__":
    main()
