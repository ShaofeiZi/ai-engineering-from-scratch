"""推理平台经济性比较器，使用 Python stdlib。

在相同的合成工作负载上对六个提供商（Fireworks、Together、Baseten、Modal、
Replicate、Anyscale）建模。将按 token、按分钟和按预测计价统一归一化，以便直接比较。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vendor:
    name: str
    model: str
    per_mtok_output: float | None   # 每百万输出 token 的费用（不适用时为 None）
    per_minute: float | None        # 专用 GPU 每分钟费用（serverless 时为 None）
    per_prediction: float | None    # 每次预测费用（按 token 计费时为 None）
    tokens_per_minute: int          # GPU 饱和时的有效 token 数
    cold_start_sec: float
    notes: str
    min_reserved_minutes_per_day: int = 0  # 按分钟计费提供商的每日预留分钟下限（热池/最低承诺）


VENDORS = [
    Vendor("Fireworks",    "Llama 70B",          0.90,  None,    None,  900_000, 1.5, "FireAttention，批处理层五折"),
    Vendor("Together",     "Llama 70B",          0.88,  None,    None,  850_000, 2.0, "200 多个模型，比 Replicate 低 50%-70%"),
    Vendor("Baseten",      "自定义 Llama 70B",   None,  0.55,    None,  900_000, 5.0, "Truss、SOC2、HIPAA，按分钟计费", 1440),
    Vendor("Modal",        "自定义 Llama 70B",   None,  0.48,    None,  800_000, 2.5, "Python 原生，按秒计费，热池至少保留 60 分钟", 60),
    Vendor("Replicate",    "Llama 70B",          None,  None,    0.006, 750_000, 4.0, "按预测计费，多模态"),
    Vendor("Anyscale",     "Llama 70B RayTurbo", None,  0.60,    None,  850_000, 3.0, "Ray 原生，分布式 Python", 1440),
]


def cost_per_day(v: Vendor, tokens_per_day: int, predictions_per_day: int) -> float:
    """根据提供商的定价模型计算实际每日成本。

    按分钟计费的提供商按饱和服务时间与预留分钟下限（warm pool 最小值 / 预留值）
    两者中的较大值计费。这样，按分钟模型在 `run_scenario` 和
    `utilization_breakeven` 中保持一致，不会一处假设可完美缩容至零、另一处又假设
    全天 24 小时预留。
    """
    if v.per_mtok_output is not None:
        return (tokens_per_day / 1e6) * v.per_mtok_output
    if v.per_minute is not None:
        saturated_minutes = tokens_per_day / v.tokens_per_minute
        minutes = max(saturated_minutes, v.min_reserved_minutes_per_day)
        return minutes * v.per_minute
    if v.per_prediction is not None:
        return predictions_per_day * v.per_prediction
    return 0.0


def effective_rate(v: Vendor, tokens_per_day: int, predictions_per_day: int) -> float:
    """归一化为 $/M token，以便跨提供商比较。"""
    c = cost_per_day(v, tokens_per_day, predictions_per_day)
    return (c / (tokens_per_day / 1e6)) if tokens_per_day else 0


def run_scenario(label: str, tokens_per_day: int, predictions_per_day: int) -> None:
    print(f"\n{label}")
    print(f"工作负载：每天 {tokens_per_day/1e6:.1f}M 输出 token  |  每天 {predictions_per_day} 次预测")
    header = f"{'供应商':12}  {'模型':22}  {'$/天':>8}  {'$/M token':>10}  说明"
    print(header)
    print("-" * len(header))
    for v in VENDORS:
        cost = cost_per_day(v, tokens_per_day, predictions_per_day)
        rate = effective_rate(v, tokens_per_day, predictions_per_day)
        print(f"{v.name:12}  {v.model:22}  ${cost:7.2f}  ${rate:9.2f}  {v.notes}")


def utilization_breakeven() -> None:
    print("\n" + "=" * 80)
    print("按 TOKEN 与按分钟计费的盈亏平衡 — Fireworks 与 Baseten")
    print("=" * 80)
    fw = VENDORS[0]
    bt = VENDORS[2]
    print(f"Fireworks：${fw.per_mtok_output:.2f}/M 输出  |  Baseten：${bt.per_minute:.2f}/分钟，{bt.tokens_per_minute/1e3:.0f}k token/分钟\n")
    print(f"{'利用率 %':>8}  {'Fireworks $/天':>16}  {'Baseten $/天':>14}  胜出方案")
    for util_pct in (5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100):
        tokens_per_day = int(bt.tokens_per_minute * 60 * 24 * util_pct / 100)
        fw_cost = cost_per_day(fw, tokens_per_day, 0)
        bt_cost = cost_per_day(bt, tokens_per_day, 0)
        winner = "Baseten" if bt_cost < fw_cost else "Fireworks"
        print(f"{util_pct:>7}%  ${fw_cost:>15.2f}  ${bt_cost:>13.2f}  {winner}")


def cold_start_penalty() -> None:
    print("\n" + "=" * 80)
    print("冷启动惩罚 — 突发型工作负载")
    print("=" * 80)
    print(f"{'供应商':12}  {'冷启动':>11}  每天 100 次冷调用的影响")
    for v in VENDORS:
        impact_sec = v.cold_start_sec * 100
        print(f"{v.name:12}  {v.cold_start_sec:>8.1f} 秒   每天额外增加 {impact_sec:.0f} 秒延迟")


def main() -> None:
    print("=" * 80)
    print("推理平台经济性 — 2026 年近似值")
    print("=" * 80)

    run_scenario("场景 A — 初创规模的 LLM 产品",
                 tokens_per_day=2_000_000, predictions_per_day=10_000)
    run_scenario("场景 B — 高流量生产环境",
                 tokens_per_day=100_000_000, predictions_per_day=500_000)

    utilization_breakeven()
    cold_start_penalty()

    print("\n经验法则：在预留分钟计费下，当 GPU 饱和利用率持续高于约 60-70% 时，")
    print("按分钟计费（Baseten、Modal）优于按 token 计费；低于此范围则按 token 更划算。")


if __name__ == "__main__":
    main()
