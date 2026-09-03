"""带分级执行机制的多租户 LLM FinOps 模拟器——使用 Python 标准库。

三级执行机制：
  1. 每租户限流
  2. 每租户每日支出上限
  3. 支出 z 分数大于 4 时触发终止开关
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import statistics


@dataclass
class TenantPolicy:
    contracted_daily_usd: float
    rate_limit_per_min: int
    spend_cap_multiplier: float = 2.0
    kill_z_score: float = 4.0


@dataclass
class TenantState:
    spend_today_usd: float = 0.0
    minute_count: int = 0
    daily_history: list = field(default_factory=list)
    paused: bool = False


TENANTS = {
    "tenant_A_normal":  (TenantPolicy(100.0, rate_limit_per_min=120), TenantState(), 1.0),
    "tenant_B_growing": (TenantPolicy(50.0,  rate_limit_per_min=60),  TenantState(), 2.5),
    "tenant_C_abusive": (TenantPolicy(20.0,  rate_limit_per_min=40),  TenantState(), 25.0),
}


def simulate_day(day: int, verbose: bool) -> None:
    for name, (policy, state, traffic_mult) in TENANTS.items():
        if state.paused:
            continue
        requests = int(100 * traffic_mult * random.uniform(0.8, 1.3))
        tokens_per_req = int(random.gauss(600, 150))
        cost_per_req = (tokens_per_req / 1e6) * 10.0
        total_spend = requests * cost_per_req
        state.spend_today_usd += total_spend

        if state.spend_today_usd > policy.contracted_daily_usd * policy.spend_cap_multiplier:
            if verbose:
                print(f"  [突破上限] {name}：${state.spend_today_usd:.2f} > 上限 ${policy.contracted_daily_usd * policy.spend_cap_multiplier:.2f} → 收紧限流并通知客户成功团队")

        if len(state.daily_history) >= 5:
            mean = statistics.mean(state.daily_history)
            sd = statistics.stdev(state.daily_history) or 1
            z = (state.spend_today_usd - mean) / sd
            if z > policy.kill_z_score:
                state.paused = True
                if verbose:
                    print(f"  [终止开关] {name}：z={z:.2f}，支出 ${state.spend_today_usd:.2f}（基线 ${mean:.2f} ± ${sd:.2f}）→ 自动暂停并呼叫值班人员")


def main() -> None:
    print("=" * 95)
    print("FINOPS 执行机制——三名租户运行 10 天，滥用租户触发终止开关")
    print("=" * 95)
    random.seed(7)

    for day in range(1, 11):
        print(f"\n— 第 {day} 天 —")
        simulate_day(day, verbose=True)
        for name, (policy, state, _) in TENANTS.items():
            status = "已暂停" if state.paused else "活跃"
            print(f"  {name}：支出=${state.spend_today_usd:7.2f}，合约额度=${policy.contracted_daily_usd:.2f}  [{status}]")
            state.daily_history.append(state.spend_today_usd)
            if not state.paused:
                state.spend_today_usd = 0.0
    print("\n解读：限流用于节流，上限触发告警，终止开关拦截支出失控。")


if __name__ == "__main__":
    main()
