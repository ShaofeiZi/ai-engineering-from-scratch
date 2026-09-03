"""AI 网关路由与回退模拟器——使用 Python 标准库。

模拟一个位于 OpenAI、Anthropic 和自托管服务之前的网关。为各提供商注入
429/5xx 错误，并比较回退策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class Provider:
    name: str
    base_latency_ms: float
    error_rate: float
    overhead_ms: float


PROVIDERS = [
    Provider("OpenAI",       180, 0.03, 0),
    Provider("Anthropic",    220, 0.02, 0),
    Provider("Self-hosted",  100, 0.05, 0),
]

GATEWAY_OVERHEAD = {
    "LiteLLM": 10,
    "Portkey": 30,
    "Kong":      5,
    "Cloudflare": 2,
}


def call_provider(p: Provider, rng: random.Random) -> tuple[bool, float]:
    if rng.random() < p.error_rate:
        return False, p.base_latency_ms * 0.3  # 出错前已完成部分工作
    return True, p.base_latency_ms


def simulate_fallback(gateway: str, n: int = 1000, seed: int = 7) -> dict:
    rng = random.Random(seed)
    success = 0
    total_latency = 0.0
    retries = 0
    fallback_hits = 0
    gw_ovh = GATEWAY_OVERHEAD[gateway]

    for _ in range(n):
        req_latency = gw_ovh
        done = False
        for attempt, p in enumerate(PROVIDERS):
            ok, ms = call_provider(p, rng)
            req_latency += ms
            if attempt > 0:
                fallback_hits += 1
            if ok:
                success += 1
                done = True
                break
            retries += 1
        total_latency += req_latency

    return {
        "gateway": gateway,
        "success_rate": success / n,
        "mean_latency": total_latency / n,
        "retries": retries,
        "fallback_hits": fallback_hits,
    }


def report(row: dict) -> None:
    print(f"{row['gateway']:12}  成功率={row['success_rate']*100:5.1f}%  "
          f"平均延迟={row['mean_latency']:6.0f}毫秒  "
          f"重试={row['retries']:4}  回退={row['fallback_hits']:4}")


def main() -> None:
    print("=" * 80)
    print("AI 网关回退——注入错误时的三提供商链路")
    print("=" * 80)
    header = f"{'网关':12}  {'成功率':>7}         {'平均延迟':>12}  重试  回退"
    print(header)
    print("-" * len(header))
    for gw in ("LiteLLM", "Portkey", "Kong", "Cloudflare"):
        report(simulate_fallback(gw))

    print("\n说明：单一提供商的错误率为 3% 时，成功率为 97%。")
    print("双提供商回退的成功率为 99.94%（0.03 × 0.02 的补集）。")
    print("三提供商回退的成功率为 99.997%，但回退会增加延迟。")


if __name__ == "__main__":
    main()
