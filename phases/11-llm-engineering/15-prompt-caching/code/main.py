"""快速缓存会计师。

模拟三种供方缓存制度(Anthropic ephemeral 5m, Anthropic 1h,
OpenAI 自动,双子座显式)针对一系列请求和报告
写入 /read/miss 计数, 加上每1K个请求混合成本 。

下面的价格是2026年4月公布的输入率tokens
供应商的前沿模型。 由编辑价格取代。

运行方式 :
python 主页.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# 输入-token价格,每1Ktokens美元 ——————————————————————————.

PRICES = {
    "anthropic_claude_opus_4_7": {"base": 0.015, "cache_write_5m": 0.01875, "cache_write_1h": 0.030, "cache_read": 0.0015},
    "openai_gpt_5": {"base": 0.005, "cache_write": 0.005, "cache_read": 0.0025},
    "gemini_3_pro": {"base": 0.00125, "cache_write": 0.00125, "cache_read": 0.0003125, "storage_per_1k_per_hour": 0.0000125},
}


@dataclass
class Request:
    """单个请求。 `prefix_tokens` 是可缓存的前缀; `suffix_tokens` 是用户输入。"""

    prefix_tokens: int
    suffix_tokens: int
    prefix_key: str


@dataclass
class CacheEntry:
    tokens: int
    written_at: int  # 请求索引
    ttl_seconds: int


@dataclass
class ProviderStats:
    writes: int = 0
    reads: int = 0
    misses: int = 0
    input_cost: float = 0.0
    storage_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.storage_cost


def simulate_anthropic(requests: Iterable[Request], ttl_seconds: int, seconds_between: int) -> ProviderStats:
    p = PRICES["anthropic_claude_opus_4_7"]
    write_rate = p["cache_write_1h"] if ttl_seconds > 300 else p["cache_write_5m"]
    stats = ProviderStats()
    cache: dict[str, CacheEntry] = {}
    for i, r in enumerate(requests):
        now_seconds = i * seconds_between
        entry = cache.get(r.prefix_key)
        expired = entry is None or (now_seconds - entry.written_at) >= entry.ttl_seconds
        if expired:
            stats.writes += 1
            stats.input_cost += (r.prefix_tokens / 1000) * write_rate
            cache[r.prefix_key] = CacheEntry(tokens=r.prefix_tokens, written_at=now_seconds, ttl_seconds=ttl_seconds)
        else:
            stats.reads += 1
            stats.input_cost += (r.prefix_tokens / 1000) * p["cache_read"]
        stats.input_cost += (r.suffix_tokens / 1000) * p["base"]
    return stats


def simulate_openai(requests: Iterable[Request], seconds_between: int) -> ProviderStats:
    """OpenAI的缓存是自动的;我们用1h最佳TTL来模拟它总是-on."""
    p = PRICES["openai_gpt_5"]
    stats = ProviderStats()
    cache: dict[str, CacheEntry] = {}
    for i, r in enumerate(requests):
        now_seconds = i * seconds_between
        entry = cache.get(r.prefix_key)
        expired = entry is None or (now_seconds - entry.written_at) >= 3600
        if expired:
            stats.writes += 1
            stats.input_cost += (r.prefix_tokens / 1000) * p["cache_write"]
            cache[r.prefix_key] = CacheEntry(tokens=r.prefix_tokens, written_at=now_seconds, ttl_seconds=3600)
        else:
            stats.reads += 1
            stats.input_cost += (r.prefix_tokens / 1000) * p["cache_read"]
        stats.input_cost += (r.suffix_tokens / 1000) * p["base"]
    return stats


def simulate_gemini(requests: Iterable[Request], ttl_seconds: int, seconds_between: int) -> ProviderStats:
    p = PRICES["gemini_3_pro"]
    stats = ProviderStats()
    cache: dict[str, CacheEntry] = {}
    for i, r in enumerate(requests):
        now_seconds = i * seconds_between
        entry = cache.get(r.prefix_key)
        expired = entry is None or (now_seconds - entry.written_at) >= entry.ttl_seconds
        if expired:
            stats.writes += 1
            stats.input_cost += (r.prefix_tokens / 1000) * p["cache_write"]
            cache[r.prefix_key] = CacheEntry(tokens=r.prefix_tokens, written_at=now_seconds, ttl_seconds=ttl_seconds)
        else:
            stats.reads += 1
            stats.input_cost += (r.prefix_tokens / 1000) * p["cache_read"]
        stats.input_cost += (r.suffix_tokens / 1000) * p["base"]
    # 存储成本:每个条目使用ttl,每token小时计费
    for entry in cache.values():
        hours = entry.ttl_seconds / 3600
        stats.storage_cost += (entry.tokens / 1000) * p["storage_per_1k_per_hour"] * hours
    return stats


def baseline_cost(requests: list[Request], provider: str) -> float:
    p = PRICES[provider]
    return sum((r.prefix_tokens + r.suffix_tokens) / 1000 * p["base"] for r in requests)


def make_traffic(n_requests: int, n_prefixes: int, prefix_size: int, suffix_size: int) -> list[Request]:
    return [
        Request(
            prefix_tokens=prefix_size,
            suffix_tokens=suffix_size,
            prefix_key=f"prefix_{i % n_prefixes}",
        )
        for i in range(n_requests)
    ]


def print_report(name: str, stats: ProviderStats, baseline: float, n: int) -> None:
    savings = 1 - (stats.total_cost / baseline) if baseline > 0 else 0
    print(f"\n{name}")
    print(f"  写入 {stats.writes:>5}  读取 {stats.reads:>5}  未命中 {stats.misses:>5}")
    print(f"  输入成本  ${stats.input_cost:>7.4f}")
    if stats.storage_cost:
        print(f"  存储成本  ${stats.storage_cost:>7.4f}")
    print(f"  无缓存成本 ${baseline:>7.4f}  ->  节省 {savings*100:>5.1f}%")
    print(f"  每千次请求 ${stats.total_cost * 1000 / n:>7.4f}")


def main() -> None:
    traffic = make_traffic(n_requests=500, n_prefixes=3, prefix_size=15000, suffix_size=400)
    seconds_between = 4  # 每4秒请求一次

    anthro_5m = simulate_anthropic(traffic, ttl_seconds=300, seconds_between=seconds_between)
    anthro_1h = simulate_anthropic(traffic, ttl_seconds=3600, seconds_between=seconds_between)
    openai = simulate_openai(traffic, seconds_between=seconds_between)
    gemini = simulate_gemini(traffic, ttl_seconds=3600, seconds_between=seconds_between)

    print("场景：500 次请求，轮换使用 3 个前缀（每个 15K token），间隔 4 秒\n")

    print_report("Anthropic Claude Opus 4.7（TTL 5 分钟）", anthro_5m, baseline_cost(traffic, "anthropic_claude_opus_4_7"), len(traffic))
    print_report("Anthropic Claude Opus 4.7（TTL 1 小时）", anthro_1h, baseline_cost(traffic, "anthropic_claude_opus_4_7"), len(traffic))
    print_report("OpenAI GPT-5（自动缓存）", openai, baseline_cost(traffic, "openai_gpt_5"), len(traffic))
    print_report("Gemini 3 Pro（显式缓存，1 小时）", gemini, baseline_cost(traffic, "gemini_3_pro"), len(traffic))


if __name__ == "__main__":
    main()
