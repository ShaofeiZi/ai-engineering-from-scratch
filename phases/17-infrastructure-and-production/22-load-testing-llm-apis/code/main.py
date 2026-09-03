"""负载测试反模式演示程序——使用 Python 标准库。

模拟统一提示词如何通过前缀缓存和请求合并夸大报告吞吐量，以及真实分布
如何揭示实际上限。
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import statistics


PREFIX_CACHE_HIT_TTFT_MS = 80
PREFIX_CACHE_MISS_TTFT_MS = 800
TPOT_MS = 15
BATCH_EFFICIENCY_SHARED_PREFIX = 0.8  # 批次占用的槽位减少到 1/0.8 = 1.25 倍


@dataclass
class Request:
    prompt_tokens: int
    prefix_hash: str


def make_uniform_workload(n: int = 500) -> list[Request]:
    return [Request(2000, "single_prefix") for _ in range(n)]


def make_realistic_workload(n: int = 500, seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    prefixes = [f"prefix_{i}" for i in range(80)]
    for _ in range(n):
        prompt = max(50, int(rng.gauss(500, 180)))
        reqs.append(Request(prompt, rng.choice(prefixes)))
    return reqs


def simulate(reqs: list[Request], concurrency: int) -> dict:
    cache: set[str] = set()
    ttft_samples: list[float] = []
    # 按 "concurrency" 大小分组串行处理
    for i in range(0, len(reqs), concurrency):
        batch = reqs[i:i + concurrency]
        unique_prefixes = len({r.prefix_hash for r in batch})
        for r in batch:
            hit = r.prefix_hash in cache
            ttft = PREFIX_CACHE_HIT_TTFT_MS if hit else PREFIX_CACHE_MISS_TTFT_MS
            if not hit:
                cache.add(r.prefix_hash)
            ttft_samples.append(ttft)
    ttft_samples.sort()
    p50 = ttft_samples[len(ttft_samples) // 2]
    p99 = ttft_samples[int(len(ttft_samples) * 0.99) - 1]
    return {
        "n": len(reqs),
        "p50": p50,
        "p99": p99,
        "mean": statistics.mean(ttft_samples),
        "cache_hits": sum(1 for t in ttft_samples if t == PREFIX_CACHE_HIT_TTFT_MS),
    }


def main() -> None:
    print("=" * 95)
    print("提示词同质化陷阱——同一测试框架，不同提示词分布")
    print("=" * 95)

    for concurrency in (10, 50, 200):
        print(f"\n并发数 = {concurrency}")
        header = f"{'工作负载':22}  {'n':>5}  {'TTFT_P50':>9}  {'TTFT_P99':>9}  {'平均值':>7}  缓存命中"
        print(header)
        print("-" * len(header))

        uniform = make_uniform_workload(500)
        u = simulate(uniform, concurrency)
        print(f"{'统一分布':22}  {u['n']:5}  {u['p50']:8.0f}毫秒  {u['p99']:8.0f}毫秒  {u['mean']:6.0f}毫秒  {u['cache_hits']:4}")

        realistic = make_realistic_workload(500)
        r = simulate(realistic, concurrency)
        print(f"{'真实分布':22}  {r['n']:5}  {r['p50']:8.0f}毫秒  {r['p99']:8.0f}毫秒  {r['mean']:6.0f}毫秒  {r['cache_hits']:4}")

    print("\n解读：统一提示词会让端点显得很快，真实提示词才能反映实际表现。")
    print("务必为 LLMPerf 同时设置 --mean-input-tokens 和 --stddev-input-tokens。")


if __name__ == "__main__":
    main()
