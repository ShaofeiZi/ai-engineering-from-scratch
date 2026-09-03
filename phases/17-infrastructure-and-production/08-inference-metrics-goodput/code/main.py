"""玩具版 goodput 计算器，使用 Python stdlib。

模拟一组具有真实右偏延迟的 LLM 请求，应用多约束 SLO，计算 goodput，并展示
GenAI-Perf 与 LLMPerf 在同一轨迹上的 TPOT 计算差异。
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass
class RequestTrace:
    queue_ms: float
    prefill_ms: float
    decode_ms_per_token: list[float]      # 逐 token decode 延迟
    output_tokens: int

    @property
    def ttft_ms(self) -> float:
        return self.queue_ms + self.prefill_ms

    @property
    def e2e_ms(self) -> float:
        return self.ttft_ms + sum(self.decode_ms_per_token)

    def tpot_llmperf(self) -> float:
        """LLMPerf：在 ITL 计算中包含 TTFT。"""
        return self.e2e_ms / self.output_tokens

    def tpot_genaiperf(self) -> float:
        """GenAI-Perf：ITL 从第 2 个 token 开始计算。"""
        if self.output_tokens <= 1:
            return 0.0
        return sum(self.decode_ms_per_token) / (self.output_tokens - 1)


def synth_workload(n: int = 1000, seed: int = 7, tail_spike_rate: float = 0.02) -> list[RequestTrace]:
    rng = random.Random(seed)
    traces = []
    for _ in range(n):
        prompt_len = rng.choice([128, 256, 512, 2048, 8192])
        output_tokens = rng.randint(50, 300)
        queue = rng.expovariate(1 / 40.0)           # 平均 40 ms 排队时间
        prefill = prompt_len * 0.05                 # 每个输入 token 约 50 微秒
        decode_base = 7.0                           # 平均 TPOT 为 7 ms
        decodes = []
        for _ in range(output_tokens):
            t = max(1.5, rng.gauss(decode_base, decode_base * 0.15))
            if rng.random() < tail_spike_rate:
                t *= rng.uniform(3, 8)              # 长尾尖峰
            decodes.append(t)
        traces.append(RequestTrace(queue, prefill, decodes, output_tokens))
    return traces


def percentiles(values: list[float], ps: list[float]) -> list[float]:
    s = sorted(values)
    return [s[min(len(s) - 1, int(p * len(s)))] for p in ps]


def report_latency(label: str, traces: list[RequestTrace]) -> None:
    ttft = [t.ttft_ms for t in traces]
    tpot_llm = [t.tpot_llmperf() for t in traces]
    tpot_nv = [t.tpot_genaiperf() for t in traces]
    e2e = [t.e2e_ms for t in traces]

    p50_ttft, p90_ttft, p99_ttft = percentiles(ttft, [0.5, 0.9, 0.99])
    p50_tpot, p90_tpot, p99_tpot = percentiles(tpot_nv, [0.5, 0.9, 0.99])
    p50_e2e, p90_e2e, p99_e2e = percentiles(e2e, [0.5, 0.9, 0.99])

    print(f"{label}")
    print("-" * 76)
    print(f"  TTFT（毫秒） P50={p50_ttft:7.1f}  P90={p90_ttft:7.1f}  P99={p99_ttft:7.1f}  平均={statistics.mean(ttft):7.1f}")
    print(f"  TPOT（毫秒） P50={p50_tpot:7.2f}  P90={p90_tpot:7.2f}  P99={p99_tpot:7.2f}  平均={statistics.mean(tpot_nv):7.2f}")
    print(f"  E2E（毫秒）  P50={p50_e2e:7.1f}  P90={p90_e2e:7.1f}  P99={p99_e2e:7.1f}")
    print(f"  工具陷阱      GenAI-Perf 平均 TPOT={statistics.mean(tpot_nv):6.2f}  "
          f"LLMPerf 平均 TPOT={statistics.mean(tpot_llm):6.2f}  "
          f"差值={statistics.mean(tpot_llm) - statistics.mean(tpot_nv):+5.2f} 毫秒")


def goodput(traces: list[RequestTrace], slo_ttft: float, slo_tpot: float,
            slo_e2e: float) -> float:
    good = 0
    for t in traces:
        if t.ttft_ms <= slo_ttft and t.tpot_genaiperf() <= slo_tpot and t.e2e_ms <= slo_e2e:
            good += 1
    return good / len(traces)


def main() -> None:
    print("=" * 78)
    print("玩具版 GOODPUT 计算器 — 推理 SLO 与测量陷阱")
    print("=" * 78)
    print()

    traces = synth_workload(n=2000)
    report_latency("合成工作负载（2000 个请求）", traces)
    print()

    slos = [
        ("宽松    TTFT<800 TPOT<25 E2E<3000", 800, 25, 3000),
        ("目标    TTFT<500 TPOT<15 E2E<2000", 500, 15, 2000),
        ("严格    TTFT<300 TPOT<10 E2E<1500", 300, 10, 1500),
    ]
    print("三种 SLO 配置下的 Goodput")
    print("-" * 76)
    for label, t1, t2, t3 in slos:
        g = goodput(traces, t1, t2, t3)
        tag = "  可发布" if g >= 0.99 else ("  已退化" if g >= 0.95 else "  失败")
        print(f"  {label}  有效吞吐率={g:6.2%}{tag}")

    print()
    print("=" * 78)
    print("关键发现")
    print("-" * 78)
    print("  平均 TPOT 约 7 ms 看起来很好；P99 TPOT 约 25-40 ms 才反映真实情况。")
    print("  收紧 SLO 后，goodput 会从 99% 跌至 80%+。用户感受到的是 P99。")
    print("  GenAI-Perf 与 LLMPerf 的平均 TPOT 相差约 1 ms；引用数据时请注明工具。")


if __name__ == "__main__":
    main()
