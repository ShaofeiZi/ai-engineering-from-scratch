# 推理指标：TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一套推理部署是否真正可用。TTFT 是 prefill、queue 与 network 的总和。TPOT（等价地也可视作 ITL）是每个 token 的 memory-bound decode 成本。端到端延迟等于 TTFT 加上 TPOT 乘以输出长度。Throughput 是整个 fleet 聚合后的 tokens per second。但对产品真正关键的是 goodput，也就是同时满足所有 SLO 的请求占比。高 throughput、低 goodput，意味着你处理了很多 token，但它们并没有按时送达用户。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数字是：mean TTFT 162 ms、mean TPOT 7.33 ms、mean E2E 1,093 ms。任何时候都该报告 P50、P90、P99，而不是只看 mean。还要警惕一个测量陷阱：GenAI-Perf 在 ITL 计算中排除了 TTFT，而 LLMPerf 把它算了进去；同一次运行，两套工具会给出不同的 TPOT 数值。

**Type:** 学习
**Languages:** Python（标准库，玩具级百分位数计算器与 goodput 报告器）
**Prerequisites:** 第 17 阶段 · 04（服务引擎内部原理）
**Time:** 约 60 分钟

## 学习目标

- 准确定义 TTFT、TPOT、ITL、E2E、throughput 与 goodput，并说出每个指标具体衡量什么。
- 解释为什么 mean 不是 LLM serving 的正确统计量，以及应当如何解读 P50/P90/P99。
- 构造一个多约束 SLO，例如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s，并据此计算 goodput。
- 说出两种会对同一次运行得出不同 TPOT 结果的 benchmark 工具，并解释原因。

## 问题

“我们的 throughput 是 15,000 tokens per second。”那又怎样？如果 40% 的请求端到端延迟超过 2 秒，用户早就放弃这次会话了。Throughput 本身并不能说明产品是否可用。

推理有多条延迟轴线，而且每条轴线的失效原因都不同。Prefill 是 compute-bound，并随 prompt length 增长。Decode 是 memory-bound，并随 batch size 扩张。Queueing delay 是运维问题。Network 是物理距离问题。你需要为这些问题分别定义指标，需要看 percentiles，还需要一个统一的复合指标来回答“用户是否按预期拿到了结果”，这就是 goodput。

## 概念

### TTFT — 首 token 时间

`TTFT = queue_time + network_request + prefill_time`

当 prompt 很长时，prefill 往往占主导。以 H100 上的 Llama-3.3-70B FP8 为例，一个 32k prompt 仅 prefill 就需要约 800 ms。Queue time 是负载下的调度器行为。Network request 是包括 TLS 在内的链路时间。TTFT 就是用户在任何内容开始回流之前感受到的等待时长。

### TPOT / ITL：词元间延迟

一个量，多个名字。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token` 实际上都指同一件事：第一个 token 之后，相邻流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

仍以同一套 Llama-3.3-70B H100 栈为例，启用 chunked prefill 时，TPOT mean 约为 7 ms；如果邻近序列正在执行一个很长的 prefill，且你没有启用 chunked prefill，TPOT 可能飙到 50 ms。盯住 P99，不要只看 mean。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 tokens），E2E 往往由 TPOT 主导。对于输出较短但 prompt 很长的请求，E2E 往往由 TTFT 主导。报告 E2E 时，应按输出长度分桶。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

这是一个聚合指标。它告诉你 fleet efficiency，却无法告诉你单个请求是否健康。

### Goodput — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是一个多约束条件。只有所有约束都满足时，请求才算“good”。Goodput 就是这类请求的占比。60% goodput 的高 throughput 是失败。99% goodput 的较低 throughput 才是目标。

到 2026 年，goodput 已成为 MLPerf Inference v6.0 提交中的核心指标，也被 AI 平台提供商用于内部 SLA 跟踪。

### 为什么 mean 是错误统计量

LLM 延迟分布通常是右偏的。一个 decode batch 可能有 500 个 token 的 TPOT 约为 7 ms，同时还有 20 个 token 因为碰上长 prefill 邻居而达到 60 ms。于是 mean TPOT 看起来只有 9 ms，而 P99 TPOT 却可能是 65 ms。用户经常遭遇的正是 P99，这就是他们离开的原因。

始终报告三元组：P50、P90、P99。对用户体验而言，真正需要优化的是 P99。

### 参考数字 — 2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT: 视 chunked-prefill 配置而定，在 10-25 ms 间变化。

这些是 NVIDIA 发布的参考点。它们会随模型大小（70B 往往高 3-5 倍）、硬件代际（H100 与 B200 可能相差约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两种 benchmark 工具，对同一次运行的 TPOT 会给出不同定义：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从 token 2 开始计算。
- **LLMPerf**：将 TTFT 算入 ITL。ITL 从 token 1 开始计算。

对于一个 TTFT 为 500 ms、100 个输出 token、总 decode 时间为 700 ms 的请求，GenAI-Perf 会报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 会报告 `ITL = 1200/100 = 12.00 ms`。工具一换，数字就变了。

必须始终说明使用了哪一种工具，也必须公开给出的定义。

### 构造 SLO

对一个 2026 年的 70B chat model 来说，一个合理的消费级 SLO 可能是：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- 对少于 300-token 的输出，E2E P99 <= 3 s。
- Goodput 目标 >= 99%。

企业级 SLO 往往会收紧 TTFT（200-400 ms），同时放宽 E2E。重点在于把三类指标都写清楚、测出来，并用 goodput 作为统一复合指标持续跟踪。

### 如何测量

- 运行真实流量，或足够逼真的 synthetic 流量，例如 LLMPerf 配合 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`
- benchmark 运行的目标并发应达到峰值并发的 2 倍
- 运行 30-50 次，对合并样本取 percentiles
- 发布结果时写明 tool name、tool version、model、hardware、concurrency、prompt distribution

```figure
throughput-latency
```

## 动手用

`code/main.py` 是一个 toy goodput calculator。它会生成一组 synthetic latency distribution，套用一个 SLO，然后计算 goodput。它还会在同一条 trace 上展示 GenAI-Perf 与 LLMPerf 在 TPOT 计算上的差异。

## 交付物

本课产出 `outputs/skill-slo-goodput-gate.md`。给定一个 workload 与 SLO，它会生成一份可直接用于 CI/CD 的 benchmark recipe，让部署门槛建立在 goodput 上，而不是 throughput。

## 练习

1. 运行 `code/main.py`。生成一个包含 1% tail spike 的分布。当你把 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 会发生什么变化？
2. 某供应商宣称“在 Llama 3.3 70B H100 上达到了 15,000 tok/s”。在相信这个数字之前，你至少要追问哪三个问题？
3. 为什么 chunked prefill 可以保护 P99 TPOT，却未必显著改变 mean TPOT？
4. 为一个 voice assistant 设计消费级 SLO。对这种产品来说，哪个指标最直接影响用户感知？
5. 阅读 LLMPerf README 与 GenAI-Perf 文档。找出另外三个定义不同的指标。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TTFT | “首 token 时间” | Queue + network + prefill；长 prompt 时往往由 prefill 主导 |
| TPOT | “每个输出 token 的耗时” | 第一个 token 之后，每个 token 的 memory-bound decode 成本 |
| ITL | “token 间延迟” | 在大多数工具里与 TPOT 相同（并非全部，见 GenAI-Perf） |
| E2E | “端到端” | TTFT + TPOT * output_len；再加响应方向的 network |
| Throughput | “tok/s” | Fleet efficiency；脱离延迟百分位没有意义 |
| Goodput | “SLO 达标率” | 同时满足所有 SLO 约束的请求占比 |
| P99 | “尾延迟” | 最差 1% 的尾延迟；真正影响用户体验的指标 |
| SLO multi-constraint | “联合约束” | 所有延迟边界的 AND；任一条件违反，请求即失败 |
| GenAI-Perf vs LLMPerf | “工具陷阱” | 工具会在 ITL 是否包含 TTFT 上给出不同定义 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的标准定义。
- [Anyscale — LLM Serving Benchmarking 指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义与测量方法。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署中的测量实践。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源 benchmark。
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) — NVIDIA 的 benchmark 工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业接受的 goodput 基准。
