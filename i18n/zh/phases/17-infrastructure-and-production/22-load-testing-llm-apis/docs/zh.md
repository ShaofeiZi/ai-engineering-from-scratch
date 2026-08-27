# LLM API 压测 —— 为什么 k6 和 Locust 会说谎

> 传统压测工具不是为流式响应、可变输出长度、token 级指标或 GPU 饱和设计的。大多数团队会踩进两个坑。第一个是 GIL trap：Locust 在 Python GIL 下做 token 级测量，tokenization 会和高并发请求生成争抢执行时间，结果 tokenization backlog 被算进 inter-token latency，最后你以为是服务端慢，其实是测试客户端成了瓶颈。第二个是 prompt-uniformity trap：循环里一直发同一个 prompt，本质上只测到了 token 分布上的一个点；真实流量的长度和前缀匹配情况要复杂得多。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 来修复这个问题。2026 年的工具格局大致是：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）负责 token 级精度；**k6 v2026.1.0** 加上 **k6 Operator 1.0 GA (Sept 2025)** 可以感知流式指标，并通过 TestRun / PrivateLoadZone CRD 做 Kubernetes 原生分布式压测，很适合做 CI/CD 闸门；Vegeta 适合 Go 生态里的恒定速率打满；Locust 2.43.3 只有配上 LLM-Locust 扩展后才适合流式 LLM 压测。常见负载模式则是 steady-state、ramp、spike 和 soak。

**Type:** 构建
**Languages:** Python（标准库，玩具级真实提示词生成器与延迟收集器）
**Prerequisites:** 阶段 17 · 08（推理指标）、阶段 17 · 03（GPU 自动扩缩容）
**Time:** 约 75 分钟

## 学习目标

- 解释两个反模式：GIL trap 和 prompt-uniformity trap，为什么它们会让通用压测器在 LLM API 上说谎。
- 针对不同目的选择工具：LLMPerf（基准跑分）、k6 + streaming extension（CI gate）、guidellm（大规模 synthetic benchmark）、GenAI-Perf（NVIDIA reference）。
- 设计四种负载模式：steady、ramp、spike、soak，并指出每种模式主要抓什么故障。
- 用 input tokens 的均值与标准差构造真实 prompt 分布，而不是固定长度输入。

## 问题

你用 k6 在 500 个并发用户下测试了自己的 LLM endpoint。它扛住了，于是你上线。结果在生产里，真实用户才 200 个，服务就开始崩：P99 TTFT 暴涨，GPU 被打满。

通常是两件事同时发生。第一，k6 发的是 500 个完全一样的 prompt，而你的 request coalescing 和 prefix caching 让系统看起来像是在处理 500 个并发 decode，实际上只是命中了同一个前缀缓存。第二，k6 对流式响应的 inter-token latency 感知方式，和用户真正感受到的响应体验并不一致；它看到的是一条 HTTP 连接，不是 500 个 token 在不同时间间隔到达的体感差异。

对 LLM 来说，压测本身就是一门单独的学科。

## 概念

### GIL 陷阱（Locust）

Locust 基于 Python，客户端侧 tokenization 会在 GIL 下运行。高并发时，tokenizer 会排在请求生成逻辑后面。这样一来，报告里的 inter-token latency 就混入了客户端 tokenization backlog。你以为是服务端慢，实际上慢的是测试 harness。

修复方式有两个：要么用 LLM-Locust 扩展，把 tokenization 挪到独立进程；要么直接换成编译型语言的 harness，例如 k6，或者使用基于 tokenizers.rs 的 LLMPerf。

### 提示词单一性陷阱

所有常见负载测试器都允许你配置“一个 prompt”。问题在于：当你循环跑 10,000 次时，每次发的都是同一个 prompt。服务端每次看到的都是同一个前缀，于是 prefix cache 命中率接近 100%，吞吐量自然会被测得很好看。

修复方式是：从 prompt 分布中采样。LLMPerf 的典型参数就是 `--mean-input-tokens 500 --stddev-input-tokens 150`，让输入长度和内容都保持多样性。

### 四种负载模式

1. **Steady-state**：在 30-60 分钟内维持恒定 RPS。它主要抓基线性能回归。
2. **Ramp**：在 15 分钟内把 RPS 从 0 线性拉到目标值。它主要抓容量拐点和预热异常。
3. **Spike**：突然把 RPS 放大到 3-10 倍，持续 2 分钟后再退回。它主要抓 autoscaling 延迟、队列饱和和 cold-start 冲击。
4. **Soak**：在稳定负载下连续跑 4-8 小时。它主要抓内存泄漏、连接池漂移和 observability 溢出。

### 2026 工具映射

**LLMPerf**（Anyscale）：
- 虽然入口是 Python，但 tokenization 由 Rust 支撑。
- 支持 mean/stddev prompt 分布。
- 能理解流式指标。
- 是做性能跑分时最稳的默认选择。

**NVIDIA GenAI-Perf**：
- NVIDIA 的参考实现，使用 Triton client，指标覆盖全面。
- 需要注意，它的 ITL 不包含 TTFT，而 LLMPerf 的口径包含 TTFT。
- 所以两者即便测的是同一台服务器，也可能给出不同的 TPOT。

**LLM-Locust**（TrueFoundry）：
- 给 Locust 打的 LLM 扩展。
- 核心价值是修掉 GIL trap。
- 保留熟悉的 Locust DSL，同时补上 streaming metrics。

**guidellm**：
- 更偏向大规模 synthetic benchmarking。

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)**：
- k6 本体用 Go 写成，没有 GIL，并新增了对流式指标的感知。
- k6 Operator 用 TestRun / PrivateLoadZone CRD 做 Kubernetes 原生分布式压测。
- 最适合 CI/CD 闸门和 SLA 回归测试。

**Vegeta**：
- 也是 Go 工具，比 k6 更简单。
- 擅长恒定速率 HTTP 打满。
- 虽然不理解 LLM 特性，但很适合做 gateway 或 rate-limit 层压测。

**Locust 2.43.3 原版**：
- 对 LLM 来说仍然带着 GIL trap。
- 只有加上 LLM-Locust 扩展后才应该使用。

### 在 CI 里做 SLA 闸门

在 PR 上跑 k6 时，常见做法是：

- 按基线 RPS 运行 30-50 次迭代。
- 闸门条件检查 P50/P95 TTFT、5xx < 5%、TPOT 是否低于阈值。
- 任一阈值 breached 就直接让构建失败。

### 更真实的 prompt 分布

最好的输入来源是真实流量样本；如果没有，就退一步使用公开分布，例如聊天场景下的 ShareGPT prompts，或代码场景下的 HumanEval。把均值和标准差喂给 LLMPerf，而不是循环重放同一个 prompt。后者几乎一定会把结果测得过于乐观。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：已支持 streaming-aware metrics。
- 典型 LLMPerf 跑分：100-1000 个请求，并发度为固定 X。
- 典型 CI gate：每个 PR 跑 30-50 次迭代。
- 四类模式：steady、ramp、spike、soak。

```figure
load-pattern-waves
```

## 用起来

`code/main.py` 会模拟一场带真实 prompt 分布的压测，测量有效 TPOT，并演示 uniform-prompt trap 为什么会把结果“测好看”。

## 交付物

这一课会产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA 后，它会帮你选工具，并设计四类负载模式。

## 练习

1. 运行 `code/main.py`。对比 uniform distribution 和 realistic distribution，差距体现在哪？
2. 写一个 CI gate 用的 k6 脚本：要求 TTFT P95 < 800 ms，100 并发，运行 5 分钟。
3. 你的 soak test 显示内存每小时增长 50 MB。请给出三个可能原因，以及如何通过埋点或监控把它们区分开。
4. 做一个从 10 RPS 突增到 100 RPS 的 spike test。如果 Karpenter + vLLM production stack 已经就位（Phase 17 · 03 + 18），合理的恢复时间预期是多少？
5. GenAI-Perf 报告 TPOT=6ms，而 LLMPerf 在同一台服务器上报告 TPOT=11ms。解释为什么这并不一定矛盾。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| LLMPerf | “那个测 LLM 的 harness” | Anyscale 的 benchmark 工具，理解流式指标 |
| GenAI-Perf | “NVIDIA 工具” | NVIDIA 官方参考压测 harness |
| LLM-Locust | “Locust for LLMs” | 修复 GIL trap 的 Locust 扩展 |
| guidellm | “合成压测工具” | 面向大规模 synthetic benchmark 的工具 |
| k6 Operator | “K8s k6” | 基于 CRD 的分布式 k6 运行方式 |
| GIL trap | “Python 客户端开销” | tokenization backlog 抬高了报告中的延迟 |
| Prompt-uniformity trap | “单 prompt 的谎言” | 一直打同一个 prompt 会命中缓存，虚高吞吐 |
| Steady-state | “恒定负载” | 持续 N 分钟的平稳 RPS |
| Ramp | “线性拉升” | 在一段时间内从 0 拉到目标值 |
| Spike | “突发测试” | 瞬时倍增后再回落 |
| Soak | “长时间测试” | 连续数小时运行，用于抓泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
