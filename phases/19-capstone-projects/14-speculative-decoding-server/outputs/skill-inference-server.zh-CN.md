---
name: inference-server
description: 上线一个投机解码推理服务器，使用 EAGLE-3 或 P-EAGLE 草稿、K8s 自动伸缩，并附带完整的吞吐量/延迟/成本报告。
version: 1.0.0
phase: 19
lesson: 14
tags: [capstone, inference, vllm, sglang, eagle-3, p-eagle, speculative-decoding, quantization, hpa]
---

给定两个开放目标模型（Llama 3.3 70B 和 Qwen3-Coder-30B MoE 或 GPT-OSS-120B），上线一个具备投机解码、量化和 Kubernetes 自动伸缩的生产服务栈。发布测得的加速比和尾延迟数据。

构建计划：

1. 在 vLLM 0.7（或 SGLang 0.4）下部署目标模型，使用 FP8 Marlin 量化。
2. 从 Red Hat Speculators 加载一个已对齐的 EAGLE-3 草稿（或通过 SpecForge 训练一个）。
3. 基线数据：在批大小 1/8/32 下、不使用投机解码时的 tokens/s 和 p50/p99 延迟。
4. 启用 EAGLE-3。重新运行同一基准测试。报告加速比、接受率、p99 尾延迟变化量。
5. 启用 P-EAGLE 并行投机；报告更深的树结构有利与有害的拐点。
6. 跨数据分布运行基准测试：ShareGPT、HumanEval、领域数据。发布接受率漂移。
7. 在第二个目标模型（MoE）上重复；识别草稿接受中的路由噪声敏感性。
8. 在 Kubernetes 上部署，使用 HPA 追踪 `queue_wait_ms`。演示负载三倍时的扩容。
9. 在匹配的评估上与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 比较 $/1M tokens。

评估标准：

| 权重 | 标准 | 度量 |
|:-:|---|---|
| 25 | 相对基线的实测加速 | 在两个模型上、匹配质量下吞吐量达 2.5x+ |
| 20 | 真实流量下的接受率 | 按数据分布的接受率报告 |
| 20 | P99 尾延迟管控 | 批大小 1/8/32 下有无投机解码的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、平滑发布、先排空后升级 |
| 15 | 写作与方法论 | 指标的清晰推导、匹配的基线 |

硬性否决项：

- 报告稳态吞吐量而不报告尾延迟。
- 基于 CPU 而非队列等待的 HPA。在 GPU 饱和时会抖动。
- 忽略草稿与目标版本的对齐。漂移的草稿比不投机代价更大。
- 省略托管 API 提示缓存折扣的成本比较。

拒绝规则：

- 拒绝在无发布排空的情况下提供服务。在有请求进行中时原地升级属于一票否决。
- 拒绝报告跨数据分布聚合的接受率。按分布报告是强制性的。
- 拒绝在没有匹配的非投机数值的情况下声称在 bs=32 下投机解码获胜。

输出：一个代码仓库，包含 vLLM / SGLang 配置、EAGLE-3 草稿下载脚本、K8s 部署清单、基于队列等待的 HPA 配置、ShareGPT / HumanEval / 领域数据的基准测试框架、$/1M tokens 比较表，以及一篇写明投机解码引入的三个尾延迟回退及各自修复措施（批处理门控、ngram 回退、量化调整）的报告。
