# 综合项目 14——推测解码推理服务器

> 推测解码已经是成熟的生产优化手段，不再只是研究技巧：一个成本较低的草稿模型（draft model）先提出词元，再由目标模型（target model）一次完成验证。vLLM 0.7 中的 EAGLE-3 在真实流量上可实现 2.5～3 倍吞吐量。P-EAGLE（AWS，2026）又把并行推测推进了一步。SGLang 的 SpecForge 可以大规模训练草稿头，Red Hat 的 Speculators Hub 为常见开放模型发布了对齐的草稿模型，TensorRT-LLM 则把推测解码做成了 NVIDIA 平台的原生能力。2026 年的生产服务栈通常采用 vLLM 或 SGLang，搭配 EAGLE 系列草稿模型、FP8 或 INT4 量化，并依据队列等待时间通过 HPA 自动扩缩容。本综合项目要部署两个开放模型，使吞吐量达到基线的 2.5 倍以上，并给出完整的尾延迟报告。

**Type:** 综合项目
**Languages:** Python（服务）、C++ / CUDA（内核检查）、YAML（配置）
**Prerequisites:** 第 3 阶段（深度学习）、第 7 阶段（Transformer）、第 10 阶段（从零构建 LLM）、第 17 阶段（基础设施）
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 小时

## 问题

到 2026 年，推测解码已经成为标准能力。EAGLE-3 草稿头在目标模型的隐藏状态上训练，可以一次向前预测 N 个词元；目标模型随后通过单次前向传播验证它们。60%～80% 的接受率通常能换来 2～3 倍的端到端吞吐量。vLLM 0.7 原生集成了这一能力，SGLang 与 SpecForge 提供训练流水线，Red Hat 的 Speculators 则为 Llama 3.3 70B、Qwen3-Coder-30B MoE 和 GPT-OSS-120B 发布了对齐的草稿模型。

真正的技术难点在服务运维，而不在模型本身。接受率会随流量分布（ShareGPT、代码或领域数据）漂移；推测词元被拒绝时，尾延迟可能比关闭推测解码还高。因此必须报告多种批大小下的 p99，不能只看稳态的每秒词元数。还要把每百万词元的成本与 Anthropic、OpenAI API 对比，才能判断方案是否真的划算。

## 概念

推测解码分为两层。**草稿模型**可以是 EAGLE-3 草稿头、ngram，也可以是与目标模型对齐的小模型；它每一步提出 k 个候选词元。**目标模型**一次验证全部 k 个词元，被接受的前缀直接替代原来的贪心解码路径。接受率取决于草稿模型与目标模型的对齐程度，也取决于输入分布。

对大多数流量而言，EAGLE-3 都优于 ngram 草稿方案。P-EAGLE 通过并行推测构建更深的草稿树，但代价是拒绝时的 p99 延迟更高，因为一次验证需要处理更多候选。服务配置必须按批大小分桶报告延迟，才能看清这项取舍。

部署环境使用 Kubernetes。vLLM 0.7 在每块 GPU 或每个张量并行分片上运行一个副本。HPA 根据队列等待时间而不是 CPU 使用率自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化可以把显存占用控制在 H100 / H200 的容量范围内。端到端报告要包含吞吐量、接受率、批大小为 1、8、32 时的 p50 / p99，以及每百万词元的美元成本。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 推理服务：vLLM 0.7 或 SGLang 0.4
- 推测方法：EAGLE-3 草稿头、P-EAGLE 并行推测、ngram 后备方案
- 草稿模型训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA Device Plugin；HPA 使用队列等待时间指标
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于测量跨领域分布的接受率
- 参考：TensorRT-LLM 推测解码，作为厂商基线

```figure
cf-spec-decode
```

## 动手构建

1. **准备目标模型。** 选择 Llama 3.3 70B，通过 Marlin 量化为 FP8。在 vLLM 0.7 下使用 1×H100 部署，也可以采用 2 路张量并行。

2. **准备草稿模型。** 从 Red Hat Speculators 获取对齐的 EAGLE-3 草稿头，或通过 SpecForge 自行训练，然后把它加载到 vLLM 的推测解码配置中。

3. **测量基线。** 启用推测前，测量批大小为 1、8、32 时的每秒词元数、p50 / p99 延迟与 GPU 利用率，并公开结果。

4. **启用 EAGLE-3。** 切换配置，重新运行同一基准测试，报告加速比、接受率和 p99 尾延迟变化。

5. **评估 P-EAGLE。** 启用并行推测，比较更深的草稿树与串行 EAGLE-3。报告 P-EAGLE 从带来收益转为造成损失的转折点。

6. **测试不同领域的流量。** 让 ShareGPT、HumanEval 与领域特定流量通过同一服务器，测量每种分布的接受率，并识别草稿模型开始漂移的时机。

7. **测试第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行同一流水线。MoE 路由噪声会增加草稿预测的难度，应如实报告结果。

8. **K8s HPA。** 在 K8s 下部署，并让 HPA 跟踪 `queue_wait_ms`。演示负载增长三倍时的水平扩容。

9. **成本比较。** 在同一评估上计算每百万词元的美元成本，并与 Anthropic Claude Sonnet 4.7、OpenAI GPT-5.4 比较后公开结果。

## 运行示例

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付成果

`outputs/skill-inference-server.md` 描述最终交付物：一套经过实测、支持推测解码的推理服务栈，一份完整的基准报告，以及 K8s 部署。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 相对基线的实测加速 | 两个模型在质量匹配时达到 2.5 倍以上吞吐量 |
| 20 | 真实流量接受率 | 按分布报告接受率 |
| 20 | p99 尾延迟控制 | 开启和关闭推测时，批大小为 1、8、32 下的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待时间的 HPA、平滑发布 |
| 15 | 文档与方法 | 清楚说明改变了什么以及原因 |
| **100** | | |

## 练习

1. 当草稿模型比目标模型落后一个版本时（例如 Llama 3.3 -> 3.4 漂移），测量接受率下降幅度，并配置监控告警。

2. 实现 ngram 后备方案：EAGLE-3 接受率低于阈值时，切换到 ngram 草稿方案，并报告可靠性提升。

3. 运行受控 MoE 实验：对同一个 Qwen3-Coder-30B 分别注入和不注入路由噪声，测量草稿接受率对此的敏感程度。

4. 扩展到 H200（141 GB）。报告单个副本可容纳的模型规模增加了多少，以及能否部署未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上对 TensorRT-LLM 推测解码进行基准测试，报告它在哪些方面优于 vLLM。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 草稿模型 | “推测器” | 提出 N 个词元供目标模型验证的小模型 |
| EAGLE-3 | “2026 年草稿架构” | 在目标模型隐藏状态上训练的草稿头；接受率约 75% |
| P-EAGLE | “并行推测” | 在目标模型的一次前向传播中验证一棵草稿分支树 |
| 接受率 | “命中率” | 无须重新采样即可接受的草稿词元比例 |
| 量化 | “FP8 / INT4” | 使用低精度权重，让 GPU 内存容纳更多模型参数 |
| 队列等待 | “HPA 指标” | 请求在推理开始前于待处理队列中等待的时间 |
| Speculators Hub | “对齐草稿模型” | Red Hat Neural Magic 为常见开放模型提供的 EAGLE 草稿模型中心 |

## 延伸阅读

- [vLLM EAGLE 与 P-EAGLE 文档](https://docs.vllm.ai)——参考推理服务栈
- [P-EAGLE（AWS，2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)——并行推测解码论文与集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge)——草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators)——对齐草稿模型中心
- [TensorRT-LLM 推测解码](https://nvidia.github.io/TensorRT-LLM/)——厂商替代方案
- [Fireworks.ai Serving 架构](https://fireworks.ai/blog)——商业参考实现
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840)——方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm)——代码与基准测试
