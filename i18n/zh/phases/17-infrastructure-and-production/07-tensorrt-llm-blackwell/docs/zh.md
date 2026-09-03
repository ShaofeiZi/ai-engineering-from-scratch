# 面向硬件特化的推理编译：Blackwell 上的 FP8 与 NVFP4

> 硬件特化推理编译，本质上是在用可移植性换吞吐量，而 TensorRT-LLM 就是这笔交易最清楚的例子。它只支持 NVIDIA，并专门针对 Blackwell 调优。SemiAnalysis InferenceX 在 2026 年 Q1-Q2 的测量里给出了一组非常直观的数字：在 GB200 NVL72 + Dynamo 编排下，一个 120B 模型的成本约为每百万 token $0.012；而在 H100 + vLLM 上，同类负载约为 $0.09/M，经济性差距接近 7x。这套优势来自三种浮点精度模式的叠加：FP8 仍然是 KV cache 和 attention kernel 的关键，因为它们需要更大的动态范围；NVFP4（4-bit microscaling）负责 weights 和 activations；再往上叠加 multi-token prediction（MTP）和 disaggregated prefill/decode，又能多拿 2-3x。它还支持首日适配新模型，直接加载 FP4 weights，不需要事后再做 post-training conversion。对 2026 年的工程团队来说，真正需要权衡的是：TRT-LLM 虽然开源，但它本质上仍只适用于 NVIDIA，并围绕 CUDA 和 Blackwell 展开。采用这套栈，就等于明确接受用可移植性换吞吐量。做决定前，必须先把自己的模型组合与硬件组合算清楚。

**Type:** 学习
**Languages:** Python（标准库，玩具级 FP8/NVFP4 内存与成本计算器）
**Prerequisites:** 第 17 阶段 · 04（服务引擎内部原理），第 10 阶段 · 13（量化）
**Time:** 约 75 分钟

## 学习目标

- 解释为什么即使 weights 已经用 NVFP4，FP8 依旧是 KV cache 与 attention 的下限精度。
- 计算 frontier model 在 BF16、FP8、NVFP4 下的 HBM footprint，并分析节省到底来自哪里。
- 说出 TRT-LLM 在 Blackwell 上利用的关键特性：day-0 FP4、MTP、disaggregated serving、all-to-all primitives。
- 判断在什么场景下，TRT-LLM 的 NVIDIA lock 值得你为 7x 的成本差距买单，而不是继续用 Hopper + vLLM。

## 问题

2026 年推理经济学真正关心的是“每一美元能换来多少 token”。而答案并不只由模型本身决定，它取决于四层叠加选择：硬件代际（Hopper H100/H200 还是 Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、serving engine（vLLM、SGLang、TRT-LLM），以及 orchestration（plain、disaggregated、Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 负载大约是每百万 token $0.09。到了 Blackwell + TRT-LLM + Dynamo，同一个模型可以做到约 $0.012，便宜约 7x。这部分差距来自硬件本身，Blackwell 的单 GPU LLM throughput 相比 Hopper 可高 11-15x；另一部分则来自软件与精度栈：FP4 weights、MTP draft、disaggregated prefill/decode，以及专门为 MoE expert communication 调优的 NVLink 5 all-to-all。

你不可能在 NVIDIA 以外的栈里完整复制这套收益。这里真正的权衡就是：用 portability 换 inference economics。这一课的重点，就是把“7x 差距”拆开，理解到底是哪几层贡献了多少。

## 概念

### 为什么 FP8 仍然是 KV cache 的下限

2026 年一个很常见的误解是：既然 Blackwell 有 NVFP4，那是不是所有地方都应该用 FP4？答案是不行。KV cache 仍然需要 FP8（8-bit floating point），因为 attention keys 和 values 的动态范围很宽。把 KV 量化到 FP4，会造成非常明显的精度灾难，分布尾部会被截断，attention score 直接塌掉。FP8 的 exponent bits，正是 KV cache 需要的动态范围保障。

NVFP4（2025-2026）真正适用的是 weights 和 activations。它依赖 microscaling：每个小 block 的 weights 都有自己的 scale factor，所以不同小块可以覆盖不同的动态范围，而不会像 per-tensor scale 那样把细节全压平。activations 之所以也能用 FP4，是因为它们通常在单层内部的动态范围较小。

典型的 Blackwell 配置是：

- Weights：NVFP4（4-bit microscaling）
- Activations：NVFP4
- KV cache：FP8
- Attention accumulator：FP32（保证 softmax 稳定性）

### TRT-LLM 在 Blackwell 上利用的特化能力

- **Day-0 FP4 weights**：模型提供商直接发布 FP4 权重；TRT-LLM 可以直接加载，不需要额外做 post-training conversion，也不用再走 AWQ / GPTQ 这类流程。
- **Multi-token prediction (MTP)**：思路与 EAGLE（Phase 17 · 05）类似，但直接内建在 TRT-LLM build 里。
- **Disaggregated serving**：把 prefill 和 decode 放在不同 GPU 池上，通过 NVLink 或 InfiniBand 传输 KV cache。这与 Dynamo（Phase 17 · 20）的思路一致。
- **All-to-all 通信原语**：NVLink 5 把 MoE expert communication latency 相比 Hopper 降低了 3x，TRT-LLM 的 MoE kernels 就是围绕这个特性调优的。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Cores 原生加速 scale-factor 处理，让 microscaling 成本不再只是软件开销。

### 你必须记住的数字

- HGX B200 上，TRT-LLM 跑 GPT-OSS-120B，成本可到 $0.02/M tokens。
- GB200 NVL72 + Dynamo（编排 TRT-LLM），可到 $0.012/M tokens。
- H100 + vLLM 在可比负载上约为 $0.09/M tokens。
- 2026 年 TRT-LLM 在三个月迭代里拿到了 2.8x throughput gain。
- Blackwell 相比 Hopper，单 GPU LLM throughput 大约高 11-15x。
- MLPerf Inference v6.0（2026 年 4 月）里，Blackwell 在所有提交任务上都占优。

### FP4 真正付出的质量代价

NVFP4 很激进。在 reasoning-heavy workload 上，例如 chain-of-thought、math、带长上下文的 code-gen，FP4 weights 的质量下降会非常明显。每 block 校准可以缓解，但并不能彻底消除。很多团队在交付 reasoning model 时，会退一步选择 FP8 weights + FP4 activations 作为折中，或者干脆继续用 H200，并在整个路径里都坚持 FP8。

规则很简单：在承诺上 NVFP4 weights 之前，必须先在自己的 eval set 上验证任务质量。

### 为什么这本质上是在权衡 NVIDIA 锁定

TRT-LLM 是 C++ + CUDA + 闭源特化 kernel 的世界。模型需要为特定 GPU SKU 编译。没有 AMD，没有 Intel，也没有 ARM。如果你的 infra strategy 是 multi-vendor，那 TRT-LLM 对那个 serving tier 来说基本就是 non-starter；你仍然可以在混合硬件上用 vLLM。只有当你明确是 NVIDIA-only 时，7x 的经济差距才足以覆盖这层 lock-in 成本。

### 2026 年的实用配方

如果你的 annual inference bill 已经是 $100M+ 级别，继续跑 Hopper + vLLM，等于把 7-10x 的成本空间放在桌上不拿。更合理的做法是：把 cost-dominant workload 迁到 Blackwell + TRT-LLM + Dynamo；把实验与快速迭代层继续留在 H100 + vLLM 上，换取更高的模型开发灵活性。每个要生产化的 NVFP4 model，都先单独过质量验证。

### 解耦式服务带来的额外倍增

TRT-LLM 的 disaggregated serving（prefill 与 decode 分池）会在 Phase 17 · 20 里深入展开。在 Blackwell 上，这个 multiplier 是可以叠乘的：FP4 weights × MTP speedup × disaggregated placement × cache-aware routing。你看到的 7x 数字，默认就是在这整套 full stack 下成立的。

```figure
pipeline-parallel
```

## 动手用

`code/main.py` 会计算一个模型在三套 stack 下的 HBM footprint、decode throughput（memory-bound regime），以及每百万 token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，你可以看到 compounding effect，以及每一次栈切换分别贡献了多少差距。

## 交付物

这一课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定 workload、model size 和 annual token volume，它会判断 Blackwell + TRT-LLM 这套栈是否值得你接受 NVIDIA lock。

## 练习

1. 跑 `code/main.py`。对一个 active parameters 为 30% 的 120B MoE，计算 H100 BF16、H100 FP8、B200 NVFP4/FP8 三种情况下 memory-bandwidth-limited 的 decode throughput。最大跃升来自哪里？
2. 某客户每年在 H100 + vLLM 上花 $2M。考虑到 7x 的经济差距，他们需要采购多少 Blackwell GPU，才能在 12 个月内把迁移到 TRT-LLM 的成本摊平？
3. 你在 NVFP4 weight conversion 后看到 MATH 分数掉了 3 个点。给出两条恢复路径：一条 quality-first（保留 FP8 weights），一条 cost-first（用域内数据重做校准）。
4. 阅读 MLPerf v6.0 inference results。哪类任务的 Blackwell-over-Hopper 差距最小？为什么？
5. 计算一个 405B 模型在 NVFP4 weights + FP8 KV cache、128k context 下所需的 HBM。它能否装进单个 GB200 NVL72 节点？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| FP8 | “8 位浮点” | 8-bit floating point；由于动态范围需求，常用于 KV cache 和 attention |
| NVFP4 | “4 位微缩浮点” | NVIDIA 的 4-bit microscaling FP 格式；Blackwell 上用于 weights 和 activations |
| MXFP8 | “MX 8 位浮点” | 一种 microscaling FP8 变体；可由 Blackwell Tensor Cores 硬件加速 |
| Day-0 FP4 | “直接交付 FP4 权重” | 模型提供商直接发布 FP4 权重，无需 post-train conversion |
| MTP | “多 token 预测” | TRT-LLM 集成的 speculative-decoding draft，见 Phase 17 · 05 |
| Disaggregated serving | “拆分 prefill/decode” | prefill 与 decode 分别跑在不同 GPU 池上，KV 通过 NVLink/IB 传输 |
| All-to-all | “MoE expert 通信” | 把 token 路由到 expert GPU 的通信模式；NVLink 5 可把延迟降 3x |
| InferenceX | “SemiAnalysis 推理基准” | 2026 年行业认可的 cost-per-token 基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果
- [NVIDIA — Blackwell 上的 MoE 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 与 MoE kernels
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的 disaggregated orchestration
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 基准数据的行业 benchmark suite
