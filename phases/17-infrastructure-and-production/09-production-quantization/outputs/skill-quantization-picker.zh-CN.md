---
name: quantization-picker
description: 根据硬件、引擎、工作负载和质量容忍度选择 2026 年的量化格式，并产出校准与验证计划。
version: 1.0.0
phase: 17
lesson: 09
tags: [quantization, awq, gptq, gguf, fp8, nvfp4, calibration]
---

给定硬件（CPU / H100 / H200 / B200 / GB200 及数量）、引擎（llama.cpp / vLLM / TRT-LLM / SGLang）、模型（规模 + 任务类型——常规对话 / 推理 / 代码 / 多 LoRA）和质量容忍度（可接受 HumanEval / MATH / MMLU 上 N 个百分点的下降），选择一种量化格式并产出验证计划。

产出内容：

1. 格式推荐。以下之一：GGUF Q4_K_M、GGUF Q5_K_M、GPTQ-Int4 + Marlin、AWQ-Int4 + Marlin、FP8、NVFP4 + FP8 KV 或叠加组合。按决策树论证：CPU → GGUF；推理 → FP8；vLLM 上多 LoRA → GPTQ；常规 GPU 对话 → AWQ；Blackwell 已验证 → NVFP4。
2. 显存预算。报告权重 + KV 缓存（按报告并发 × 上下文）+ 激活值。确认是否适配目标 GPU，或指出多 GPU 需求。
3. 校准计划。数据集来源（AWQ/GPTQ 使用领域匹配数据；通用 C4/WikiText 作为最后手段）。样本数量（领域数据 500-2000）。验证集（从校准池中留出 10%）。
4. 验证计划。匹配任务的评估集：代码用 HumanEval、推理用 MATH/MMLU、对话用 MT-Bench。BF16 基线对比量化版本。下降幅度 ≤ 质量容忍度时上线。
5. KV 缓存决策。与权重量化分离。推理场景推荐 FP8 KV；注意力精度处于边际水平时使用 BF16 KV；INT8 KV 仅在验证后使用。
6. 回退路径。在磁盘上保留 BF16/FP8 权重；标记在生产质量退化时切回。

硬性拒绝条件：
- 在推理密集型工作负载上推荐 NVFP4 权重而未包含评估集验证。
- 对领域模型使用通用网络数据做校准。必须使用领域内数据。
- 在 HBM 预算中遗漏 KV 缓存。必须逐项列明。
- 在不指明所用内核的情况下声称吞吐量数字（Marlin-AWQ 与普通 AWQ 相差 10 倍）。

拒绝规则：
- 如果工作负载本身质量处于边际水平（开放式创意生成、边缘情况推理），拒绝激进 INT4。保持 FP8 或 BF16。
- 如果引擎为 llama.cpp，拒绝除 GGUF 外的任何格式。格式与引擎匹配是基本要求。
- 如果用户无法运行 1,000 样本评估，拒绝。生产环境不允许盲量化。

输出：一份单页量化选择，列出所选格式、HBM 预算、校准计划、验证计划、KV 缓存决策和回退路径。以一段"下一步测量什么"收尾，根据关键风险指出评估集差值、峰值并发下的 KV 缓存压力或真实 batch size 下的吞吐量中的某一项。
