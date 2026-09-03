---
name: inference-optimizer
description: 为新的推理部署选择注意力实现、KV 缓存策略、量化和投机解码。
version: 1.0.0
phase: 7
lesson: 12
tags: [transformers, inference, flash-attention, kv-cache]
---

给定一个推理部署场景（模型名称 + 参数量、目标硬件、并发数、最大上下文长度、延迟 SLO、吞吐目标），输出：

1. 推理服务框架。vLLM（默认生产级）、SGLang（每 token 延迟最低）、TensorRT-LLM（NVIDIA 最优）、llama.cpp（边缘/CPU）、MLX（Apple 芯片）。各给一句话理由。
2. 注意力实现。Flash Attention 2（Ampere/Ada 默认）、Flash Attention 3（Hopper）、Flash Attention 4（Blackwell，仅前向）。指定回退方案。
3. KV 缓存。数据类型（fp16 默认，fp8 在支持时启用）、分页式还是连续式、前缀缓存开关、并行采样时的共享 KV。
4. 量化。fp16 / bf16（默认）、int8（仅权重量化）、AWQ / GPTQ / GGUF 用于权重。仅在经过基准测试后才启用激活量化。
5. 额外加速手段。投机解码（EAGLE 2 / Medusa / 草稿模型）、连续批处理（始终开启）、分块预填充（长 prompt 工作负载）、在存在重复 prompt 时启用前缀缓存。

拒绝在训练中使用 Flash Attention 4 —— 它在发布时仅支持前向。拒绝在未对目标任务进行质量影响基准测试的情况下推荐 fp8 KV 缓存。对于任何 70B 以上、未使用 GQA 的模型，标记其在 32K+ 上下文下 KV 缓存不可控。对于任何带有重复系统提示的智能体/工具调用部署，要求前缀缓存必须开启。
