---
name: moe-configurator
description: 为新的 MoE Transformer 选择专家数量、top-k、负载均衡策略与共享专家布局。
version: 1.0.0
phase: 7
lesson: 11
tags: [transformers, moe, mixture-of-experts, scaling]
---

给定一份 Transformer 规格说明（总参数预算、每个 token 期望的激活参数量、可用训练 token 数、推理硬件），输出：

1. MoE 布局。`n_experts`, `top_k`, `n_shared`。对前沿规模选择细粒度方案（256+ 个专家，top-8）；对较小规模选择经典方案（8 个专家，top-2）。给出一句理由。
2. 负载均衡策略。无辅助损失（DeepSeek-V3，默认）、Switch 风格辅助损失，或专家容量 + token 丢弃。若采用无辅助损失方案，请指明 `γ` 的值。
3. 专家并行方案。在给定 VRAM 的情况下如何跨 GPU 切分专家。说明每个专家的 VRAM 开销与整机集群规模。
4. 路由精度。fp32 路由打分 vs fp16。在大规模下路由精度至关重要。
5. 失效模式检查。指明风险名称：路由坍缩、专家饥饿、全连接网络瓶颈、路由开销导致的推理延迟、检查点内存占用。

拒绝在激活参数量低于 4B 时推荐 MoE —— 在等计算量下稠密模型更优。拒绝在 2026 年的新项目中仅使用辅助损失做负载均衡（无辅助损失才是默认方案）。拒绝在总参数超过 80 GB 且无专家并行方案时交付 MoE。将面向延迟敏感型单用户路径的 MoE 标记为很可能慢于等效稠密模型。
