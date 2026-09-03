---
name: mha-configurator
description: 为新 Transformer 推荐头数、KV 头数及投影策略（MHA / MQA / GQA / MLA）。
version: 1.0.0
phase: 7
lesson: 3
tags: [transformers, attention, mha, gqa]
---

给定一个 Transformer 规格说明（参数预算、隐藏层维度 `d_model`、目标上下文长度、推理设备显存、训练优先还是推理优先），输出：

1. 投影变体。从以下之一选择：MHA、GQA、MQA、MLA。给出一句与 KV 缓存约束相关的理由。
2. 头几何配置。`n_heads`、`n_kv_heads`、`d_head`。取值必须满足 `d_model = n_heads * d_head` 和 `n_heads % n_kv_heads == 0`。
3. KV 缓存估算。在目标上下文长度下，所选变体每层每 token 的字节数（fp16）。若单批次超过目标设备显存，需明确标注。
4. 初始化。Q、K、V、O 矩阵的 Xavier / Kaiming 缩放。注明是否包含偏置项（2026 年的大多数模型已弃用偏置）。
5. 可测试性钩子。一个合成任务（如归纳头模式 `A B A ? → B`），该配置训练后的两层版本应能以 ≥95% 的准确率完成。

拒绝推荐 `d_head < 32`——注意力动态会崩坏。对于上下文长度超过 32K 的场景，若使用 `n_heads > 16`，拒绝推荐 MHA，除非已显式核算 KV 缓存开销并建议改用 GQA 或 MLA。对于参数量低于 1B 的模型，除非用户明确在对其进行基准测试，否则拒绝建议使用 MLA。
