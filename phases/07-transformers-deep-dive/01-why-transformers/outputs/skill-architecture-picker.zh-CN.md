---
name: sequence-architecture-picker
description: 根据序列长度、吞吐量和训练预算选择序列架构(RNN、Transformer、SSM、混合架构)。
version: 1.0.0
phase: 7
lesson: 1
tags: [transformers, architecture, rnn, ssm]
---

给定一个序列问题(最大长度、批次形状、预算训练 token 数、推理延迟目标、设备类别),输出:

1. 首选架构。以下之一:transformer、状态空间模型(Mamba/RWKV)、SSM+注意力混合架构、RNN。用一句话说明理由,并与主要约束条件挂钩。
2. 上下文长度策略。若为 transformer:全注意力截断阈值、滑动窗口大小、RoPE 缩放因子。若为 SSM:扫描块大小。若为 RNN:隐藏层宽度。
3. 训练 FLOP 特征。基于架构与上下文估算每 token 的近似 FLOPs,并说明该规格是否匹配算力预算。
4. 推理内存特征。Transformer 的 KV 缓存、SSM 的状态大小、RNN 的每 token 内存。标注目标设备是否能容纳单个 batch size 为 1 的样本。
5. 风险说明。指出该选择在规格所述规模下已知存在的一个具体失败模式(例如:在 24GB GPU 上、未使用 Flash Attention 的情况下,transformer 在 64K 上下文处出现 OOM)。

对于任何超过 10 亿 token 的训练任务,若推荐纯 RNN,必须明确说明其梯度流与并行化方面的代价,否则不予推荐。对于超过 64K 上下文的情况,若推荐全注意力 transformer,必须说明其 `O(N^2)` 内存开销,否则不予推荐。对于生产环境,若推荐全新架构(发表不足 12 个月),必须提供一个具名的备选方案,否则不予推荐。
