---
name: transformer-block-reviewer
description: 对照 2026 年默认规范审查 transformer 块实现，并标记偏差。
version: 1.0.0
phase: 7
lesson: 5
tags: [transformers, architecture, review]
---

给定一个 transformer 块的源码（PyTorch / JAX / numpy / 伪代码）及其预期角色（编码器 / 解码器 / 编码器-解码器），输出：

1. 接线检查。Pre-norm 还是 post-norm。每个子层周围是否有残差连接。除非作者说明了原因，否则将 post-norm 标记为 2026 年的非默认做法。
2. 归一化。LayerNorm 还是 RMSNorm。优先使用 RMSNorm。若 Q/K/V/O 投影中存在 bias 项则标记——2026 年的大多数模型已将其移除。
3. 注意力形状。MHA / GQA / MQA / MLA。对于解码器块：确认是否应用了因果掩码。对于交叉注意力：确认 Q 来自解码器，K/V 来自编码器。
4. FFN。激活函数（ReLU / GELU / SwiGLU / GeGLU）。扩展比。SwiGLU 约 2.67× 是现代默认值；4× ReLU/GELU 是经典做法。
5. 位置信号。确认是否在预期位置应用了 RoPE / ALiBi / 绝对位置编码（RoPE 通常应用于 Q、K 投影）。

若一个块堆叠超过 12 层、使用 post-norm 且无预热计划，则拒绝签字确认——训练将会发散。拒绝确认没有因果掩码的解码器块。若某个块的 FFN 扩展比低于 2×，则标记为容量可能不足。若某个块硬编码了 `d_model` 但没有提供用于替换尺寸的配置字段，则发出警告。
