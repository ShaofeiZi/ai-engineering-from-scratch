---
name: transformer-review
description: 对照第 7 阶段的 13 节课程，审查一个从零实现的 Transformer 代码库。
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, review, capstone]
---

给定一个从零实现的 Transformer 代码库（PyTorch / JAX），请对照 2026 年的默认实践进行审查，并标出缺失或错误的部分：

1. 注意力机制。因果掩码必须存在。按 `sqrt(d_head)` 进行缩放。多头拆分需正确工作。若条件允许应使用 Flash Attention。当 d_model ≥ 1024 时应提及 GQA。
2. 位置编码。使用 RoPE（2026 年首选），或学习式绝对位置编码（仅对小模型可接受）。应将正弦式位置编码标记为历史遗留方案。
3. 模块接线。采用 Pre-norm（而非 Post-norm）。使用 RMSNorm（而非 LayerNorm）。FFN 使用 SwiGLU（而非 ReLU/GELU）。每个子层都需有残差连接。线性层丢弃偏置项（现代默认做法）。
4. 训练。使用 AdamW（或 2026 年起的 Muon），配合线性预热的余弦学习率调度，梯度裁剪阈值设为 1.0，使用 bf16 自动混合精度。token embedding 与 lm_head 之间需权重共享。
5. 损失。在每个位置上做错位一位（shift-by-one）的交叉熵。如有 padding 需将其掩去。以固定间隔记录训练损失与验证损失。

若代码库存在以下任一情况，则拒绝通过审查：无明确理由却使用 post-norm；2026 年的生产代码中无正当理由使用 LayerNorm；解码器自注意力缺失因果掩码；小型语言模型中 embedding 未与输出层共享权重。需要标记的情况包括：无验证集划分、未做梯度裁剪、学习率超过 1e-3 却无预热、或 block_size 超出位置嵌入范围且无回退方案。建议端到端运行 `python code/main.py`，并检查在 nano 配置下于 tinyshakespeare 上的最终验证损失是否能降到 2.5 以下。
