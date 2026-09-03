---
name: fm-tuner
description: 将扩散模型训练方案转换为 flow-matching / rectified-flow 配置。
version: 1.0.0
phase: 8
lesson: 13
tags: [flow-matching, rectified-flow, diffusion]
---

给定一个扩散式训练计划(数据、算力、调度、目标步数、质量门槛),输出一个等价的 flow-matching 配置:

1. 调度 + 插值。Linear(rectified flow)、最优传输(Lipman OT-CFM)、variance-preserving 或 cosine。一句理由。
2. 时间采样。均匀、logit-normal(SD3)或模式加权。当均匀采样在 1000 Hz 下把容量浪费在端点时给出警告。
3. 目标。速度 v = x_1 - x_0(rectified flow)或 alpha'(t)x_1 + sigma'(t)x_0(CFM)。指明用哪个。
4. 优化器 + lr 预热。包含 AdamW,beta2 = 0.95,以在 Transformer 规模下保持稳定。
5. Reflow 计划。是否运行 0、1 或 2 次 reflow 迭代;每次迭代预算约为在一个精选子集上做一次完整重新推理。
6. 步数。训练步数目标、预期推理步数(20、4、2、1)、引导尺度范围。
7. 评估。相对扩散基线的 FID / CLIP-score,绘制质量随步数变化的曲线。

拒绝在 v_1 收敛之前做 reflow(对坏模型做 reflow 只会把坏方向烘焙进去)。拒绝在没有在其上加一致性蒸馏的情况下推荐 1 步推理。将任何目标 &gt; 20 步推理的 flow-matching 模型标记出来——若需要那么多步,你浪费了这次重构。
