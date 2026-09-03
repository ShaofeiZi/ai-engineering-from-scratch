---
name: diffusion-trainer
description: 配置一次扩散模型训练：包括调度、预测目标、采样器和评测方案。
version: 1.0.0
phase: 8
lesson: 06
tags: [diffusion, ddpm, training]
---

给定数据集画像(模态、分辨率、数据集大小)、计算预算(GPU 小时数、VRAM 下限)和质量门槛(FID 目标或下游用途),输出:

1. 调度。Linear、cosine(Nichol)或 sigmoid。步数 T(DDPM 基线 1000;更快变体 256)。
2. 预测目标。epsilon、v-prediction 或 x_0。理由结合分辨率和整条调度上的信噪比。
3. 架构。像素扩散用 U-Net 深度 + 通道宽度,潜在扩散用 DiT,视频用 3D U-Net / DiT。包含时间嵌入方案(正弦 + MLP、FiLM 或 AdaLN)。
4. 采样器。DDIM(20-50 步)、DPM-Solver++(10-20)、Euler-A(创意)或蒸馏 1-4 步。包含引导尺度(CFG w)建议。
5. 评估方案。FID / KID / CLIP-score / 人类偏好,附样本数(FID >=10k),以及 CFG w 的扫描协议。

拒绝在 &gt;=256x256 时推荐训练像素空间扩散——潜在扩散能用 1/16 的 FLOPs 达到同等质量。拒绝交付不含 CFG 的条件生成模型——从条件模型直接做零样本无条件采样通常退化。将任何 beta_T &gt; 0.1 的调度标记为很可能产生饱和或不稳定的训练。
