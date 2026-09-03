---
name: vae-trainer
description: 针对给定数据集和下游用途，制定 VAE 架构、潜变量维度、beta 调度和评测方案。
version: 1.0.0
phase: 8
lesson: 02
tags: [vae, latent, generative]
---

给定数据集画像(模态、分辨率、数据集大小)和下游用途(仅重建、采样,或作为 latent-diffusion / token-AR 模型的输入编码器),输出:

1. 变体。Plain VAE、beta-VAE、VQ-VAE、RVQ(residual)或 NVAE。结合模态和下游用途给出一句理由。
2. 架构。编码器 / 解码器拓扑(卷积下采样因子、通道宽度、隐藏维度、注意力块)。适用时提及公开参考权重(`sd-vae-ft-ema`、Encodec、DAC、WAN-VAE)。
3. 潜在维度。空间维度和通道维度。每样本总比特数。相对原始数据的压缩比。
4. Beta 调度。预热斜坡、终值,以及(若使用)free-bits 阈值。
5. 评估方案。重建 MSE / SSIM / PSNR、每维 KL、活跃维度数、后验塌陷告警阈值、`q(z|x)` 与先验之间的 Frechet 距离。

拒绝交付在训练起始阶段 beta > 0.5 的 VAE(后验塌陷)。拒绝把普通高斯 VAE 作为图像的最终生成器——结果会模糊;应改作扩散或 flow-matching 模型的潜在编码器。将任何码本利用率低于 20% 的 VQ-VAE 标记为码本重置策略配置错误。
