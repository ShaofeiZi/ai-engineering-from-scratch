---
name: img2img-chooser
description: 根据数据是否成对、领域特性和延迟预算，选择图像到图像方案。
version: 1.0.0
phase: 8
lesson: 04
tags: [pix2pix, img2img, conditional]
---

给定任务描述(源域、目标域、数据可得性——配对/非配对/N 个样本、延迟预算、质量门槛),输出:

1. 方案。Pix2Pix(配对、窄域)、Pix2PixHD(配对、高分辨率)、CycleGAN(非配对)、SPADE(语义图→图像),或基于 SD3 / Flux.1 的 ControlNet 变体(通用、开放域)。
2. 训练数据规格。最小配对数、分辨率、增强方式、许可注意事项。
3. 架构。G(U-Net 深度、通道宽度)、D(PatchGAN 感受野、谱归一化)、loss 权重(对抗、L1、VGG 感知)。
4. 推理延迟。单张消费级 GPU(RTX 4090、M3 Max)上每张图的目标毫秒数,以及分辨率权衡。
5. 评估。在留出配对数据上的 LPIPS、5k 样本上的 FID、任务专属指标(分割任务的 mIoU、超分辨任务的 PSNR)、人类偏好。

拒绝在数据为非配对时推荐 Pix2Pix——应改为 CycleGAN 或 ControlNet。拒绝在没有增强 / 预训练建议的情况下训练少于 500 对的配对模型。将任何提到 "任意文本提示" 的请求标记出来——这类需求需要扩散 + ControlNet,而非配对 GAN。
