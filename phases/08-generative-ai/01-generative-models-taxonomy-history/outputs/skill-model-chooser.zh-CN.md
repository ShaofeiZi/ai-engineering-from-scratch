---
name: generative-model-chooser
description: 针对给定任务和预算，选择生成模型家族、主干架构和托管服务替代方案。
version: 1.0.0
phase: 8
lesson: 01
tags: [generative, taxonomy]
---

给定任务描述（模态、领域、延迟预算、算力预算、条件信号），输出：

1. 模型族。显式可精确求解（explicit-tractable）、显式近似（VAE / diffusion）、隐式（GAN）、score / flow matching，或 token-AR。给出一句与模态和延迟相关的理由。
2. 骨干网络 + 开源参考。列出一个用户现在即可微调的预训练开放权重模型（例如 Stable Diffusion 3、Flux.1-dev、AudioCraft 2、StyleGAN3、3D Gaussian Splatting）。
3. 托管替代方案。按质量 / 成本 / 延迟权衡排序，列出三个生产级 API（fal.ai、Replicate、Stability、Runway、Veo、Kling、ElevenLabs 等）。
4. 失败模式。说明所选模型族的已知问题（模式崩塌、暴露偏差、采样器漂移、分词器伪影、针对 CLIP-score 的投机优化）。
5. 预算。估算单张 A100 上的训练时长、每个样本的推理成本与最低 VRAM 要求。

当任务需要似然评分时，拒绝推荐 GAN。对于高分辨率实时场景，拒绝推荐逐像素自回归模型。如果所列开放权重骨干网络已经覆盖该领域，应对任何“从零训练”的建议明确提示风险。
