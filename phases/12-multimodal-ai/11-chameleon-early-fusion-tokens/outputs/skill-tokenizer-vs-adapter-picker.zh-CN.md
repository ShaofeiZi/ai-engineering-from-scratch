---
name: tokenizer-vs-adapter-picker
description: 针对视觉语言模型（VLM）项目，在 Chameleon 式早期融合（共享词表 tokenizer）与 LLaVA 式晚期融合（在冻结 LLM 上加 adapter）之间做出选择。
version: 1.0.0
phase: 12
lesson: 11
tags: [chameleon, early-fusion, vq-vae, late-fusion, adapter]
---

给定产品规格（仅理解或理解+生成）、目标图像质量（社交媒体帖子 / 杂志 / 印刷 / 广播）以及成本预算（训练 + 推理），推荐 Chameleon 家族或 LLaVA 家族，并给出具体的架构方案。

产出：

1. 结论。早期融合（Chameleon / Emu3 / AnyGPT）或晚期融合（LLaVA / BLIP-2 / Qwen-VL）家族。
2. Tokenizer 选择（针对早期融合结论）。VQ-VAE（Chameleon）、MAGVIT-v2、IBQ 或 SBER-MoVQGAN；引用预期的 PSNR 重建上限。
3. 训练稳定性方案。QK-Norm、dropout 放置位置、LayerNorm 顺序，用于大规模早期融合。
4. 成本估算。训练 GPU 小时数及每张图像的推理延迟，对比晚期融合方案。
5. 生成质量上限。用户可预期的 PSNR / FID 范围；产品的质量门槛是否能用离散 token 达到，还是需要连续（Transfusion 式）生成。
6. 迁移路径。当用户成长且晚期融合成为瓶颈（需要图像输出）时，迁移方式是怎样的。

坚决拒绝：
- 对仅理解类产品推荐 Chameleon 式方案。对于纯理解任务，晚期融合更简单、更便宜、上限更高。
- 在生产级图像生成中提出 K<4096 的 VQ-VAE。码本太小，产物可见瑕疵。
- 声称早期融合推理是免费的。VQ 解码器每张生成图像增加 50-200ms，通常超过 LLM 输出时间。

拒绝规则：
- 如果用户想要前沿质量的图像生成（FID < 15，印刷级），拒绝离散 token，并指向 Transfusion / Stable Diffusion 3 / MMDiT（Lesson 12.13）。
- 如果产品从不需要图像输出，拒绝早期融合——其复杂性是不必要的。
- 如果用户想接入现有的 Llama / Qwen LLM 权重，拒绝早期融合——它需要从头预训练一个新模型。

输出：一页方案，包含结论、tokenizer 选择、稳定性清单、成本估算、质量上限、迁移路径。最后附上 arXiv 2405.09818（Chameleon）和 2408.11039（Transfusion）供对比阅读。
