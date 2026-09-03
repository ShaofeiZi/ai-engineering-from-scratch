---
name: unified-gen-model-picker
description: 在 Show-o / Transfusion / Emu3 / Janus-Pro 系列中，为一个需要多模态理解与生成且使用开放权重的产品选择模型族。
version: 1.0.0
phase: 12
lesson: 14
tags: [show-o, masked-diffusion, unified, t2i, inpainting]
---

给定一个需要统一理解 + 生成（VQA、captioning、T2I，可选 inpainting）、有开放权重约束且有延迟预算的产品，选择一个模型族并输出一份参考配置。

产出：

1. 模型族判定。Show-o（掩码离散 diffusion）、Transfusion / MMDiT（连续 diffusion）、Emu3 / Chameleon（自回归离散）、或 Janus-Pro（解耦编码器）。
2. 推理步数预算。Show-o 16 步，Transfusion 20 步，Emu3 1024+ 步。用用户的延迟预算论证该选择。
3. Inpainting 支持。Show-o 天然支持；Transfusion 增加一个掩码通道；Emu3 需要单独微调。为用户标注此项。
4. Tokenizer 选择。离散系列推荐 IBQ / MAGVIT-v2 / SBER；连续系列推荐 SD3 的 VAE。
5. 训练稳定性。双损失（Transfusion）需要权重调参；Show-o 的单损失更简洁。
6. 用户成长时的迁移路径。当质量成为瓶颈时，从 Show-o 迁移到 Transfusion。

硬性拒绝：
- 当每张图像推理延迟 <10s 时推荐 Emu3 / Chameleon。对 ~1024 个 token 做自回归太慢。
- 声称 Show-o 在前沿图像质量上匹敌 Transfusion。并非如此。tokenizer 决定了质量上限。
- 为需要 VQA 的产品推荐 Stable Diffusion。SD 无法对图像进行推理。

拒绝规则：
- 如果用户要求每张图像生成 <2s，拒绝 Show-o 并推荐 Stable Diffusion + 一个独立的 VLM 做理解。接受多模型复杂性。
- 如果用户在开放权重下追求"同类最佳质量"，拒绝 Show-o / Emu3 并推荐 Transfusion 系列（MMDiT）或 JanusFlow。
- 如果用户无法确定 tokenizer（担心许可或质量上限），拒绝纯离散系列并推荐 Transfusion。

输出：一页选型文档，包含模型族判定、步数预算、inpainting 支持、tokenizer 推荐、稳定性方案与迁移路径。最后给出 arXiv 2408.12528 (Show-o)、2408.11039 (Transfusion)、2501.17811 (Janus-Pro)。
