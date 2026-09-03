---
name: token-gen-cost-analyzer
description: 计算 Emu3 式下一 token 生成的 token 数量、推理延迟和质量上限，并在 Emu3 家族与扩散模型之间做出选择。
version: 1.0.0
phase: 12
lesson: 12
tags: [emu3, next-token-prediction, video-gen, diffusion, cfg]
---

给定生成产品规格（图像或视频、目标分辨率、质量等级、吞吐量需求），计算 Emu3 式下一 token 生成的 token 数量，估算推理成本，并在 Emu3 家族与扩散模型之间做出选择。

产出：

1. Token 数量。所选 tokenizer 压缩率下的每张图像 token 数（图像通常每维度 8x）。3D VQ 下的每视频 token 数（通常 4x4x4 时空）。
2. 推理延迟。Emu3 家族的 token 数 / 吞吐量（每秒 token 数）；扩散模型的去噪步数 * 每步耗时。引用具体的 A100 / H100 范围。
3. 质量上限。Tokenizer 重建 PSNR（IBQ 级别为 30-32 dB）、MJHQ-30K 上的 FID 预期、视频的 FVD。
4. CFG 配置。每个任务推荐的引导权重（gamma）；标准生成通常为 3.0，强提示词遵循为 5-7。
5. 选择。如果产品需要统一的理解 + 生成或任意模态灵活性，选择 Emu3 家族；如果产品仅为图像生成且对延迟严格，选择扩散模型（SDXL / SD3 / Flux）。

坚决拒绝：
- 声称 Emu3 在推理上比扩散模型更快。并非如此；对数千个图像 token 的自回归解码是固有成本。
- 推荐 Emu3 家族却不指定 CFG 权重。没有它质量会崩溃。
- 对严格的 4K 图像生成提出 Emu3。2048+ 分辨率下的 token 数量会撑爆 KV cache 并耗时数分钟。

拒绝规则：
- 如果延迟预算 <5s 每张图像，拒绝 Emu3 并推荐 SDXL 或 SD3。
- 如果产品必须输出图像并描述它们且对第三方图像进行推理，推荐 Emu3 家族（统一的损失函数正是关键所在）；扩散模型在没有单独 VLM 的情况下无法做到这一点。
- 如果用户想要具有宽松商业许可的开源权重，拒绝 Emu3——先检查其许可证；某些版本仅限研究用途。

输出：一页分析，包含 token 数量、延迟估算、质量上限、CFG 配置以及带理由的选择。最后附上 arXiv 2409.18869（Emu3）和 2408.11039（Transfusion）作为替代参考。
