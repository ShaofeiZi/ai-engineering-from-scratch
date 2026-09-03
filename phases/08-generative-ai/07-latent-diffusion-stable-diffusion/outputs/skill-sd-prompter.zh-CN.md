---
name: sd-prompter
description: 针对给定提示词、风格和质量标准，配置 Stable Diffusion / Flux 推理。
version: 1.0.0
phase: 8
lesson: 07
tags: [stable-diffusion, flux, latent-diffusion]
---

给定提示词、目标风格和质量门槛(快速预览 / 作品集级 / 可印刷),输出:

1. 模型 + 检查点。SD 1.5(遗留工具)、SDXL-base + refiner、SDXL-Turbo(快)、SD3.5-Large、Flux.1-dev(最佳开源)、Flux.1-schnell(快速开源),或托管 API(DALL-E 3、Imagen 4、Midjourney v7)。一句理由。
2. 采样器。Euler A(创意)、DPM-Solver++ 2M Karras(稳定)、LCM(快)或 flow-matching 采样器(SD3/Flux)。附步数。
3. CFG 尺度。turbo / LCM 用 0,Flux 用 3-4,SDXL 用 5-7,SD1.5 用 7-10。记录权衡。
4. 附加组件。ControlNet(姿态、深度、canny、语义分割)、IP-Adapter(参考图)、LoRA(风格或主体)、SD3+ 的 T5 开关。
5. 负面提示词。显式空串与填充内容(伪影、低质、解剖错误)有差别;两者都需指定。

拒绝 SDXL+ 使用 CFG &gt; 10(输出饱和)。拒绝在非遗留检查点上用 &gt; 50 步采样器(质量在 30 步时已趋于平台)。拒绝把在不同基座模型上训练的 LoRA 混用(SD 1.5 LoRA 用在 SDXL 上会静默失效)。将任何对写实人物的请求,在没有提醒 NSFW、深度伪造和版权政策的情况下,标记出来。
