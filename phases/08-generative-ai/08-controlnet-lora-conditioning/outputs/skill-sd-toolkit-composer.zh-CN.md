---
name: sd-toolkit-composer
description: 针对给定的一组输入，在 SD / Flux 基座上组合 ControlNets、LoRAs 和 IP-Adapters。
version: 1.0.0
phase: 8
lesson: 08
tags: [controlnet, lora, ip-adapter, diffusion]
---

给定任务(目标图像)、输入(提示词、参考图、姿态 / 深度 / 涂鸦 / 语义分割、主体身份)和基座模型(SDXL、SD3.5、Flux.1-dev),输出:

1. ControlNet 堆栈。用哪些 ControlNet(canny / openpose / depth / scribble / seg / lineart / tile),权重多少,顺序如何。权重总和上限 &lt;= 1.5。
2. LoRA 堆栈。具名 LoRA、rank、alpha。当 alpha &gt; 1.5 或多个 LoRA 指向同一概念时给出警告。
3. IP-Adapter。无、普通版或 FaceID 变体;典型权重 0.4-0.8。
4. 文本提示词 + 负面提示词。关键词顺序、token 预算、负面脚手架。
5. 采样器 + CFG + 种子。Euler A / DPM-Solver++ / LCM;CFG 尺度与基座绑定。可复现种子协议。
6. QA 检查清单。目视检查 ControlNet 漂移、LoRA 过饱和、IP-Adapter 身份泄漏、解剖问题。

拒绝把 SD 1.5 LoRA 堆到 SDXL 基座上(维度不匹配)。拒绝 3+ 个 ControlNet 各以权重 1.0 运行(特征冲突)。当用户有 GPU 预算上 SDXL 或 Flux 时,把任何 SD 1.5 推荐标记出来。把在 &lt; 10 张图像上做 LoRA 身份训练标记为很可能过拟合。
