---
name: vlm-recipe-picker
description: 选择一套开源权重 VLM 方案（编码器、连接器、LLM、数据配比、分辨率策略），并为每一项选择提供消融实验表格引用。
version: 1.0.0
phase: 12
lesson: 07
tags: [vlm, mm1, idefics2, molmo, cambrian, prismatic, ablation]
---

给定一个任务组合（OCR、图表、UI 智能体、推理、定位）、一个算力预算（LLM 参数量、训练 GPU 小时数或推理延迟目标），以及一个部署约束（边缘、云端、端侧），输出一份完整的开源权重 VLM 方案并附引用。

产出内容：

1. 编码器选择。默认 SigLIP 2 SO400m/14；若任务组合中包含定位/分割，则与 DINOv2 ViT-g/14 拼接；引用 MM1 Table 3 与 Cambrian-1 的视觉编码器对比。
2. 连接器选择。默认 2 层 MLP，除非受 token 约束（此时使用 Q-Former 32 queries）；引用 Prismatic VLMs 的连接器消融实验，其显示差距 <1 分。
3. LLM 选择。依据预算决定：低于 10B 选 Qwen2.5-7B，高于 30B 选 Llama-3.1-70B 或 Qwen2.5-72B。标注 MMMU 在 70B 之后趋于平台期。
4. 数据配比。默认 PixMo + ShareGPT4V + Cauldron；引用 Molmo 的详细人工描述（detailed-human-caption）结果（在相同 token 数下比蒸馏高 +2-3 MMMU）。
5. 分辨率策略。默认动态（256-1280），并设 stage-1 固定 384 对齐预训练；引用 Idefics2 分辨率消融实验（AnyRes 在 DocVQA 上 +3-5）和 Qwen2.5-VL 动态 M-RoPE。
6. 训练阶段。Stage 1 仅训练 projector，Stage 2 全量微调，Stage 3 任务专用。

硬性否决：
- 在未标注新项目应以 SigLIP 2 替代 CLIP ViT-L/14 的情况下，将 CLIP ViT-L/14 作为默认编码器推荐。
- 将 Q-Former 建议为优于 MLP 的质量提升手段。它是 token 预算的调节杠杆，而非质量杠杆。
- 在存在人工描述替代数据时，提出将合成 GPT-4V 描述作为主要训练数据。引用 Molmo。
- 声称连接器架构解释了实际来自 token 数量的方差。

拒绝规则：
- 如果用户希望使用 1-3B 的 VLM 完成推理密集型任务，予以拒绝并推荐更大的 LLM；推理上限由 LLM 决定。
- 如果用户无法负担详细人工描述数据，明确标注预期 2-3 分的 MMMU 上限，并提供尽力而为的蒸馏回退方案。
- 如果任务组合在冻结编码器部署中包含 4K+ 文档图像，拒绝 AnyRes 并推荐原生分辨率 M-RoPE 编码器，如 Qwen2.5-VL。

输出：一页方案卡，包含各维度选择、消融引用（arXiv ID）、训练阶段计划以及预期基准范围。结尾列出三篇接下来应阅读的消融论文：arXiv 2403.09611（MM1）、2405.02246（Idefics2）、2409.17146（Molmo）。
