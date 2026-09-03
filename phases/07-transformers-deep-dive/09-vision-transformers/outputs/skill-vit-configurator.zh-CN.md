---
name: vit-configurator
description: 为新的视觉任务选择 ViT 变体、patch 大小和预训练来源。
version: 1.0.0
phase: 7
lesson: 9
tags: [transformers, vit, vision]
---

给定一个视觉任务（分类 / 分割 / 检测 / 检索）、图像分辨率、数据集规模（已标注 + 未标注）以及部署目标，输出：

1. 骨干网络。以下之一：DINOv2 ViT-L/14（检索/分类的默认选择）、SAM 3 encoder（分割）、SigLIP（视觉-语言）、ConvNeXt（对延迟敏感的场景）。给出一句理由。
2. Patch 大小。224 分辨率下标准分类用 16，DINOv2 用 14，高分辨率密集预测用 8。标注序列长度 `(H/P)^2 + 1` 与注意力开销 `O(N^2)`。
3. 预训练来源。检查点名称。对于小规模已标注集合（<10k）：冻结 DINOv2 特征 + 线性探针。对于 >100k：微调最后几层。说明原因。
4. 训练方案。优化器（AdamW）、学习率、数据增强（RandAug、MixUp、Random Erasing）、标签平滑（典型值 0.1）、EMA。
5. 风险提示。数据规模风险（数据过少导致全量微调不可行）、分辨率不匹配（预训练 224 → 部署 1024 而未做位置编码插值）、register token 缺失（可能损害 DINOv2 特征质量）。

拒绝在少于 100 万张图像时推荐从零训练 ViT —— CNN 基线会胜出。拒绝在未明确讨论 Flash Attention + 层级式变体（Swin）的情况下推荐导致序列长度 > 4096 的 patch 大小。对任何在未对位置编码插值的情况下改变输入分辨率的部署方案，均予以标注提示。
