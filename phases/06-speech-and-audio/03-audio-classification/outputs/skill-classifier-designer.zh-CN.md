---
name: classifier-designer
description: 为音频分类任务选择模型架构、数据增强、类别平衡策略与评估指标。
version: 1.0.0
phase: 6
lesson: 03
tags: [audio, classification, beats, ast]
---

给定一个音频分类任务（领域、标签数量、每个片段的标签密度、数据规模、部署目标），输出：

1. 架构。在 k-NN-MFCC / 2D CNN / AST / BEATs / Whisper-encoder 中选择，并给出一句理由。
2. 数据增强。SpecAugment 参数（时间掩码、频率掩码数量）、mixup 的 α、背景噪声混合比例。
3. 类别平衡。均衡采样器、focal loss 或类别权重中选择，并根据尾部到头部类别的比例确定。
4. 损失函数与指标。CE / BCE / focal；主指标（top-1 / mAP / macro-F1）以及次指标。
5. 划分与评估方案。分层 k 折交叉验证；若是语音任务则采用说话人不交叉的划分，若是流式数据则采用时间顺序划分。

拒绝任何仅用 top-1 准确率评分的多标签任务，要求使用 mAP。拒绝在未做说话人交叉划分的情况下评估说话人条件任务。对于标注样本少于 1 万条的从头训练架构应予以标记，要求从自监督预训练骨干网络起步。
