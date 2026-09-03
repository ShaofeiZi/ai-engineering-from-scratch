---
name: stylegan-inversion
description: 为使用预训练 StyleGAN 处理真实照片选择反演与编辑流水线。
version: 1.0.0
phase: 8
lesson: 05
tags: [stylegan, inversion, editing]
---

给定一张真实照片 + 预训练 StyleGAN 检查点(FFHQ-1024、StyleGAN-XL、自定义微调版)和目标编辑(年龄、微笑、姿态、发型、身份保持),输出:

1. 反演方法。e4e(快、低保真)、ReStyle(迭代编码器)、HyperStyle(超网络)、PTI(pivotal tuning)或直接 W 优化。结合保真度与速度给出一句理由。
2. 目标空间。W、W+ 或 StyleSpace。权衡:W = 最解耦但保真最低,W+ = 逐层 w,StyleSpace = 通道级。
3. 编辑方向。具名方向来源:InterFaceGAN(基于 SVM)、StyleSpace 通道、GANSpace PCA 或已学习分类器。
4. 保真预算。身份漂移前的 LPIPS 阈值;回滚启发式。
5. 评估。ID 相似度(ArcFace 余弦)、与原图的 LPIPS、编辑强度(目标属性分类器得分)。

拒绝任何直接在 Z 中编辑的流程(已耦合)。拒绝在 W 中超过 1.5 sigma 的大幅编辑而不做身份检查。将需要开放域编辑的请求(例如 "把他变成卡通")标记出来——这类需求需要扩散 + IP-Adapter,而非 StyleGAN。
