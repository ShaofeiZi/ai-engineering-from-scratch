---
name: skill-segmentation-mask-inspector
description: 报告类别分布、预测掩码统计信息，以及最可能被欠预测或在边界处模糊的类别
version: 1.0.0
phase: 4
lesson: 7
tags: [computer-vision, segmentation, debugging, evaluation]
---

# 分割掩码检查器

一种用于诊断“损失已下降”与“掩码实际看起来正确”之间差距的工具。

## 何时使用

- 在一次训练运行刚结束时，mIoU 看起来正常但可视化检查结果并非如此。
- 部署前：检查预测的类别分布是否与真值一致。
- 当大目标的逐类别 IoU 较高，而小目标的逐类别 IoU 较低时。
- 调试那些因像素占比很小而未在 IoU 中体现的边界伪影。

## 输入

- `preds`: (N, H, W) 预测类别 ID 张量。
- `targets`: (N, H, W) 真值类别 ID 张量。
- `num_classes`: 整数。
- 可选 `class_names`: 由 C 个字符串组成的列表。

## 步骤

1. **类别像素直方图。** 分别计算 `preds` 和 `targets` 中各类别像素的百分比。对任何满足 `|pred% - gt%| / max(gt%, 1e-6) > 0.30`（相对偏差超过 30%）的类别进行标记。对于真值中不存在的类别（`gt% == 0`），只要预测占比超过 `0.3` 就直接标记。

2. **逐类别 IoU** 和 **逐类别边界 F1**。边界 F1 的计算方式为：将每个掩码膨胀 3 个像素，然后求交集并评分。IoU > 0.7 但边界 F1 < 0.5 的类别表明其边缘正在模糊。

3. **小目标召回率。** 将每个真值的连通组件按大小分桶（tiny < 100 像素，small < 1000 像素，medium < 10000 像素，large >= 10000 像素）。报告每个类别在每个桶中的召回率。当小目标召回率低于 0.3 而大目标召回率高于 0.9 时，表明存在分辨率/感受野问题。

4. **混淆类别对。** 对每个类别，找出与其最容易混淆的类别（在其真值掩码内最常见的错误预测类别）。报告前 3 对。

5. **饱和度检查（需要 `probs` 或 `logits`，而不仅仅是 `preds`）。** 如果调用方传入原始的逐像素概率分布 `probs: (N, C, H, W)`，则计算每个类别中满足 `probs.max(dim=1) > 0.99` 的像素占比。高饱和度（某类别超过 0.9 的像素）表明过度自信——可考虑使用标签平滑或校准。当仅有经过 argmax 的 `preds` 可用时，跳过此步骤并在报告中注明。

## 报告格式

```
[mask-inspector]
  classes: C

[class distribution]
  name       gt %    pred %   delta
  ...

[metrics]
  class       IoU     bF1    recall_tiny  recall_small  recall_medium  recall_large
  ...

[confusion pairs]
  class A confused with class B: <N> pixels (most common)
  class B confused with class A: <N> pixels
  ...

[verdict]
  most impactful issue: <one sentence>
```

## 规则

- 按真值像素占比降序排列类别行，使最频繁的类别排在最前。
- 将 IoU < 0.4 或边界 F1 < 0.3 的类别标记为 `critical`。
- 当小目标召回率是主要失败原因时，建议：更高分辨率的训练、在最后一个编码器阶段使用更小的步长，或使用特征金字塔解码器。
- 当边界 F1 是主要失败原因时，建议：边界感知损失（Lovasz 或 BoundaryLoss）、带水平翻转的 TTA，以及无步长解码器。
- 切勿仅以类别索引作为唯一标识；如果提供了 `class_names`，则在每一行中使用它。
