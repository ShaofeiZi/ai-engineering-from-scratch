---
name: skill-mask-rcnn-head-swapper
description: 为 torchvision Mask R-CNN 生成针对自定义 num_classes 替换 box 与 mask 头的精确代码
version: 1.0.0
phase: 4
lesson: 8
tags: [computer-vision, mask-rcnn, fine-tuning, torchvision]
---

# Mask R-CNN 头替换器

专门针对 Mask R-CNN 生成头替换的样板代码。下面的模板假设存在 `model.roi_heads.box_predictor` 和 `model.roi_heads.mask_predictor`，它们仅存在于 `maskrcnn_resnet50_fpn` 和 `maskrcnn_resnet50_fpn_v2` 上。Faster R-CNN 有 box predictor 但没有 mask predictor；RetinaNet 使用 `RetinaNetHead` 且根本没有 `roi_heads` —— 两者都需要不同的技能。

## 何时使用

- 在自定义类别集合上微调 `maskrcnn_resnet50_fpn` 或 `maskrcnn_resnet50_fpn_v2`。
- 将在 COCO 上训练的 Mask R-CNN 检查点迁移到非 COCO 类别数。
- 调试因 `cls_score.out_features` 或 `mask_predictor` 不匹配而崩溃的 Mask R-CNN 训练运行。

## 超出范围

- `fasterrcnn_*` —— 没有 mask_predictor。只替换 `box_predictor`；使用单独的 Faster R-CNN 头替换方案。
- `retinanet_*` —— 没有 `roi_heads`；分类和回归头位于 `model.head.classification_head` 和 `model.head.regression_head` 下。使用 RetinaNet 专用技能。
- `keypointrcnn_*` —— 使用 `keypoint_predictor` 而非 `mask_predictor`。

## 输入

- `model_name`：torchvision 检测模型构造器，例如 `maskrcnn_resnet50_fpn_v2`。
- `num_classes`：包含背景。一个 4 类目标的 dataset 表示 `num_classes=5`。
- `freeze`：`backbone`、`backbone_fpn`、`none` 之一。

## 步骤

1. 导入模型构造器和两个 predictor 类（`FastRCNNPredictor`、`MaskRCNNPredictor`）。
2. 加载默认预训练权重的模型。
3. 用新的 `model.roi_heads.box_predictor` 替换 `FastRCNNPredictor(in_features, num_classes)`。
4. 用新的 `model.roi_heads.mask_predictor` 替换 `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)`。
5. 应用所请求的冻结策略。
6. 打印确认块，列出每个模块的可训练参数。

## 输出代码模板

```python
from torchvision.models.detection import {MODEL_NAME}, {MODEL_WEIGHTS}
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes={NUM_CLASSES}):
    model = {MODEL_NAME}(weights={MODEL_WEIGHTS}.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    {FREEZE_BLOCK}

    return model
```

其中 `{FREEZE_BLOCK}` 为：

- `none` -> 为空
- `backbone` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  ```
- `backbone_fpn` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters live inside backbone.fpn
  ```

## 报告

```
[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>
```

## 规则

- 永远不要在未包含背景的情况下推荐 `num_classes`；始终提醒用户。
- 只要可用，始终使用 torchvision 检测模型的 `_v2` 变体；它们的预训练权重优于旧版本。
- 不要在此技能内部实例化模型 —— 生成代码块并让用户自行运行。
- 如果用户在超过 10,000 张图像的 dataset 上请求 `freeze backbone`，建议其考虑也微调 backbone。
