---
name: skill-image-tensor-inspector
description: 检查任意图像形状的张量或数组，并报告其 dtype、布局、取值范围，以及它看起来是原始数据、归一化数据还是标准化数据
version: 1.0.0
phase: 4
lesson: 1
tags: [computer-vision, debugging, preprocessing, tensors]
---

# 图像张量检查器

一种诊断技能，适用于视觉流水线中任何你正持有一个图像形状数组、却需要确切知道它处于何种状态的环节。

## 何时使用

- 预训练模型返回了无意义的预测结果，你怀疑是预处理的问题。
- 在 OpenCV 与 torchvision 之间迁移流水线，且通道顺序不明确。
- 堆叠来自多个框架的层时，批次轴总是出现在错误的位置。
- 调试训练循环时，损失卡在 `log(num_classes)`。

## 输入

- `x`：任意二维、三维或四维类数组对象（NumPy、PyTorch、JAX）。
- 可选的 `expected`：一个用于校验不变量的字典，例如 `{"layout": "CHW", "range": "standardized"}`。

## 步骤

1. **解析后端** — 检测 `x` 是 NumPy、Torch 还是 JAX。在不修改原始数据的前提下，将其转换为 NumPy 以便检查。

2. **分类秩（rank）**：
   - rank 2 -> 单通道图像 (H, W)。
   - rank 3 -> 当最后一个轴为 1、3 或 4 且严格小于另外两个轴时为 `HWC`；否则为 `CHW`。
   - rank 4 -> 当轴 1 属于 {1, 3, 4} **且** 轴 2 或轴 3 大于 16 时，优先判定为 `NCHW`；否则优先判定为 `NHWC`。仅检查轴 1 会误判小图像的 NHWC 批次，例如 `(3, 4, 224, 3)`。
   - 对于歧义情况（例如 `(1, 3, 3, 3)`），始终标记为 `ambiguous`，不要猜测；要求调用方提供 `expected`。

3. **分类 dtype 和取值范围**：
   - `uint8` 且位于 [0, 255] -> `raw`。
   - `float*` 且 min >= 0、max <= 1.01 -> `normalized`。
   - `float*` 且 min < 0、|mean| < 0.5、0.5 <= std <= 1.5 -> `standardized`。
   - 其他情况 -> `unusual`，打印直方图。

4. **逐通道统计** — 报告每个通道的均值和标准差。如果数组看起来是标准化的，则与 ImageNet 的均值/标准差进行比对，并给出匹配置信度。

5. **报告**，使用以下精确格式：

```
[inspector]
  backend:   numpy | torch | jax
  rank:      2 | 3 | 4
  layout:    HW | HWC | CHW | NHWC | NCHW
  dtype:     <dtype>
  shape:     <shape>
  range:     raw | normalized | standardized | unusual
  min/max:   <min> / <max>
  per-channel mean: [ ... ]
  per-channel std:  [ ... ]
  likely source:    camera | PIL | OpenCV | torchvision | random init
  likely target:    display | training | inference
```

6. **建议下一步操作**，基于 `likely target`：
   - 对于 `display`：转置为 HWC，截断，转换为 uint8。
   - 对于 `training`：使用数据集统计量进行标准化，转置为 CHW，添加批次轴。
   - 对于 `inference`：严格匹配模型卡片中的不变量。

## 规则

- 永远不要修改输入。仅打印诊断信息。
- 如果提供了 `expected`，则对每一处不匹配标记 `[expected X got Y]`。
- 当布局或通道顺序存在歧义时，指出静默失败的风险。
- 每次只建议一个操作，而不是列出一组选项。
