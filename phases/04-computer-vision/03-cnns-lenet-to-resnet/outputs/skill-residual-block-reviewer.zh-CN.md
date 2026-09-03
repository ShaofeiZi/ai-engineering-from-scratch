---
name: skill-residual-block-reviewer
description: 审查 PyTorch 残差块的跳跃连接正确性、BN 放置位置、激活顺序和形状对齐
version: 1.0.0
phase: 4
lesson: 3
tags: [computer-vision, resnet, code-review, pytorch]
---

# 残差块审查器

一个专注于审查任何声称实现了残差块的 PyTorch `nn.Module` 的审查器。能够捕获几乎所有错误 ResNet 重写中的四类常见错误。

## 何时使用

- 有人编写了自定义的 BasicBlock 或 Bottleneck，但 loss 为 NaN 或 accuracy 卡住不动。
- 你正在将一个 block 从一个框架移植到另一个框架，并希望验证等价性。
- 你正在审查一个修改 ResNet 内部结构（pre-activation、squeeze-excite、anti-alias）的 PR。
- 模型在 CIFAR 大小输入上运行正常，但在 ImageNet 分辨率上崩溃，原因是 shortcut 不正确。

## 输入

- 一个 PyTorch 类定义，以源代码文本形式或可导入路径形式提供。
- 可选 `variant`：`basic` | `bottleneck` | `preact` | `seblock`。

## 四项检查

### 1. Shortcut 形状对齐

对于任何 `stride != 1` 或 `in_channels != out_channels` 的 block，shortcut 路径**必须**是一个形状匹配的模块——通常是一个 1x1 卷积加 BN。在这种情况下使用裸的 `nn.Identity()` 会在前向计算时必然导致形状不匹配错误。

诊断信息：
```
[shortcut]
  detected:  nn.Identity | 1x1 Conv + BN | 1x1 Conv + BN + ReLU | other
  required:  shape-matching Conv if (stride != 1 or in_c != out_c) else Identity
  verdict:   ok | wrong | unnecessarily heavy
```

### 2. BN 相对于加法的放置位置

加法 `out + shortcut(x)` 必须发生在最终 ReLU **之前**（post-activation，原始 ResNet），或最终 ReLU 必须完全不存在（pre-activation，ResNet v2）。如果一个 block 在主分支中应用了 ReLU，然后再加上一个未经处理的 shortcut，会产生不对称的激活范围，从而损害训练效果。

诊断信息：
```
[activation order]
  pattern:  post-act (conv-BN-ReLU-conv-BN-add-ReLU) | pre-act (BN-ReLU-conv-BN-ReLU-conv-add) | other
  verdict:  ok | suspect
```

### 3. 卷积层的 bias

紧接 BatchNorm 的卷积层应设置 `bias=False`。BN 的 beta 参数已经参数化了偏置，因此额外的卷积 bias 会浪费参数并可能拖慢收敛。

诊断信息：
```
[bias]
  convs with BN and bias=True: <count>
  recommended fix: set bias=False on those layers
```

### 4. 原地操作 ReLU 与 autograd

在将被加到 shortcut 上的张量上使用 `nn.ReLU(inplace=True)` 会覆盖在残差加法中仍可能需要的值。标记任何 `inplace=True` 且其后没有在加法之前产生新张量的层。

诊断信息：
```
[in-place]
  risky inplace ops: <list>
  fix: inplace=False before the residual add
```

## 报告

```
[block-review]
  variant:       basic | bottleneck | preact | se | other
  shortcut:      ok | wrong | heavy
  activation:    ok | suspect
  bias-bn:       ok | <N> convs need bias=False
  in-place:      ok | <N> risky ops
  summary:       one sentence
```

## 规则

- 不要重写 block。仅作报告。
- 如果 block 是正确的，在各处都标注 `ok` 并停止。不提任何建议。
- 如果有多处错误，按上述顺序列出（shortcut 优先，因为它是导致崩溃最常见的原因）。
- 当用户已明确指定时，切勿将刻意设计的 pre-activation 或 squeeze-excite 变体标记为错误。
