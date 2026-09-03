---
name: skill-vit-patch-and-pos-embed-inspector
description: 验证 ViT 的 patch embedding 与位置编码形状是否匹配模型预期的序列长度
version: 1.0.0
phase: 4
lesson: 14
tags: [vision-transformer, debugging, pytorch]
---

# ViT Patch 与位置编码检查器

最常见的 ViT 移植 bug：将在 224x224 预训练的检查点加载到配置为 384x384 的模型中（或反之）。位置编码的序列长度不匹配，模型会悄无声息地输出垃圾结果。

## 何时使用

- 在非默认分辨率下微调预训练 ViT。
- 审查 ViT-B/16 与 ViT-B/32 之间权重迁移为何失败；检查器会标记 patch 大小不匹配，使调用方知道应更换架构而非强行迁移。
- 调试加载无报错但训练效果不佳的 ViT。

## 输入

- `model`：已实例化的 ViT `nn.Module`。
- `expected_image_size`：模型在生产中将看到的 H x W。
- `patch_size`：预期的 patch 大小。

## 步骤

1. 在模型中定位 patch embedding 卷积层。报告其 `kernel_size`、`stride`、`in_channels`、`out_channels`。
2. 计算预期的 patch 数量。对于方形图像：`(image_size / patch_size)^2`。对于矩形图像：`(H / patch_size) * (W / patch_size)`。要求 `H % patch_size == 0` 且 `W % patch_size == 0`；否则标记并拒绝。
3. 定位学习到的位置编码。报告其形状 `(1, N, dim)`。
4. 将 `N` 与 `num_patches + 1`（含 CLS）或 `num_patches`（不含 CLS）进行比较。不匹配意味着检查点是在不同分辨率或 patch 大小下预训练的。
5. 检查 patch 卷积的 `out_channels` 是否等于位置编码的 `dim`。
6. 如果模型应当对新分辨率插值位置编码，验证插值工具是否存在（大多数 `timm` ViT 通过 `resize_pos_embed` 自动完成此操作）。

## 报告

```
[vit-inspector]
  image_size:         HxW
  patch_size:         <int>
  num_patches (computed): <int>
  patch_conv:         k=<int>  s=<int>  in=<int>  out=<int>
  pos_embed shape:    (1, N, dim)
  has CLS token:      yes | no
  pos_embed N:        <int>    expected: <int>
  verdict:            ok | mismatch

[if mismatch]
  action:  reinitialise pos_embed for new sequence length
  tool:    timm.models.vision_transformer.resize_pos_embed
```

## 规则

- 永远不要在无警告的情况下静默插值；应揭示该操作，让用户知道预训练的位置结构可能已发生变化。
- 如果 patch_size 不匹配，拒绝推荐插值——应更换为正确的架构。
- 不要尝试就地修复模型；只报告并给出建议。
