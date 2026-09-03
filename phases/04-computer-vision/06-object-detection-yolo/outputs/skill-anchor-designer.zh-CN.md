---
name: skill-anchor-designer
description: 给定一组真实标注框数据集，对 (w, h) 运行 k-means，返回每个 FPN 层级的锚框集合及覆盖率统计
version: 1.0.0
phase: 4
lesson: 6
tags: [computer-vision, detection, anchors, kmeans]
---

# 锚框设计器

锚框是基于锚框的检测器中最依赖数据集的超参数。COCO 的默认锚框在细胞培养图像、卫星切片或小目标监控场景下表现不佳。本技能用于推导出真正匹配目标数据的锚框。

## 何时使用

- 在新数据集上首次训练之前。
- 当一个整体健康的模型在极小或极大目标上召回率较弱时。
- 数据集大幅扩充后，框尺寸分布可能已发生偏移时。

## 输入

- `boxes`：形状为 (N, 4) 的 numpy 数组，格式为 `(cx, cy, w, h)` 或 `(x1, y1, x2, y2)`；建议至少 1000 个正样本框。
- `num_anchors_per_level`：通常为 3。
- `num_fpn_levels`：通常为 3（P3、P4、P5）或 4。
- `input_size`：训练分辨率 HxW。
- 可选 `strides`：各层级步长；省略时取 `num_fpn_levels` 的前 `[8, 16, 32, 64]` 个元素。若检测器的 FPN 步长不同，请显式传入更长或更短的数组。

## 步骤

1. **将框归一化** 为以像素为单位的 `(w, h)` 对，分辨率对应 `input_size`。丢弃 w 或 h < 2 像素的框。

2. **对 `(w, h)` 对运行 k-means**，令 `k = num_anchors_per_level * num_fpn_levels`。使用 `1 - IoU(box, cluster)` 作为距离函数，而非欧氏距离——在 `(w, h)` 上使用欧氏距离会把细长高框和方形框合并到一起。所有框权重相等（不加权）；若数据集类别不平衡且希望提升大框召回率，可在输入数组中重复稀有类别的框，而非传入权重向量。

3. **按面积升序排列聚类簇**。将其分为 `num_fpn_levels` 组，每组 `num_anchors_per_level` 个。面积最小的归到最高分辨率层级（步长最小）。

4. **按层级计算覆盖率统计**：
   - `median IoU`：每个真实框在该层级上与最佳锚框的 IoU 中位数。
   - `recall@IoU=0.5`：最佳锚框 IoU >= 0.5 的框所占百分比。
   - `area coverage`：框面积落在该层级 `[anchor_min_area / 4, anchor_max_area * 4]` 范围内的比例。

5. **报告各层级锚框**，并对 `recall@IoU=0.5 < 0.9` 的层级进行标记；该层级的锚框与数据匹配不佳，应重新调参或增加每层锚框数。

## 报告格式

```
[anchor-designer]
  total boxes:         <N>
  clusters:            <k>
  distance metric:     1 - IoU

[level P3  stride=8]
  anchors (w, h):      [(A, B), (C, D), (E, F)]
  median IoU:          <X>
  recall@IoU=0.5:      <X>
  coverage:            <X>
  flag:                ok | retune

[level P4  stride=16]
  ...

[summary]
  overall recall@IoU=0.5: <X>
  smallest anchor:        <w x h>
  largest anchor:         <w x h>
  recommendation:         <one sentence if any level flagged>
```

## 规则

- 始终使用基于 IoU 的距离；欧氏 k-means 产生的锚框在视觉上看似合理，但经验上效果更差。
- 按面积排序聚类簇，再按升序分配到各层级。
- 当 `num_anchors_per_level = 1` 时，完全跳过 k-means：按面积分位数将框分为 `num_fpn_levels` 个区间（例如 3 个层级用三分位数），每个层级的锚框取对应区间的 (w, h) 中位数。这在小型数据集上比运行 `k = num_fpn_levels` 的 k-means 更稳健。
- 绝不输出负的锚框尺寸；最小钳制为 1。
- 如果数据集少于 200 个框，警告用户锚框搜索结果不可靠，并建议使用 COCO 默认锚框并补充更多训练数据。
