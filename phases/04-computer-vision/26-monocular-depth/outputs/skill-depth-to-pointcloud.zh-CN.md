---
name: skill-depth-to-pointcloud
description: 从深度图构建点云，正确处理内参并导出为 .ply
version: 1.0.0
phase: 4
lesson: 26
tags: [depth, point-cloud, 3d, intrinsics]
---

# 深度图转点云

将深度图和彩色图像转换为带纹理的点云，可导出用于可视化或进一步的 3D 处理。

## 适用场景

- 将深度预测结果可视化为真实的 3D 场景。
- 从单张图像引导构建稀疏 3D 重建。
- 在 SfM 失败时为 3DGS 训练生成输入数据。
- 将预测深度与 LiDAR 真值进行对比。

## 输入

- `depth`：`(H, W)` numpy 数组，深度单位与输出所需单位一致（推荐使用米）。
- `rgb`：`(H, W, 3)` numpy 数组，颜色值（uint8 或 float32 [0, 1]）。
- `intrinsics`：`(fx, fy, cx, cy)`，以像素为单位。
- 可选 `depth_scale`：乘数，用于将预测深度单位转换为米。

## 处理流程

1. **校验** —— 深度在所有你计划纳入的位置必须为正值且有限。将无效像素掩膜掉。
2. **上提** —— 对每个像素执行 `X = (u - cx) * d / fx`，`Y = (v - cy) * d / fy`，`Z = d`。
3. **配对** RGB —— 每个三维点从对应像素获取一个 `(r, g, b)` 三元组。
4. **导出** —— PLY（通用），`.xyz`（轻量），`.pcd`（Open3D 原生），`.las`/`.laz`（地理空间）。

## 实现模板

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics, depth_scale=1.0, min_depth=0.1, max_depth=100.0):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.astype(np.float32) * depth_scale
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def write_ply(path, points, colors=None, valid_mask=None):
    p = points.reshape(-1, 3)
    if valid_mask is not None:
        p = p[valid_mask.flatten()]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {p.shape[0]}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        c = colors.reshape(-1, 3).astype(np.uint8)
        if valid_mask is not None:
            c = c[valid_mask.flatten()]
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if colors is not None:
            for pt, col in zip(p, c):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")
        else:
            for pt in p:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
```

## 报告

```
[export]
  input depth shape:  (H, W)
  valid points:       <N> of <H*W>
  output format:      ply | xyz | pcd | las
  coordinate system:  camera (+X right, +Y down, +Z forward)
  scale:              metres | millimetres | normalised
```

## 规则

- 必须始终掩膜掉无效深度（零、NaN、inf、饱和值）；若将其纳入，会在原点附近产生一堆垃圾点云。
- 对于来自相对深度模型的预测结果，不要作为度量值导出；在输出文件名前加 `relative_` 前缀以标明此约定。
- 保持相机坐标系约定一致（OpenCV：+X 向右，+Y 向下，+Z 向前）。若下游工具采用 OpenGL（+Y 向上），则需交换符号。
- 对于稠密场景（> 1M 个点），提供下采样参数；超过 500 MB 的 PLY 文件在多数平台上都难以加载。
- 绝不要为了产生“合理”输出而静默裁剪深度；应使用带告警的阈值进行显式裁剪，让用户知道被丢弃了哪些数据。
