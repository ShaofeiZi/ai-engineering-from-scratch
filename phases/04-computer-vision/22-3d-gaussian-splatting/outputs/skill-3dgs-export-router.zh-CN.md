---
name: skill-3dgs-export-router
description: 根据下游查看器或引擎，选择正确的 3DGS 导出格式（.ply / .splat / glTF KHR_gaussian_splatting / USD）
version: 1.0.0
phase: 4
lesson: 22
tags: [3d-gaussian-splatting, export, glTF, OpenUSD, pipeline]
---

# 3DGS 导出路由器

将下游目标映射到正确的 3DGS 文件格式。可省去数小时「加载不出来」的调试时间。

## 何时使用

- 在训练完一个 3DGS 场景之后，将其交给内容流水线之前。
- 在研究级格式（.ply）和生产级格式（glTF / USD）之间做选择时。
- 流水线交接：采集团队 -> 3DGS 工程师 -> 游戏设计师 / VFX 美术 / Web 开发者。

## 输入

- `target_engine`: unreal | unity | omniverse | blender | vision_pro | three_js | babylon_js | cesium | playcanvas | supersplat
- `priority`: portability | file_size | quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3

## 格式决策

| 目标 | 推荐格式 | 原因 |
|--------|--------------------|-----|
| Unreal Engine（虚拟制片） | Volinga 插件或 glTF KHR_gaussian_splatting | 原生 Unreal SDK 路径 |
| Unity（XR / 游戏） | 通过 Aras-P Unity-GaussianSplatting 插件的 .ply | 社区标准的 Unity 流水线 |
| NVIDIA Omniverse、Pixar 工具 | OpenUSD 26.03（UsdVolParticleField3DGaussianSplat） | 原生 USD prim 类型 |
| Apple Vision Pro | OpenUSD 26.03 | visionOS 2.x 原生支持 |
| Blender | .ply + KIRI Engine 附加组件 | 社区附加组件可读取原始 splat |
| Three.js Web 查看器 | glTF KHR_gaussian_splatting 或 .splat | 浏览器标准，兼容 `GaussianSplats3D` |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9 已添加原生支持 |
| Cesium（CesiumJS 1.139+、Cesium for Unreal 2.23+） | glTF KHR_gaussian_splatting | 已显式支持 |
| PlayCanvas | .splat | PlayCanvas 原生量化格式 |
| SuperSplat（编辑器） | .ply 或 .splat | 支持导入 + 导出 |

## 量化权衡

- `.ply` 全精度：文件最大，无损，兼容任意查看器。
- `.splat`：体积小 4 到 8 倍，SH3 系数有轻微质量损失，PlayCanvas 生态标准。
- glTF KHR：可通过 EXT_meshopt_compression 配置；体积最小且兼容性最高。
- USD：通过 USDZ 打包压缩；对 Apple 流水线体积最小。

## 输出报告

```
[export plan]
  target:         <engine>
  format:         <name>
  sh degree:      <0|1|2|3>
  compression:    <none|meshopt|quantisation|usdz>
  expected size:  <MB>
  compatible with: <list of viewers>

[pipeline]
  1. source: <.ply from training>
  2. optional: SuperSplat cleanup pass
  3. convert: <tool + CLI or API call>
  4. package: <.gltf / .glb / .usd / .usdz / .splat / .ply>
  5. validate: <viewer sanity check>
```

## 规则

- 绝不要静默剥离 SH3 系数——这会明显改变镜面反射效果。
- 如果 `priority == file_size`，推荐 `.splat` 或带 meshopt 的 glTF；并警告会有质量损失。
- 对于 Apple 平台，在 2026 年优先选择 USD / USDZ 而非 glTF；USDZ 在 visionOS 上有一等支持。
- 如果目标查看器的 3DGS 支持属于标准化前阶段（2026 年 2 月之前），推荐 `.ply` 及该查看器的自定义加载器；Khronos 标准的 glTF 尚不会被识别。
- 交接前务必在至少一个查看器中验证导出的文件；量化过程中会发生静默损坏。
