---
name: 3d-pipeline
description: 根据输入类型、输出格式和使用场景，选择 3D 生成或重建流水线。
version: 1.0.0
phase: 8
lesson: 12
tags: [3d, gaussian-splatting, nerf, mesh]
---

给定输入(文本提示 / 单张图像 / 少量图像 / 照片采集 / 视频)、目标输出(网格 / 高斯泼溅 / NeRF / 点云)和用途(实时渲染、游戏引擎、AR / VR、影视级),输出:

1. 流程。(a)多视角扩散 + 3D 拟合(SV3D、CAT3D + 3DGS),(b)直接单次生成(LRM、TripoSR、InstantMesh),(c)带 PBR 的文本转网格(Meshy 4、Rodin Gen-1.5、Hunyuan3D 2.0),(d)照片采集 + 3DGS(Gsplat、Postshot、Scaniverse)。
2. 基座模型 + 托管。具名模型 + 开源 / 托管。包含商用许可的相关性。
3. 迭代预算。首次产出的预期时间、迭代成本、精修策略。
4. 拓扑 + 材质。是否需要重网格化?PBR 通道需求(反照率、粗糙度、金属度、法线)?UV 展开自动还是手动?
5. 评估。留出视角上的 SSIM、CLIP 得分、网格水密性、面数、纹理分辨率。
6. 平台目标。Unity / Unreal / Blender / Web(three.js / Babylon)/ AR(USDZ / glb)。

拒绝在没有经过网格转换步骤的情况下把 3DGS 直接交付到游戏引擎(多数引擎不原生渲染泼溅)。拒绝用文本转 3D 做复杂铰接角色——改用支持绑骨的流程。当下游工具无法渲染 NeRF 时(多数 DCC 工具),将任何仅 NeRF 的输出标记出来。
