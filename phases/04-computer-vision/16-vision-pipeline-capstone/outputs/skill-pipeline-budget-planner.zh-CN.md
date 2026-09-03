---
name: skill-pipeline-budget-planner
description: 给定目标延迟和吞吐量，为每个流水线阶段分配时间预算，并标记哪个阶段会最先超出预算
version: 1.0.0
phase: 4
lesson: 16
tags: [vision, pipeline, performance, deployment]
---

# 流水线预算规划器

将延迟/吞吐量目标转化为逐阶段的预算，使每个团队成员都清楚自己要达成的指标。

## 何时使用

- 在构建新的视觉服务之前，为每个阶段设定预期。
- 在首次基准测试之后，查看哪个阶段离其预算差距最大。
- 当 SLA 发生变化、预算需要重新协商时。

## 输入

- `p95_latency_target_ms`：单请求预算。
- `target_qps`：每个副本的吞吐量。
- `stages`：`{ name: str, current_ms: float }` 列表。

## 分配规则

如果未提供当前测量值，则按以下默认分配比例划分七个标准阶段：

| 阶段 | 占比 |
|-------|-------|
| 解码 + 预处理 | 15% |
| 检测器前向计算 | 55% |
| 检测后处理（NMS、截断） | 5% |
| 裁剪 + 调整大小用于分类器 | 5% |
| 分类器前向计算 | 15% |
| schema 校验 | <1% |
| 响应序列化 | 4% |

在受 GPU 限制的流水线（云端）中，检测器占比通常会上升到 70%。在 CPU 上，预处理和分类器批处理会占用更多时间。

## 报告

```
[budget plan]
  p95 target:  <ms>
  throughput:  <qps per replica>

| stage               | target_ms | current_ms | headroom | gate |
|---------------------|-----------|------------|----------|------|
| decode+preprocess   | ...       | ...        | ...      | ok|X |
| detector            | ...       | ...        | ...      | ok|X |
| ...                 | ...       | ...        | ...      |      |

[bottleneck]
  stage:  <name>
  miss:   <ms over budget>
  lever:  <specific action>

[levers]
  decode+preprocess:   Pillow-SIMD, libjpeg-turbo, decode on GPU via NVJPEG
  detector:            smaller backbone, lower input resolution, INT8, TensorRT
  postprocess:         GPU-side NMS (torchvision.ops), fused masks
  crop+resize:         GPU crop with grid_sample, batched interpolate
  classifier:          smaller backbone, INT8, warm cache, batch
  schema:              skip validation in hot path, validate at boundaries only
  response:            orjson, stream protobuf
```

## 规则

- 切勿建议在生产路径中移除 schema 校验；应提议将其移至边界处。
- 如果预处理超出预算，务必先尝试 Pillow-SIMD 或 NVJPEG，再考虑更换模型。
- 如果检测器超出预算的幅度大于目标的 30%，应更换模型而非优化当前模型。
- 当 current_ms > 1.1 * target_ms 时，将 gate 标记为 `X`；当处于预算 10% 范围内时，标记为 `ok`。
