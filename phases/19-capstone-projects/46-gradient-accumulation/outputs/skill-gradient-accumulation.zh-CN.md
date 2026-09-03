---
name: gradient-accumulation
description: 通过对微批次损失进行缩放并在每个窗口仅执行一次优化器步进，以超过设备显存限制的有效批次进行训练。
version: 1.0.0
phase: 19
lesson: 46
tags: [training, batch-size, distributed, scaling]
---

## 适用场景

有效批次是平滑梯度并匹配学习率调度表的关键杠杆。当你无法在单次前向传播中承受该批次大小时，这就是解决方案。

## 配方

1. 选择 `micro_batch` 为能放入显存并使加速器饱和的最大尺寸。
2. 根据学习率调度表选择 `effective_batch`。
3. 设定 `accum_steps = effective_batch // (micro_batch * world_size)` 并断言其能整除。
4. 每个微批次：`loss = criterion(model(x), y) / accum_steps; loss.backward()`。
5. 在非最终微批次上，进入 `model.no_sync()` 以跳过 DDP 中的梯度 all-reduce。
6. 最后一个微批次结束后，执行一次 `optimizer.step()`。在下一个窗口之前清零梯度。
7. 优化器状态每个有效批次推进一次；学习率调度表每个有效批次滴答一次。

## 日志

每个有效步输出一条小型 JSON 记录，包含 `samples_per_sec`、`median_step_ms`、`sync_calls`、`accum_steps`、`effective_batch`。没有这些信息，成本权衡将不可见。

## 故障模式

- 遗忘 `/ accum_steps` 缩放：梯度膨胀 N 倍。
- 窗口中途步进：参数漂移。
- 每个微批次都同步：网络受限却无统计收益。
- 与混合精度反缩放混用：仅对反缩放后的损失进行缩放。
