---
name: skill-frame-sampler-auditor
description: 审计视频流水线的帧采样器，排查 off-by-one、短片段处理和裁剪一致性
version: 1.0.0
phase: 4
lesson: 12
tags: [computer-vision, video, sampling, debugging]
---

# 帧采样器审计

帧采样是视频流水线最容易出错的环节。这里的缺陷会传播到每一个下游指标中。

## 何时使用

- 编写新的视频数据加载器。
- 复现论文中的数值，但训练精度低于报告值。
- 调试评估精度在多次运行间不稳定的视频模型。

## 输入

- `sampler_code`：Python 函数，接收 (num_frames_total, T) 并返回 T 个索引。
- `T`：目标片段长度。
- 可选测试用例：需要演练的 `num_frames_total` 取值（例如 `[3, T-1, T, T+1, 30, 300, 3000]`）。

## 检查项

### 1. 短片段处理
输入 `num_frames_total < T`。每个返回的索引都必须落在 `[0, num_frames_total - 1]` 内。标准的填充策略是为剩余位置重复最后一帧。

### 2. 边界索引
输入 `num_frames_total == T`。返回的索引应恰好为 `[0, 1, ..., T-1]`。

### 3. 均匀分布
输入 `num_frames_total == 10 * T`。返回的索引应单调递增且大致等距分布。

### 4. 密集窗口边界
对于密集采样，输入 `num_frames_total == 3 * T`。返回的索引应构成一个连续窗口，绝不跨越片段末尾。

### 5. 确定性
使用相同输入（对于确定性采样器，还包括相同的 RNG）调用采样器两次。索引应完全一致。

### 6. 裁剪一致性
如果流水线还会为每帧返回一个空间裁剪框，则对同一片段使用相同种子运行采样器两次，确认每一帧使用相同的裁剪框（相同的 `(x, y, w, h)`）。同一片段内每帧使用不同裁剪框会破坏时间一致性，是一种典型的静默缺陷。可接受的变化：增强以 *每片段* 方式应用，且在片段内部保持一致。

## 报告

```
[sampler audit]
  name: <function name>
  T:    <int>

[short-clip handling]
  passed | failed (<details>)

[boundary]
  passed | failed

[uniform spacing]
  passed | failed (<stddev of gaps>)

[dense window]
  passed | failed (<details>)

[determinism]
  passed | failed

[crop consistency]
  passed | failed (<per-frame crop varies: yes/no>)

[verdict]
  ok | fix required
```

## 规则

- 如果短片段处理返回了越界索引，绝不能将采样器标记为 "ok"。
- 密集采样器返回的窗口绝不应跨越 `num_frames_total - 1`。
- 如果采样器是随机的（密集），仅在提供显式带种子 RNG 时才测试确定性。
- 建议但不要静默修复以下规范策略：用最后一帧填充、将窗口钳制到末尾、采用半开区间舍入。
