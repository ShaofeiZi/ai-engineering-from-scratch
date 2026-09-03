---
name: skill-conv-shape-calculator
description: 逐层遍历 CNN 规格，报告每个块的输出形状、感受野和参数量
version: 1.0.0
phase: 4
lesson: 2
tags: [computer-vision, cnn, architecture, debugging]
---

# 卷积形状计算器

一个用于规划或调试 CNN 的确定性辅助工具。给定输入形状和一层规格列表，无需运行模型即可追踪形状、感受野和参数量。

## 何时使用

- 设计新的 CNN 时，想验证每次下采样都落在整齐的尺寸上。
- 阅读论文并将其架构表翻译成代码时。
- 预训练主干网络在分类头处因形状不匹配而崩溃，需要知道是哪一层改变了空间尺寸。
- 在训练两个主干网络之前，比较它们的参数效率。

## 输入

- `input_shape`：`(C, H, W)`。
- `layers`：有序的层字典列表。每个支持：
  - `{type: "conv", c_out, k, s, p, groups=1, bias=true}`
  - `{type: "pool", mode: "max"|"avg", k, s, p=0}`
  - `{type: "adaptive_pool", out_h, out_w}`
  - `{type: "flatten"}`
  - `{type: "linear", out_features, bias=true}`

## 步骤

1. **初始化追踪**，使用 `(C, H, W)`、感受野 `1`、有效步幅 `1`、累计参数量 `0`。

2. **对于每一层**，按以下顺序更新：
   - 计算 `C_out`（conv/linear），或在 pool 时沿用 `C_in`。
   - 计算空间输出：conv 和 pool 使用 `(H + 2P - K) / S + 1`，adaptive pool 使用 `out_h/out_w`，flatten 输出形状 `(1, 1)` 在 linear 之前为 `(C * H * W, 1, 1)`，linear 为标量 `1x1`。
   - 更新感受野和有效步幅：
     - Conv/pool：`RF_new = RF_old + (K - 1) * effective_stride`，`effective_stride *= S`。
     - Adaptive pool：视为有效 `S = H_in / out_h`（向下取整）的 pool。`RF_new = RF_old + (H_in - 1) * effective_stride_old`；`effective_stride *= S`。注意 adaptive pool 的感受野等于之前完整的空间范围。
     - Flatten / linear：感受野和有效步幅不再有意义；将它们冻结为 flatten 之前的值，并在后续行中省略。
   - 计算参数量：
     - Conv：`C_out * (C_in / groups) * K * K + (C_out if bias else 0)`。
     - Linear：`out_features * in_features + (out_features if bias else 0)`。
     - Pool 和 flatten：0。

3. **检测问题**并标记：
   - 非整数输出尺寸（步幅/填充未对齐）。
   - 在堆栈结束前出现 `H_out <= 0`。
   - 感受野超过输入尺寸（此后可能存在浪费的计算）。
   - 单层参数量突然出现 10 倍跳跃，暗示通道方案有误。

4. **报告**为单一表格：

```
idx  layer                C_in  C_out  K  S  P  H_out  W_out  RF    params     cum_params
1    conv 3x3 s=1 p=1     3     32     3  1  1  224    224    3     896        896
2    conv 3x3 s=2 p=1     32    64     3  2  1  112    112    7     18,496     19,392
3    pool max 2x2         64    64     2  2  0  56     56     11    0          19,392
...
```

5. **总结行**：最终的 `(C, H, W)`、最终感受野、总参数量、警告。

## 规则

- 空间尺寸始终返回整数。如果公式产生非整数，标记为错误，不要默默向下取整。
- 当 `groups > 1` 时，验证 `C_in % groups == 0` 和 `C_out % groups == 0`；否则报错。
- 对于深度卷积（`groups == C_in`），在 `layer` 列中标注，以便读者了解参数量为何较低。
- 如果用户提供 BatchNorm 或激活层，出于形状目的忽略它们，但继续累计参数量（每个 BatchNorm `2 * C`）。
- 切勿为缺失字段猜测默认值。每个 conv 和 pool 都必须提供 `k`、`s`、`p`。
