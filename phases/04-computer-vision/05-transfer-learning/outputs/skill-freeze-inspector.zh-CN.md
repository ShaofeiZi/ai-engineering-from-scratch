---
name: skill-freeze-inspector
description: 报告哪些参数可训练、哪些 BatchNorm 层处于 eval 模式，以及优化器是否确实消费了这些可训练参数
version: 1.0.0
phase: 4
lesson: 5
tags: [computer-vision, transfer-learning, debugging, pytorch]
---

# 冻结检查器（Freeze Inspector）

迁移学习的 bug 通常隐藏在三个地方：本应冻结却没有冻结的参数、本应可训练却不可训练的参数，以及在冻结状态改变之前就已构建好的优化器。本技能一次性排查这三类问题。

## 何时使用

- 在对一部分参数设置 `requires_grad` 之后。
- 在微调训练首次训练步之前。
- 在调用 `freeze_bn_stats` 或任何切换 BN 模式的辅助函数之后。
- 当验证准确率卡在随机水平、你怀疑实际上没有任何参数在训练时。

## 输入

- `model`：一个 PyTorch `nn.Module`。
- `optimizer`：即将用于训练的优化器。
- 可选 `expected_frozen_prefixes`：应当被冻结的参数名前缀列表（例如 `["conv1", "bn1", "layer1"]`）。

## 步骤

1. **遍历参数。** 对每个 `(name, param)`：
   - 记录 `requires_grad`
   - 记录 `shape` 和 `numel`

2. **遍历模块。** 对每个模块：
   - 如果是 BatchNorm，记录它是否处于 eval 模式，以及其仿射参数是否可训练。

3. **检查优化器。** 对每个参数组：
   - 将其 `params` 展平为一个 `id(p)` 集合。
   - 与所有 `id(p)` 的参数的 `requires_grad == True` 集合进行比较。

4. **检测四种失败模式：**
   - `leaked_train`：某个参数 `requires_grad=True`，但未出现在优化器中（梯度已计算但从未被应用）。
   - `ghost_train`：某个参数出现在优化器中，但 `requires_grad=False`（优化器状态被浪费；如果之后重新启用 requires_grad，还可能引发 bug）。
   - `bn_mismatch`：要么（a）一个 BN 层处于 train 模式（会累积运行统计量）但其仿射参数（`weight`、`bias`）被冻结，要么（b）一个 BN 层处于 eval 模式（统计量已冻结）但其仿射参数可训练。这两种状态都不一致，几乎总是 bug。
   - `expected_vs_actual`：`expected_frozen_prefixes` 中列出的任一前缀仍然存在可训练参数。

## 报告

```
[freeze-inspector]
  model trainable params: <N>
  model frozen params:    <N>
  batchnorm layers in eval mode: <count>
  batchnorm layers in train mode: <count>

[optimizer coverage]
  trainable params fed to optimizer: <M> of <N>
  leaked_train: <list of names> (trainable but not in optimizer)
  ghost_train:  <list of names> (in optimizer but frozen)

[bn audit]
  mismatched layers: <list of names>

[expectations]
  expected_frozen_prefixes: <...>
  violating params:         <list>

[verdict]
  ok | <one-line summary of the most severe issue>
```

## 规则

- 只报告参数名；绝不打印权重本身。
- 对每个列表按参数名字母顺序排序。
- 如果优化器覆盖率为 100% 且不存在任何不一致，返回 `ok` 并停止。
- 对于 `leaked_train`，始终建议在冻结状态改变后重建优化器。
- 对于 `ghost_train`，建议移除该参数组，或者如果意图是训练该参数，则设置 `requires_grad=True`。
