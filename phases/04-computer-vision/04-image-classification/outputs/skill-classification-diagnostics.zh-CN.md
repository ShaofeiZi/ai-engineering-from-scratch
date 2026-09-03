---
name: skill-classification-diagnostics
description: 给定混淆矩阵和类别名称，暴露每个类别的失败情况并提出最具影响力的单一修复方案
version: 1.0.0
phase: 4
lesson: 4
tags: [computer-vision, classification, evaluation, debugging]
---

# 分类诊断

混淆矩阵的一种解读视角。总体准确率告诉你分类器是有效的，而混淆矩阵告诉你它*尚未掌握什么*。

## 何时使用

- 首次查看已训练分类器的验证性能时。
- 在训练轮次之间，用于决定下一步修改什么。
- 在发布模型之前：验证没有关键类别在悄然失败。
- 调试生产回归问题时，整体准确率下降了一个百分点，你需要知道原因。

## 输入

- `cm`：CxC 混淆矩阵（行 = 真实，列 = 预测）。
- `labels`：C 个类别名称的列表，顺序与矩阵一致。
- 可选 `class_priors`：每个类别的训练频率（默认为 `cm` 的行和）。

## 步骤

1. **计算每个类别的指标。** 将任何除以零的情况视为该类别的指标未定义，并报告为 `n/a`；切勿静默替换为 0。
   - precision_i = cm[i,i] / sum(cm[:, i])   （当该类别从未被预测时未定义）
   - recall_i    = cm[i,i] / sum(cm[i, :])   （当该类别没有真实样本时未定义）
   - f1_i        = 2 * p * r / (p + r)        （当任一组成部分未定义时未定义）

2. **按 F1 排序，列出最多三个最差的类别。** 如果混淆矩阵少于三个类别，则列出所有现有类别。排除所有指标均未定义的类别。

3. **找出每行最大的非对角单元**——即最常窃取该类别的那个类别。以 `true -> predicted` 的形式报告。

4. **对每个最差类别判定失败模式。** 使用以下量化阈值，以确保标签可复现：
   - `ambiguity`（歧义）——与另一个类别存在双向混淆：`cm[i,j] / sum(cm[i, :]) >= 0.15` 且 `cm[j,i] / sum(cm[j, :]) >= 0.15`。
   - `imbalance`（不平衡）——该类别的训练样本数少于其首要混淆对象的 `< 0.5x`。
   - `label_noise`（标签噪声）——`|precision_i - recall_i| >= 0.2` 且该类别未处于 imbalance / ambiguity 路径上。
   - `systematic`（系统性）——没有任何单一混淆对象超过该类别错误份额的 0.2；错误分散在三个或更多其他类别上。

5. **推荐最具影响力的单一后续动作**：
   - `ambiguity` -> 收集或合成具有区分性的样本，添加能够保留区分特征的针对性数据增强。
   - `imbalance` -> 对少数类过采样，或采用类别加权损失。
   - `label_noise` -> 对该类别进行分层抽样审计；在进行任何其他修改之前先修正错误标签。
   - `systematic` -> 为该类别增加数据，或在损失中以更高权重微调该类别。

## 报告

```
[diagnostics]
  aggregate accuracy: X.XX
  macro F1:           X.XX

[top-3 worst classes]
  1. class <name>  F1 = X.XX  prec = X.XX  rec = X.XX
     top confusion: <name> -> <other>  (N cases)
     failure mode:  ambiguity | imbalance | label_noise | systematic
     action:        <one sentence>

  2. ...
  3. ...

[recommendation]
  single biggest lever: <one sentence naming the class and the fix>
```

## 规则

- 最多返回三个类别。更多会掩盖信号。
- 为每个最差类别指出首要混淆对象；切勿笼统地概括为“与多个类别混淆”。
- 每条建议都应基于混淆矩阵的证据。不得在未指明具体类别的情况下给出“增加更多数据”这类泛泛建议。
- 当精确率与召回率相差超过 0.2 时，始终将标签噪声标记为候选原因——真实类别在训练后通常具有对齐的 P 和 R。
