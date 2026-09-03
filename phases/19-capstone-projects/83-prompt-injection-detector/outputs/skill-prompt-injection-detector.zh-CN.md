---
name: skill-prompt-injection-detector
description: 分层检测器流水线，对任意提示词返回类别和置信度，具备可量化的精确率与召回率
version: 1.0.0
phase: 19
lesson: 83
tags: [safety, detector, prompt-injection]
---

# 提示词注入检测器

此处的检测器是一个从提示词到判定结果的函数。判定结果携带一个来自课程 82 分类法的类别以及一个 [0, 1] 区间的置信度。

## 流水线

1. 归一化 - 剥离零宽字符，还原同形字，解码 base64/hex，折叠 leet-speak 数字，尝试 rot13 并以常见词做合理性检查。
2. 子串规则 - 手写的匹配串，如 `ignore previous`、`from now on you are`、`decode this base64`。
3. 正则规则 - token 级别的模式，如 `\bignor\w*\s+(all|prior|previous|earlier)\b`。

聚合阶段保留每个类别的最高分数，返回分数最大的类别，若无任何规则触发则返回 `benign`。

## 添加规则

编辑 `code/rules.py`。一条规则是一个字典，包含 `name`、`category`（六大分类法类别之一）、`score`（0 到 1 的浮点数）以及 `substring` 或 `regex` 之一。重新运行 `main.py` 即可查看对各类别精确率与召回率的影响。

## 产物

`outputs/detector_report.json` 是各类别指标文件。课程 87 中的端到端安全闸读取该文件以设定置信度阈值。
