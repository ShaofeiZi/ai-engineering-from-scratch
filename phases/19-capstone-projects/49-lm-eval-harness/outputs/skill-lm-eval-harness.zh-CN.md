---
name: lm-eval-harness
description: 最小化语言模型评测工具，包含 JSONL 任务规格、五项指标、可替换适配器以及排行榜 JSON 输出。
version: 1.0.0
phase: 19
lesson: 49
tags: [evaluation, metrics, leaderboard, harness]
---

## 适用场景

在固定任务集上比较两个模型、两个检查点或两个提示模板。任何要上线且需要长期监控的内容。

## 任务规格

每条示例一行 JSONL：

```json
{"id": "ex-001", "prompt": "...", "targets": ["..."], "metric": "exact_match", "extras": {}}
```

同一文件中的所有示例共享一个指标。文件名即为任务名。

## 指标

| 指标 | 签名 | 用途 |
|--------|-----------|---------|
| exact_match | 归一化小写 + 空白后判等 | 算术、事实类答案 |
| substring_contains | 目标须出现在归一化后的预测中 | 带锚词的自由生成 |
| multiple_choice | 首字母匹配 | A/B/C/D 式选择题 |
| rouge_l | 基于分词文本的 LCS F1 | 摘要、改写 |
| code_exec | 在 io_pairs 上运行预测的 `f`，计数匹配项 | 代码生成 |

所有指标返回 [0.0, 1.0] 范围内的浮点数。任务得分为均值。

## 适配器

```python
class Adapter(Protocol):
    name: str
    def generate(self, prompts: list[str]) -> list[str]: ...
```

适配器是唯一与模型相关的代码。

## 排行榜 JSON

包含 schema 字符串、时间戳、各任务得分与延迟、总体均值。比较运行时需包含逐条示例记录，以便预测级别的回归可见。

## 故障模式

- 指标返回值超出 [0, 1]：总体得分变得不可解读。
- 单个任务文件中混用指标：断言触发；每个文件保持一个指标。
- code_exec 未限制命名空间：任意代码执行。
- 缺少 schema 字符串：格式演进会破坏下游仪表盘。
