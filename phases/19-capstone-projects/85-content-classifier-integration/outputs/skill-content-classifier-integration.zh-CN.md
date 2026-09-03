---
name: skill-content-classifier-integration
description: 三个输出侧分类器（毒性、PII、指令泄露），由单一严重度路由器统一调度，支持 block、redact、warn、log 四种动作
version: 1.0.0
phase: 19
lesson: 85
tags: [safety, classifier, output-filter]
---

# 内容分类器集成

三个分类器，一个路由器，四种动作。

## 判定结果结构

```text
ClassifierVerdict
  name: str
  severity: none | low | medium | high
  score: float in [0, 1]
  findings: list[str]
```

## 动作表

| 严重度 | 动作 | 效果 |
|---|---|---|
| high | block | 输出替换为策略拒绝消息 |
| medium | redact | 按分类器顺序依次应用脱敏器 |
| low | warn | 输出照常发送，追加软提示 |
| none | log | 输出照常发送，记录判定结果 |

## 各分类器行为

- toxicity - 以空白符为边界匹配骚扰用语，并做小窗口左向否定检查；脱敏为 `[redacted-language]`
- pii - 电子邮件、电话号码、SSN、Luhn 校验的银行卡、IPv4；SSN 和银行卡触发严重度升级；将每种形态分别脱敏为标签
- instruction-leakage - 与已知系统提示词做三元组余弦相似度对比；严重度随重叠度递增；脱敏系统提示词的第一行

## 产物

`outputs/classifier_report.json` 包含每个用例的动作动词、严重度、脱敏后输出及完整判定列表。
