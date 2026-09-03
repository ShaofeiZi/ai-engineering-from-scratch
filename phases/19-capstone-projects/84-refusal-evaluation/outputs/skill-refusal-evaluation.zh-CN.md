---
name: skill-refusal-evaluation
description: 双向拒绝指标，包含校准和按类别的细分，基于标注好的提示词语料库
version: 1.0.0
phase: 19
lesson: 84
tags: [safety, evaluation, calibration]
---

# 拒绝评估

一个包含安全与不安全提示词的标注语料库，经过一条或多条模型策略处理。输出被分类为拒绝或回答。该框架返回以下指标：

- under-refusal：被回答但标注为不安全的提示词数 / 不安全总数
- over-refusal：被拒绝但标注为安全的提示词数 / 安全总数
- accuracy：(正确拒绝数 + 正确回答数) / 总数
- ECE：按声明的置信度分箱计算的期望校准误差
- per-category under-refusal：与课程 82 分类法联接后按类别的细分

## 接入真实模型

模拟 LLM 是一个 `(prompt: str) -> str` 的可调用对象。将其替换为返回模型输出并嵌入置信度标签的 HTTP 封装器（或修改 `parse_confidence` 以读取你所用服务提供商暴露的字段）。其余部分保持不变。

## 产物

`outputs/refusal_eval_report.json` 包含按策略的指标。课程 87 读取该报告以设定阈值。
