---
name: skill-end-to-end-safety-gate
description: 三检查点安全闸，组合输入检测器、流式 token 过滤器、输出分类器和规则引擎，附带确定性聚合表和逐请求追踪
version: 1.0.0
phase: 19
lesson: 87
tags: [safety, harness, composition]
---

# 端到端安全闸

## 生命周期

1. pre-gen - 对提示词运行课程 83 检测器
   - 若 confidence >= block_threshold：返回拒绝，发出追踪记录，停止
2. during-gen - 从模型流式获取输出，缓冲两个 chunk，扫描已知有害续写
   - 若匹配：终止迭代器，标记追踪记录，视为 medium 严重度
3. post-gen - 若未提前终止，对完整输出运行课程 85 分类器路由器和课程 86 规则引擎
4. aggregate - 取 pre、during、post.classifier、post.rules 中的最大严重度
5. apply - 映射为 block、redact、warn 或 allow

## 聚合表

| 信号状态 | 动作 |
|---|---|
| 任一为 high 严重度 | block |
| 任一为 medium 严重度 | redact |
| 任一为 low 严重度 | warn |
| 无任何信号 | allow |

## 追踪结构

```text
RequestTrace
  request_id: str
  prompt: str
  pre_gen: { category, confidence, fired[] }
  during_gen: { terminated_early, matched_pattern, partial_chunks }
  post_gen: { classifier_action, classifier_severity, rules_max_severity, rules_violations[] } | null
  final_action: block | redact | warn | allow
  final_output: str
  latency_ms: float
```

## 产物

`outputs/gate_trace.json` 包含汇总信息和逐请求追踪，涵盖 50 条分类法测试用例和 10 条良性提示词。
