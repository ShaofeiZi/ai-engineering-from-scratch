---
name: verification-gate
description: 生成一个确定性验证门控，将范围、规则和反馈产物组合为每个任务的单一 verification_report.json，并附带在缺少通过裁定时拒绝合并的 CI 配置。
version: 1.0.0
phase: 14
lesson: 38
tags: [verification, gate, deterministic, ci, override-log]
---

给定项目的验收标准和现有工作台产物，产出验证门控和覆盖审计日志。

产出内容：

1. `tools/verify_agent.py`，暴露 `verify(task_id, artifacts) -> VerdictReport`。纯函数，确定性，不调用任何 LLM。
2. `outputs/verification/<task_id>.json`，作为裁定的唯一事实来源。
3. `tools/override.py`，向 `outputs/verification/overrides.jsonl` 追加已签名的覆盖条目（必须包含原因、用户 ID、时间戳、发现代码）。
4. CI 工作流，在 `passed: false` 时失败并内联展示报告。
5. `docs/verification.md`，列出每一项检查及其严重级别、来源产物和覆盖策略。

硬性拒绝：

- 调用 LLM 的检查。门控是确定性管道；LLM 判断属于评审者的职责。
- 智能体可在无签名条目情况下使用的覆盖路径。覆盖仅限人工操作。
- 省略其消费的产物路径的验证报告。报告必须可审计。
- 工作流可静默降级的 block 严重级别发现。严重级别在写入时固定，而非在读取时。

拒绝规则：

- 如果项目没有验收命令，则拒绝交付门控，直到存在验收命令。一个什么也不证明的门控只是摆设。
- 如果规则报告不存在，则拒绝跳过规则检查；失败时封闭。
- 如果反馈日志不存在，则拒绝跳过验收检查；缺失的日志本身就是 block 级别发现。
- 如果覆盖条目未纳入版本控制，则拒绝接入覆盖路径；未记录的覆盖会使门控形同虚设。

输出结构：

```
<repo>/
├── tools/
│   ├── verify_agent.py
│   └── override.py
├── outputs/verification/
│   ├── overrides.jsonl
│   └── <task_id>.json
├── docs/verification.md
└── .github/workflows/verify.yml
```

以"接下来读什么"结尾，指向：

- 第 39 课，关于在通过裁定后接手的评审者智能体。
- 第 40 课，关于将裁定包含在交接包中的交接生成器。
- 第 41 课，关于针对真实风格示例应用运行门控。
