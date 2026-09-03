---
name: scope-contract
description: 生成针对每个任务的范围契约，包含允许/禁止的 glob 模式、验收标准和回滚计划，并提供一个可在每次智能体 diff 上运行的、CI 就绪的 glob 感知检查器。
version: 1.0.0
phase: 14
lesson: 36
tags: [scope, contract, globs, diff-check, ci]
---

给定一个任务描述和一个仓库布局，生成一个范围契约和一个 diff 感知检查器。

需要产出：

1. 针对该任务的 `scope_contract.json`，包含以下字段：`task_id`、`goal`、`allowed_files`（glob 模式）、`forbidden_files`（glob 模式）、`acceptance_criteria`、`rollback_plan`、`approvals_required`。
2. `tools/scope_check.py`，接收一个契约路径和一个被修改文件列表，返回一个 `ScopeReport`，并在出现任何违规时以非零退出码退出。
3. CI 步骤（`.github/workflows/scope-check.yml` 或等效配置），对合并 diff 运行该检查器。
4. `outputs/scope/closed/<task_id>.json` 归档约定，使契约随变更历史一同交付。

硬性拒绝：

- 没有 `forbidden_files` 的契约。负空间是契约的一部分。
- 对代码目录使用原始路径而非 glob 模式的契约。重构会在一夜之间使原始路径失效。
- `rollback_plan` 字段为空或为 "see runbook" 的。必须详细写明。
- 审批列为 "case by case" 的。审批边界必须是可枚举的。

拒绝规则：

- 如果任务描述未约束仓库的某个区域，则不得仅凭描述自行编写 `allowed_files`。需询问该任务所在的目录。
- 如果仓库没有测试命令，则在提供或创建测试桩之前，拒绝添加 `acceptance_criteria`。无法验证的契约只是一厢情愿。
- 如果智能体运行时无法遵守审批边界（没有 human-in-the-loop），则在交付前暴露该差距；范围蔓延至需要审批的操作将成为主要失败原因。

输出结构：

```
<repo>/
├── scope_contract.json
├── outputs/scope/closed/
│   └── T-XXX.json
├── tools/
│   └── scope_check.py
└── .github/
    └── workflows/
        └── scope-check.yml
```

最后以 "延伸阅读" 结尾，指向：

- 第 37 课，了解将运行的命令与契约关联的运行时反馈。
- 第 38 课，了解消费范围报告的验证门禁。
- 第 39 课，了解审计已关闭契约归档的审查者智能体。
