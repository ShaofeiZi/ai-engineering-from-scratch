---
name: minimal-workbench
description: 为任意仓库铺设三文件最小可行智能体工作台 —— 简短的 AGENTS.md 路由、持久化的 agent_state.json，以及键映射到项目当前 backlog 的 JSON task_board.json。
version: 1.0.0
phase: 14
lesson: 32
tags: [workbench, agents-md, state, task-board, scaffold]
---

给定一个仓库路径和一份简短 backlog，搭建最小可行的智能体工作台。

产出：

1. `AGENTS.md` 不超过 80 行。它必须路由到：状态文件、任务看板、更详细的规则文档（即使为空），以及验证命令。此文件中不得包含散文式教程。
2. `agent_state.json` 包含以下键：`active_task_id`、`touched_files`、`assumptions`、`blockers`、`next_action`。所有可选字段默认为空数组或空字符串，数组绝不为 `null`。
3. `task_board.json` 为一个 JSON 任务数组。每个任务包含 `id`、`goal`、`owner`（`builder` | `reviewer` | `human`）、`acceptance`（字符串列表）和 `status`（`todo` | `in_progress` | `done` | `blocked`）。
4. `docs/agent-rules.md` 占位文件，每个方面使用一个 H2 标题，以便后续课程填充内容。

硬性拒绝条件：

- `AGENTS.md` 超过 80 行或不足 10 行。太长则智能体会跳过它；太短则它不承载任何路由。
- 状态文件引用聊天记录而非仓库。仓库才是事实来源。
- 任务看板缺少 `acceptance`。没有验收标准的任务会沦为“看起来不错”的橡皮图章。
- 任务的 `owner` 为 `agent` 或 `model`。拥有者是角色，而非实体。

拒绝规则：

- 如果仓库没有验证命令，拒绝编写 `AGENTS.md`，直到提供或占位一个验证命令。指向缺失闸门的路由比没有路由更糟。
- 如果 backlog 中有超过 12 个未完成任务，拒绝并要求用户拆分。超过一屏的看板会退化为规划表演。
- 如果项目在受跟踪文件中包含密钥，拒绝编写状态文件，并首先将密钥泄露作为阻断性发现报告。

输出结构：

```
<repo>/
├── AGENTS.md
├── agent_state.json
├── task_board.json
└── docs/
    └── agent-rules.md
```

以“接下来阅读什么”结尾，指向：

- 第 33 课，了解如何将规则占位文件转化为可执行约束。
- 第 34 课，了解持久化状态模式。
- 第 36 课，了解每个任务的范围契约。
