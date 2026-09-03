---
name: state-schema
description: 为智能体状态和任务看板生成项目专属 JSON Schema、带有原子写入的 Python StateManager，以及迁移脚手架，确保 schema 升级不会损坏工作台。
version: 1.0.0
phase: 14
lesson: 34
tags: [state, schema, json-schema, atomic-writes, migrations]
---

给定一个仓库以及运行其中的智能体产品，为工作台生成 schema 优先的状态文件。

产出：

1. `schemas/agent_state.schema.json`，覆盖必填键、允许的状态值、数组与 null 的纪律，以及一个 `schema_version` 整数。
2. `schemas/task_board.schema.json`，覆盖任务 id 模式、允许的归属人、允许的状态，以及验收数组。
3. `tools/state_manager.py`，暴露 `load`、`commit` 和 `update`，采用临时文件加重命名的原子写入方式。
4. `tools/migrate_state.py`，作为下一次 schema 升级的脚手架，如果文件来自未知版本则大声失败。
5. `agent_state.json` 和 `task_board.json`，以 `schema_version: 1` 和全新的待办列表进行播种。

硬性拒绝：

- 没有 `schema_version` 字段的 schema。迁移不是可选项。
- 在期望数组的地方允许 `null`。`null` 是伪装成数据的写入期 bug。
- 使用普通 `open(path, "w")` 的写入器。仅允许原子写入；不完整的文件会损坏事实来源。
- 在状态中存储 token、原始聊天记录或 PII。状态仅用于与仓库相关的事实。

拒绝规则：

- 如果仓库没有版本控制，拒绝交付状态文件。原子写入加 git diff 是持久性的保障方式。
- 如果项目没有至少一个验收命令来校验 `done` 状态转换，拒绝 `status: done` 枚举值。在没有验收检查的情况下添加 `done` 只是形式主义。
- 如果项目打算在没有锁策略的情况下跨进程共享状态，在交付之前提出该发现；原子重命名是必要但不充分的。

输出结构：

```
<repo>/
├── agent_state.json
├── task_board.json
├── schemas/
│   ├── agent_state.schema.json
│   └── task_board.schema.json
└── tools/
    ├── state_manager.py
    └── migrate_state.py
```

以"接下来阅读什么"结尾，指向：

- 第 35 课，关于在启动时调用该管理器的初始化脚本。
- 第 38 课，关于读取状态来评分完成度的验证门。
- 第 40 课，关于消费同一 schema 的交接生成器。
